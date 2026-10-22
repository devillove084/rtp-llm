"""Two-GPU NCCL regression for the native PyWrappedModel CP capture boundary.

Only layer outputs and the external Mooncake store are synthetic. CP planning,
restoration, publication, and every transport callback run their real paths.
This manual target requires two CUDA GPUs and fails rather than skipping when
that prerequisite is missing. It uses BF16 storage, not the TorchSpec quantizer.
"""

import os
import sys
import time
import types
import unittest
from datetime import timedelta
from itertools import product
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.cpp.models.test import (
    pywrapped_model_cache_store_integration_test as capture_fixture,
)
from rtp_llm.test.utils.port_util import PortManager

WORLD_SIZE = 2
NCCL_TIMEOUT_SECONDS = 60
WORKER_TIMEOUT_SECONDS = 300
CAPTURE_LAYERS = (1, 0)


class _NativeNcclTransport:
    # Values are the C++ ReduceOp / ParallelMode enum values in OpData.h.
    REDUCE_OPS = {0: dist.ReduceOp.SUM, 3: dist.ReduceOp.MIN}
    DTYPES = {torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32}

    def __init__(self, checks, rank):
        self.checks = checks
        self.device = torch.device("cuda", rank)
        self.group = dist.group.WORLD
        self.reset()
        # Register with the same native registry that PyWrappedModel calls,
        # not the registry of a separately imported production ops extension.
        checks.ext.register_comm_ops(self.broadcast, self.allreduce, self.allgather)

    def reset(self):
        self.events = []
        self.reductions = []

    def check_tensor(self, tensor, mode):
        self.checks.assertEqual(mode, 0)  # TP == the two-rank WORLD group here.
        self.checks.assertEqual(tensor.device, self.device)
        self.checks.assertIn(tensor.dtype, self.DTYPES)
        self.checks.assertTrue(tensor.is_contiguous())

    def broadcast(self, tensors, root, mode):
        self.checks.assertIn(root, range(WORLD_SIZE))
        for tensor in tensors:
            self.check_tensor(tensor, mode)
            dist.broadcast(tensor, src=root, group=self.group)
        self.events.append("broadcast")

    def allreduce(self, tensor, op, mode, dest):
        self.check_tensor(tensor, mode)
        self.checks.assertEqual(tensor.dtype, torch.int32)
        self.checks.assertIn(op, self.REDUCE_OPS)
        local = tensor.tolist()
        target = tensor if dest is None else dest
        if dest is not None:
            self.check_tensor(dest, mode)
            self.checks.assertEqual(dest.dtype, tensor.dtype)
            self.checks.assertEqual(dest.shape, tensor.shape)
            dest.copy_(tensor)
        dist.all_reduce(target, op=self.REDUCE_OPS[op], group=self.group)
        self.reductions.append((op, local, target.tolist()))
        self.events.append("sum" if op == 0 else "min")
        return target

    def allgather(self, receive, mode, send, inplace):
        self.checks.assertFalse(inplace)
        self.checks.assertEqual(len(receive), len(send))
        for output, tensor in zip(receive, send):
            self.check_tensor(output, mode)
            self.check_tensor(tensor, mode)
            self.checks.assertEqual(output.dtype, tensor.dtype)
            self.checks.assertEqual(
                tuple(output.shape), (WORLD_SIZE * tensor.shape[0], *tensor.shape[1:])
            )
            dist.all_gather_into_tensor(output, tensor, group=self.group)
        self.events.append("gather")


class _RankChecks(capture_fixture.CaptureFixture):
    """Reuse the existing model/store fixture without loading its test suite."""

    def run_success(self, transport, rank, lengths, cuda_ids, capture=True, mtp=False):
        device = torch.device("cuda", rank)
        ids = torch.arange(1, sum(lengths) + 1, dtype=torch.int32, device=device)
        input_ids = ids if cuda_ids else ids.cpu()
        split = self.ext.split_cp_inputs(
            input_ids, lengths, WORLD_SIZE, rank, local_device_id=rank
        )
        self.assertEqual(self.ext.get_device_id(), rank)
        self.assertEqual(split["input_ids"].device, device)
        self.assertEqual(split["original_lengths"].tolist(), lengths)
        model = self.model(mtp=mtp, capture_layer_ids=CAPTURE_LAYERS)
        if not mtp:
            # GLM/GenericMoe has no DSv4 pre-norm getter.
            del type(model).get_mtp_target_hidden_states
        stores_before = len(capture_fixture.MockStore.instances)
        transport.reset()
        # run_model constructs a fresh native model/publisher on every call.
        result = self.ext.run_model(
            model,
            cp=True,
            tp_size=WORLD_SIZE,
            tp_rank=rank,
            local_device_id=rank,
            lengths=lengths,
            input_ids_on_cuda=cuda_ids,
            capture_layer_ids=CAPTURE_LAYERS,
            capture=capture,
            fp8=False,
        )
        self.assertEqual(model.flags, [capture])
        torch.testing.assert_close(model.token_ids[0], split["input_ids"])
        self.assertIsNone(result["error"])
        self.assertIsNone(result["second_error"])
        self.assertEqual(result["failures"], 0)
        self.assertEqual(result["broken_rejections"], 0)
        expected_events = (["sum"] if capture else []) + ["gather"] * (1 + int(mtp))
        self.assertEqual(transport.events, expected_events)
        if capture:
            self.assertEqual(
                transport.reductions,
                [
                    (
                        0,
                        [1, 1, int(mtp)],
                        [WORLD_SIZE, WORLD_SIZE, WORLD_SIZE * int(mtp)],
                    )
                ],
            )
        else:
            self.assertEqual(transport.reductions, [])
        hidden = ids.to(torch.float16).unsqueeze(1).expand(-1, 2)
        expected = hidden + (100 if mtp else 30 if capture else 0)
        self.assertEqual(result["hidden"].device, device)
        torch.testing.assert_close(result["hidden"], expected)
        stores = capture_fixture.MockStore.instances[stores_before:]
        self.assertEqual(len(stores), int(rank == 0))
        if rank == 0:
            store = stores[0]
            self.assertTrue(store.closed)
            self.assertEqual(store.device, device)
            self.assertEqual(len(store.batches), int(capture))
            if capture:
                batch = store.batches[0]
                self.assertEqual(
                    batch["request_ids"],
                    [f"rtp_GlmBase_{100 + i}" for i in range(len(lengths))],
                )
                offset = 0
                for index, length in enumerate(lengths):
                    request_hidden = hidden[offset : offset + length]
                    torch.testing.assert_close(
                        batch["input_ids"][index], ids[offset : offset + length].long()
                    )
                    torch.testing.assert_close(
                        batch["hidden_states"][index],
                        torch.cat(
                            [
                                request_hidden + (layer + 1) * 10
                                for layer in CAPTURE_LAYERS
                            ],
                            dim=-1,
                        ).to(torch.bfloat16),
                    )
                    torch.testing.assert_close(
                        batch["last_hidden_states"][index],
                        (request_hidden + 30).to(torch.bfloat16),
                    )
                    offset += length
                for key in ("input_ids", "hidden_states", "last_hidden_states"):
                    self.assertEqual([value.shape[0] for value in batch[key]], lengths)
        # Boundaries prevent a slow owner's store assertions from leaking into
        # the next case, without substituting any model collective.
        dist.barrier(group=dist.group.WORLD, device_ids=[rank])

    def run_rejection(self, transport, rank, lengths, cuda_ids, fault):
        packed_fault = fault.startswith("packed_")
        getter_fault = None if packed_fault else fault
        model = self.model(
            mtp=not (rank == 1 and fault == "absent"),
            malformed=(
                fault.removeprefix("packed_") if rank == 1 and packed_fault else None
            ),
            mtp_failure=getter_fault if rank == 1 else None,
            capture_layer_ids=CAPTURE_LAYERS,
        )
        if rank == 1 and fault == "no_getter":
            del type(model).get_mtp_target_hidden_states
        stores_before = len(capture_fixture.MockStore.instances)
        transport.reset()
        result = self.ext.run_model(
            model,
            cp=True,
            tp_size=WORLD_SIZE,
            tp_rank=rank,
            local_device_id=rank,
            lengths=lengths,
            input_ids_on_cuda=cuda_ids,
            capture_layer_ids=CAPTURE_LAYERS,
            fp8=False,
        )
        if packed_fault:
            bad_status = [0, 1, 0]
        elif fault in ("absent", "empty", "no_getter"):
            bad_status = [1, 1, 0]
        else:
            bad_status = [1, 0, int(fault != "exception")]
        good_status = [1, 1, 1]
        expected_sum = [good + bad for good, bad in zip(good_status, bad_status)]
        # Both real ranks observe the same rejection. Neither may enter the
        # MTP or packed all-gather; the owner alone holds the deferred error.
        self.assertEqual(transport.events, ["sum"])
        self.assertEqual(
            transport.reductions,
            [(0, good_status if rank == 0 else bad_status, expected_sum)],
        )
        self.assertEqual(result["hidden"].device, torch.device("cuda", rank))
        torch.testing.assert_close(
            result["hidden"],
            torch.zeros(
                (sum(lengths), 2),
                dtype=torch.float16,
                device=torch.device("cuda", rank),
            ),
        )
        self.assertIsNone(result["second_error"])
        if lengths == [1, 3]:
            self.assertEqual(model.token_ids[0].numel(), sum(lengths))
        stores = capture_fixture.MockStore.instances[stores_before:]
        self.assertEqual(len(stores), int(rank == 0))
        if rank == 0:
            self.assertIsNotNone(result["error"])
            self.assertIn(
                (
                    "presence differs"
                    if fault in ("absent", "empty", "no_getter")
                    else "another TP rank"
                ),
                result["error"],
            )
            self.assertEqual(result["failures"], 1)
            self.assertEqual(stores[0].batches, [])
            self.assertTrue(stores[0].closed)
        else:
            self.assertIsNone(result["error"])
            self.assertEqual(result["failures"], 0)
        dist.barrier(group=dist.group.WORLD, device_ids=[rank])
        # Hard layout errors break the old publisher. Do not pretend it recovers:
        # prove the communicator continues with a NEW model/publisher instance.
        self.run_success(transport, rank, lengths, cuda_ids, capture=True, mtp=True)


def _nccl_worker(rank, port):
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    checks = None
    try:
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=WORLD_SIZE,
            timeout=timedelta(seconds=NCCL_TIMEOUT_SECONDS),
            device_id=device,
        )
        _RankChecks.setUpClass()
        checks = _RankChecks()
        # No TorchSpec installation is needed for this BF16 target. Reuse the
        # existing Mooncake fake, supplying only its absent parent packages.
        torchspec = types.ModuleType("torchspec")
        torchspec.__path__ = []
        transfer = types.ModuleType("torchspec.transfer")
        transfer.__path__ = []
        torchspec.transfer = transfer
        with patch.dict(
            sys.modules, {"torchspec": torchspec, "torchspec.transfer": transfer}
        ):
            try:
                checks.setUp()
                transfer.mooncake = sys.modules["torchspec.transfer.mooncake"]
                transport = _NativeNcclTransport(checks, rank)
                for lengths, cuda_ids in product(([6, 9], [1, 3]), (False, True)):
                    for capture, mtp in ((False, False), (True, False), (True, True)):
                        checks.run_success(
                            transport, rank, lengths, cuda_ids, capture, mtp
                        )
                    for fault in (
                        "packed_width",
                        "packed_dtype",
                        "exception",
                        "rows",
                        "rank",
                        "width",
                        "dtype",
                        "device",
                        "absent",
                        "empty",
                        "no_getter",
                    ):
                        checks.run_rejection(transport, rank, lengths, cuda_ids, fault)
                # The once-initialized fixture must reject another local device,
                # not silently route a later call to GPU0.
                with checks.assertRaisesRegex(
                    RuntimeError, "different local_device_id"
                ):
                    checks.ext.split_cp_inputs(
                        torch.tensor([1], dtype=torch.int32),
                        [1],
                        WORLD_SIZE,
                        rank,
                        local_device_id=1 - rank,
                    )
                checks.assertEqual(checks.ext.get_device_id(), rank)
                torch.cuda.synchronize(device)
                dist.barrier(group=dist.group.WORLD, device_ids=[rank])
                print(f"CP_CAPTURE_NCCL_PASS rank={rank}", flush=True)
            finally:
                if not checks.doCleanups():
                    raise RuntimeError(f"rank {rank}: capture fixture cleanup failed")
    except BaseException:
        print(f"CP_CAPTURE_NCCL_FAIL rank={rank}", flush=True)
        raise  # torch.multiprocessing.spawn propagates the worker traceback.
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class PyWrappedModelCPCaptureTest(unittest.TestCase):
    def test_two_gpu_nccl_capture_and_rank_local_rejection(self):
        self.assertTrue(dist.is_nccl_available(), "this manual target requires NCCL")
        self.assertGreaterEqual(
            torch.cuda.device_count(),
            WORLD_SIZE,
            "this manual target requires two visible CUDA GPUs",
        )
        ports, locks = PortManager().get_consecutive_ports(1)
        context = None
        try:
            context = mp.spawn(
                _nccl_worker, args=(ports[0],), nprocs=WORLD_SIZE, join=False
            )
            deadline = time.monotonic() + WORKER_TIMEOUT_SECONDS
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.fail(
                        f"CP capture NCCL workers exceeded {WORKER_TIMEOUT_SECONDS}s"
                    )
                if context.join(timeout=min(1.0, remaining)):
                    break
        finally:
            # Like user_buffers_test: reap every child before releasing ports,
            # including the peer of a failed worker. Limit forced cleanup too.
            try:
                if context is not None:
                    for process in context.processes:
                        if process.is_alive():
                            process.terminate()
                    for process in context.processes:
                        process.join(timeout=10)
                        if process.is_alive():
                            process.kill()
                            process.join(timeout=10)
            finally:
                for lock in locks:
                    lock.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
