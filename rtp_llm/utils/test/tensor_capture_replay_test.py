"""Model-free replay tests. CPU tests never skip.

CPU only:
  /opt/conda310/bin/python3 -m unittest rtp_llm.utils.test.tensor_capture_replay_test.TensorCaptureReplayTest -v
Real CUDA (explicit opt-in; missing GPU/library FAILS, never passes/skips):
  RTP_LLM_TEST_CAPTURE_REPLAY_CUDA=1 /opt/conda310/bin/python3 -m unittest rtp_llm.utils.test.tensor_capture_replay_test.TensorCaptureReplayCudaTest -v
Temporary bundles live in the checkout wrapper's local-lab, not /tmp or the repo.
For Bazel/runfiles, set RTP_LLM_TEST_LOCAL_LAB to an existing local-lab directory.
"""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from rtp_llm.utils import tensor_capture as capture
from rtp_llm.utils import tensor_capture_replay as replay


class _ReplayFixture(unittest.TestCase):
    def setUp(self):
        lab = Path(
            os.environ.get(
                "RTP_LLM_TEST_LOCAL_LAB",
                Path(__file__).resolve().parents[4] / "local-lab",
            )
        )
        self.temporary = tempfile.TemporaryDirectory(
            prefix="tensor-replay-test-", dir=lab
        )
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.serial = 0

    def bundle(
        self,
        logits=None,
        starts=None,
        ends=None,
        indices=None,
        backend="dsv4_per_row",
        **params
    ):
        # Exercise the real raw serializer/loader. Synthetic bounds (including
        # length zero) also cover legal operator inputs not emitted by prefill.
        if logits is None:
            logits = torch.tensor([[0.0, -3.0, -1.0, -2.0, 0.0]])
        if starts is None:
            starts = torch.tensor([1], dtype=torch.int32)
        if ends is None:
            ends = torch.tensor([4], dtype=torch.int32)
        if indices is None:
            indices = torch.tensor([[2, 1]], dtype=torch.int32)
        rows, k = indices.shape
        parameters = dict(
            k=k,
            index_space="relative_to_row_start",
            stride0=logits.stride(0),
            stride1=logits.stride(1),
        )
        if backend == "topk_v3_tie_break":
            parameters["max_seq_len"] = logits.shape[1]
        else:
            parameters.update(num_rows=rows, force_radix_sort=False)
        parameters.update(params)
        manifest = dict(
            schema_version=1,
            complete=True,
            op="indexer.topk",
            phase="prefill",
            backend=backend,
            parameters=parameters,
            query_row_start=0,
            total_query_rows=rows,
            query_row_count=rows,
            invalid_logits="zero_filled_snapshot_only",
            finite_values_checked=False,
            nonfinite_policy="valid_domain_bits_preserved_backend_semantics",
            output_contract="raw_relative_indices_minus_one_padding_backend_order",
            tensors={},
        )
        directory = self.root / ("bundle-%d" % self.serial)
        self.serial += 1
        directory.mkdir()
        tensors = dict(logits=logits, row_starts=starts, row_ends=ends, indices=indices)
        for name, tensor in tensors.items():
            desc, data = capture._encode_tensor(tensor)
            desc["file"] = name + ".bin"
            manifest["tensors"][name] = desc
            (directory / desc["file"]).write_bytes(data)
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest))
        return path

    def mutate(self, path, change):
        manifest = json.loads(path.read_text())
        change(manifest)
        path.write_text(json.dumps(manifest))

    def boundary_bundle(self, backend, k, force=False):
        rows, width = 4, k + 8
        stride1 = 1 if backend == "topk_v3_tie_break" else 2
        storage = torch.zeros(rows * (width * stride1 + 8) + 3)
        logits = storage.as_strided((rows, width), (width * stride1 + 8, stride1), 3)
        starts = torch.tensor([2, 3, 1, 4], dtype=torch.int32)
        lengths = torch.tensor([0, 1, k, k + 1], dtype=torch.int32)
        ends = starts + lengths
        output = torch.full((rows, k), -1, dtype=torch.int32)
        output[1, 0] = 0
        output[2] = torch.arange(k)
        # All scores tie: smaller relative indices win, but long-row order is free.
        output[3] = torch.arange(k - 1, -1, -1)
        params = dict(force_radix_sort=force) if backend == "dsv4_per_row" else {}
        return self.bundle(logits, starts, ends, output, backend, **params)


class TensorCaptureReplayTest(_ReplayFixture):
    def test_cpu_is_contract_validation_and_does_not_import_ops(self):
        with mock.patch.object(
            replay.importlib, "import_module", side_effect=AssertionError("ops import")
        ), mock.patch.object(
            torch.cuda, "is_available", side_effect=AssertionError("CUDA probe")
        ), mock.patch.object(
            torch, "topk", side_effect=AssertionError("not an oracle")
        ):
            result = replay.replay_capture(self.bundle())
        self.assertFalse(result["kernel_replayed"])
        self.assertEqual(result["mode"], "cpu_contract_validation")
        self.assertEqual(result["counts"]["selected_indices"], 2)

    def test_lengths_zero_short_equal_long_and_nonzero_starts(self):
        for backend, k in (
            ("dsv4_per_row", 1),
            ("dsv4_per_row", 3),
            ("topk_v3_tie_break", 512),
            ("topk_v3_tie_break", 1024),
            ("topk_v3_tie_break", 2048),
        ):
            with self.subTest(backend=backend, k=k):
                result = replay.replay_capture(self.boundary_bundle(backend, k))
                self.assertEqual(result["counts"]["empty_rows"], 1)
                self.assertEqual(result["counts"]["selected_indices"], 1 + 2 * k)
                self.assertEqual(result["counts"]["boundary_tie_rows"], 1)

    def test_k_larger_than_width(self):
        result = replay.replay_capture(
            self.bundle(
                torch.zeros(1, 2),
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([2], dtype=torch.int32),
                torch.tensor([[0, 1, -1, -1]], dtype=torch.int32),
            )
        )
        self.assertEqual(result["counts"]["padding_indices"], 2)

    def test_duplicate_missing_out_of_range_negative_and_suboptimal(self):
        for indices in ([[1, 1]], [[1, -1]], [[1, 3]], [[1, -2]], [[0, 1]]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                replay.replay_capture(
                    self.bundle(indices=torch.tensor(indices, dtype=torch.int32))
                )

    def test_padding_and_short_row_order(self):
        for indices in ([[0, -2]], [[-1, 0]], [[0, 0]]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                replay.replay_capture(
                    self.bundle(
                        torch.zeros(1, 1),
                        torch.tensor([0], dtype=torch.int32),
                        torch.tensor([1], dtype=torch.int32),
                        torch.tensor(indices, dtype=torch.int32),
                    )
                )
        with self.assertRaisesRegex(ValueError, "sequential"):
            replay.replay_capture(
                self.bundle(
                    torch.zeros(1, 2),
                    torch.tensor([0], dtype=torch.int32),
                    torch.tensor([2], dtype=torch.int32),
                    torch.tensor([[1, 0]], dtype=torch.int32),
                )
            )

    def test_ties_use_smaller_index_not_output_order(self):
        logits = torch.tensor([[9.0, 7.0, 7.0, 7.0, -1.0]])
        for indices, valid in (([1, 0], True), ([0, 2], False), ([1, 2], False)):
            path = self.bundle(
                logits,
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([5], dtype=torch.int32),
                torch.tensor([indices], dtype=torch.int32),
            )
            if valid:
                self.assertEqual(replay.replay_capture(path)["status"], "validated")
            else:
                with self.assertRaisesRegex(ValueError, "maximal candidate"):
                    replay.replay_capture(path)

    def test_dsv4_current_source_ties_not_legacy_arrival_slot_order(self):
        # The pre-stabilization insertion branch used candidate-slot i < j,
        # not token indices. Both current dispatches require smaller indices.
        cases = (
            ([0.0] * 4, [2, 1, 0], [3, 2, 1]),
            ([-7.0] * 4, [2, 1, 0], [3, 2, 1]),
            ([-0.0, 0.0, -0.0, 0.0], [3, 1, 0], [3, 1, 2]),
            ([0.0] * 2049, [2, 1, 0], [2048, 2, 1]),
        )
        for force in (False, True):
            for values, expected, legacy in cases:
                with self.subTest(force=force, length=len(values), prefix=values[:4]):
                    for indices, valid in ((expected, True), (legacy, False)):
                        path = self.bundle(
                            torch.tensor([values]),
                            torch.tensor([0], dtype=torch.int32),
                            torch.tensor([len(values)], dtype=torch.int32),
                            torch.tensor([indices], dtype=torch.int32),
                            force_radix_sort=force,
                        )
                        if valid:
                            replay.replay_capture(path)
                        else:
                            with self.assertRaisesRegex(
                                ValueError, "index tie-break.*provenance"
                            ):
                                replay.replay_capture(path)

    def test_signed_zero_and_infinities(self):
        logits = torch.tensor([[-float("inf"), -0.0, 0.0, float("inf")]])
        for indices, valid in (([2, 3], True), ([1, 3], False), ([0, 3], False)):
            path = self.bundle(
                logits,
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([4], dtype=torch.int32),
                torch.tensor([indices], dtype=torch.int32),
            )
            if valid:
                replay.replay_capture(path)
            else:
                with self.assertRaises(ValueError):
                    replay.replay_capture(path)

    def test_nan_explicitly_unsupported_not_silently_passed(self):
        for bits in (0x7FC00001, -0x003FFFFF):
            logits = torch.tensor([[0.0, 1.0, 0.0]])
            logits.view(torch.int32)[0, 1] = bits
            with self.subTest(bits=bits), self.assertRaisesRegex(
                replay.UnsupportedCapture, "NaN"
            ):
                replay.replay_capture(
                    self.bundle(
                        logits,
                        torch.tensor([1], dtype=torch.int32),
                        torch.tensor([2], dtype=torch.int32),
                        torch.tensor([[0]], dtype=torch.int32),
                    )
                )

    def test_v3_effective_length_clamped_by_saved_max_seq_len(self):
        logits = torch.zeros(1, 520)
        logits[0, 514:] = 100
        output = torch.full((1, 512), -1, dtype=torch.int32)
        output[0, :2] = torch.tensor([0, 1])
        path = self.bundle(
            logits,
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([520], dtype=torch.int32),
            output,
            "topk_v3_tie_break",
            max_seq_len=2,
        )
        self.assertEqual(replay.replay_capture(path)["counts"]["selected_indices"], 2)

    def test_bounds_and_invalid_domain_zero_policy(self):
        for starts, ends in (([-1], [4]), ([3], [2]), ([0], [6])):
            with self.subTest(starts=starts, ends=ends), self.assertRaisesRegex(
                ValueError, "bounds"
            ):
                replay.replay_capture(
                    self.bundle(
                        starts=torch.tensor(starts, dtype=torch.int32),
                        ends=torch.tensor(ends, dtype=torch.int32),
                    )
                )
        for invalid in (123.0, float("nan")):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "zero-filled"
            ):
                replay.replay_capture(
                    self.bundle(logits=torch.tensor([[invalid, -3.0, -1.0, -2.0, 0.0]]))
                )

    def test_tensor_types_shapes_and_missing_extra_tensors(self):
        cases = (
            dict(logits=torch.zeros(1, 5, dtype=torch.float16)),
            dict(starts=torch.tensor([1], dtype=torch.int64)),
            dict(ends=torch.tensor([[4]], dtype=torch.int32)),
            dict(indices=torch.tensor([[2, 1]], dtype=torch.int64)),
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replay.replay_capture(self.bundle(**kwargs))
        for change in (
            lambda m: m["tensors"].pop("row_starts"),
            lambda m: m["tensors"]["indices"].update(shape=[2, 1]),
        ):
            path = self.bundle()
            self.mutate(path, change)
            with self.assertRaises(ValueError):
                replay.replay_capture(path)

    def test_metadata_tampering(self):
        mutations = [
            lambda m: m.update(op="other"),
            lambda m: m.update(phase="decode"),
            lambda m: m.update(schema_version=2),
            lambda m: m.update(complete=False),
            lambda m: m.update(query_row_count=2),
            lambda m: m.update(query_row_start=1),
            lambda m: m.update(finite_values_checked=True),
            lambda m: m.update(nonfinite_policy="ignore"),
            lambda m: m.update(output_contract="sorted"),
            lambda m: m["parameters"].update(stride0=7),
            lambda m: m["parameters"].update(num_rows=2),
            lambda m: m["parameters"].update(force_radix_sort=1),
            lambda m: m["parameters"].update(k=True),
            lambda m: m["parameters"].update(k=0),
            lambda m: m["parameters"].update(index_space="absolute"),
            lambda m: m["parameters"].update(extra=1),
            lambda m: m["parameters"].pop("stride1"),
            lambda m: m["tensors"]["logits"].update(nbytes=1),
        ]
        for index, change in enumerate(mutations):
            with self.subTest(index=index):
                path = self.bundle()
                self.mutate(path, change)
                with self.assertRaises(ValueError):
                    replay.replay_capture(path)

    def test_checksum_tampering(self):
        path = self.bundle()
        binary = path.parent / "indices.bin"
        binary.write_bytes(bytes(len(binary.read_bytes())))
        with self.assertRaisesRegex(ValueError, "checksum"):
            replay.replay_capture(path)

    def test_unsupported_backend_and_v3_k(self):
        for backend in ("torch", "unknown", "topk_v3_tie_break"):
            with self.subTest(backend=backend), self.assertRaises(
                replay.UnsupportedCapture
            ):
                replay.replay_capture(self.bundle(backend=backend))

    def test_layout_restoration_and_rejection(self):
        path = self.boundary_bundle("dsv4_per_row", 3)
        manifest, tensors = capture.load_capture(path)
        desc = manifest["tensors"]["logits"]
        restored = replay._restore(tensors["logits"], desc, "cpu")
        self.assertEqual(restored.stride(), tuple(desc["original_stride"]))
        self.assertEqual(restored.storage_offset(), 3)
        self.assertTrue(torch.equal(restored, tensors["logits"]))
        for name, change in (
            ("logits", dict(original_stride=[1, 0])),
            ("indices", dict(original_stride=[6, 2])),
            ("logits", dict(storage_offset=2**40)),
        ):
            with self.subTest(name=name, change=change):
                path = self.bundle()

                def mutate(m):
                    m["tensors"][name].update(change)
                    if name == "logits" and "original_stride" in change:
                        m["parameters"].update(stride0=1, stride1=0)

                self.mutate(path, mutate)
                with self.assertRaises(replay.UnsupportedCapture):
                    replay.replay_capture(path)

    def test_cuda_dispatch_wiring_on_cpu_not_a_kernel_test(self):
        # Mock only device allocation/dispatch to check orchestration without
        # claiming to execute CUDA. Real kernels have a separate opt-in class.
        restore = replay._restore
        for backend, k in (("dsv4_per_row", 3), ("topk_v3_tie_break", 512)):
            for behavior in ("write", "unwritten", "failure"):
                with self.subTest(backend=backend, behavior=behavior):
                    path = self.boundary_bundle(backend, k, force=True)
                    manifest, tensors = capture.load_capture(path)

                    def operator(logits, starts, ends, output, *params):
                        self.assertEqual(
                            logits.stride(),
                            tuple(manifest["tensors"]["logits"]["original_stride"]),
                        )
                        self.assertEqual(logits.storage_offset(), 3)
                        self.assertTrue(bool((output == -2).all()))
                        saved = manifest["parameters"]
                        expected = (
                            (4, saved["stride0"], saved["stride1"], k, True)
                            if backend == "dsv4_per_row"
                            else (k, saved["max_seq_len"])
                        )
                        self.assertEqual(params, expected)
                        if behavior == "failure":
                            raise RuntimeError("synthetic launch error")
                        if behavior == "write":
                            output.copy_(tensors["indices"])
                            output[3] = output[3].flip(0)

                    with mock.patch.object(
                        replay, "_cuda_operator", return_value=operator
                    ), mock.patch.object(
                        replay,
                        "_restore",
                        side_effect=lambda t, d, device: restore(t, d, "cpu"),
                    ), mock.patch.object(
                        torch.cuda, "synchronize"
                    ):
                        if behavior == "write":
                            result = replay.replay_capture(path, "cuda")
                            self.assertFalse(result["raw_order_equal"])
                        elif behavior == "unwritten":
                            with self.assertRaisesRegex(
                                ValueError, "row 0: invalid -1 padding"
                            ):
                                replay.replay_capture(path, "cuda")
                        else:
                            with self.assertRaisesRegex(
                                RuntimeError, "synthetic launch"
                            ):
                                replay.replay_capture(path, "cuda")

    def test_v3_invalid_parameters_and_layout(self):
        for value in (0, -1, 521, True, 1.5):
            path = self.boundary_bundle("topk_v3_tie_break", 512)
            self.mutate(path, lambda m: m["parameters"].update(max_seq_len=value))
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "max_seq_len"
            ):
                replay.replay_capture(path)
        path = self.boundary_bundle("topk_v3_tie_break", 512)

        def mutate(m):
            m["parameters"].update(stride0=1048, stride1=2)
            m["tensors"]["logits"]["original_stride"] = [1048, 2]

        self.mutate(path, mutate)
        with self.assertRaisesRegex(replay.UnsupportedCapture, "stride1"):
            replay.replay_capture(path)

    def test_cli_summary_and_unavailable_exit_codes(self):
        path = self.bundle()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(replay.main([str(path)]), 0)
        self.assertFalse(json.loads(output.getvalue())["kernel_replayed"])
        with mock.patch.object(
            torch.cuda, "is_available", return_value=False
        ), contextlib.redirect_stderr(io.StringIO()) as error:
            self.assertEqual(replay.main([str(path), "--device", "cuda"]), 3)
        self.assertEqual(json.loads(error.getvalue())["status"], "unavailable")
        with mock.patch.object(
            torch.cuda, "is_available", return_value=True
        ), mock.patch.object(torch.version, "cuda", "test"), mock.patch.object(
            replay.importlib,
            "import_module",
            side_effect=ImportError("missing library"),
        ), self.assertRaisesRegex(
            replay.ReplayUnavailable, "operator unavailable"
        ):
            replay.replay_capture(path, "cuda")
        self.mutate(path, lambda m: m.update(op="other"))
        with contextlib.redirect_stderr(io.StringIO()) as error:
            self.assertEqual(replay.main([str(path)]), 1)
        self.assertEqual(json.loads(error.getvalue())["status"], "failed")
        self.mutate(path, lambda m: m.update(op="indexer.topk", backend="unknown"))
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(replay.main([str(path)]), 2)

    def test_real_capture_round_trip_no_reader_authorization(self):
        config = self.root / "capture.json"
        config.write_text(
            json.dumps(
                dict(
                    enabled=True,
                    authorized_capture=True,
                    layers=[0],
                    ops=["indexer.topk"],
                    phases=["prefill"],
                    rank_filter="record_owner",
                    max_calls=1,
                    max_bundle_bytes=100000,
                    max_total_bytes=100000,
                    max_inflight_bytes=1000000,
                )
            )
        )
        identity = dict(
            schema_version=1,
            session_id="test",
            owner_id="owner",
            replica_id="dp0",
            world_rank=0,
            dp_rank=0,
            exec_id="1",
            model_role="target",
            requests=[dict(phase="prefill", request_id="r1", q_len=2, kv_len=1)],
        )
        with mock.patch.dict(
            os.environ,
            RTP_LLM_RECORD_DIR=str(self.root / "records"),
            RTP_LLM_RECORD_SESSION="test",
        ):
            recorder = capture.TensorCaptureRecorder(str(config))
            try:
                context = recorder.for_execution(json.dumps(identity))
                self.assertIsNotNone(context)
                token = context.begin_topk(
                    layer=0,
                    logits=torch.tensor([[2.0, 1.0, 9.0], [1.0, 3.0, 2.0]]),
                    row_starts=torch.tensor([0, 0], dtype=torch.int32),
                    row_ends=torch.tensor([2, 3], dtype=torch.int32),
                    k=2,
                    backend="dsv4_per_row",
                    parameters=dict(
                        num_rows=2, stride0=3, stride1=1, force_radix_sort=False
                    ),
                    query_row_start=0,
                    total_query_rows=2,
                )
                self.assertIsNotNone(token)
                token.finish(torch.tensor([[0, 1], [2, 1]], dtype=torch.int32))
            finally:
                self.assertTrue(recorder.close(5))
        paths = list((self.root / "records").rglob("manifest.json"))
        self.assertEqual(len(paths), 1)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(replay.replay_capture(paths[0])["counts"]["rows"], 2)


@unittest.skipUnless(
    os.environ.get("RTP_LLM_TEST_CAPTURE_REPLAY_CUDA") == "1",
    "CUDA replay not requested; opt in with RTP_LLM_TEST_CAPTURE_REPLAY_CUDA=1",
)
class TensorCaptureReplayCudaTest(_ReplayFixture):
    def check_kernel(self, backend, k, force=False):
        result = replay.replay_capture(self.boundary_bundle(backend, k, force), "cuda")
        self.assertTrue(result["kernel_replayed"])
        self.assertEqual(result["mode"], "cuda_kernel_replay")

    def test_dsv4_insertion(self):
        self.check_kernel("dsv4_per_row", 3)

    def test_dsv4_forced_radix(self):
        self.check_kernel("dsv4_per_row", 3, force=True)

    def test_dsv4_unique_negative_scores(self):
        for force in (False, True):
            with self.subTest(force=force):
                result = replay.replay_capture(
                    self.bundle(force_radix_sort=force), "cuda"
                )
                self.assertTrue(result["kernel_replayed"])

    def test_v3_tie_break(self):
        self.check_kernel("topk_v3_tie_break", 512)


if __name__ == "__main__":
    unittest.main()
