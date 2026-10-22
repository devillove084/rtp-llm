"""Execute the native publisher and PyWrappedModel with real torch tensors.

Requires the built extension and CUDA; FP8 cases also require the real TorchSpec
quantizer. Only the external Mooncake store, legacy network writer and remote TP
peers are substituted. The Python model supplies tensors, not wrapper behavior.
No source-contract tests, extracted code, dependency skips or runtime fallbacks.
"""

import importlib
import importlib.util
import sys
import types
import unittest
from itertools import product
from pathlib import Path
from unittest.mock import Mock, patch


class MockStore:
    instances = []
    failure = None

    def __init__(self, config):
        self.config = config
        self.batches = []
        self.errors = []
        self.closed = False
        self.flushes = 0
        self.events = []
        self.instances.append(self)

    def setup(self, device):
        self.device = device

    def warmup_rdma(self):
        if self.failure == "init":
            raise RuntimeError("injected warmup failure")

    def take_async_errors(self):
        self.events.append("poll")
        errors, self.errors = self.errors, []
        return errors

    def put_batch(self, **payload):
        self.events.append("put")
        if self.failure == "admit":
            raise RuntimeError("injected admission failure")
        if self.failure == "duplicate":
            raise FileExistsError("duplicate request key")
        self.batches.append(payload)
        if self.failure == "async":
            self.errors.append(
                RuntimeError(f"batch {payload['batch_id']} completion failed")
            )
        return (
            [] if self.failure == "contract" else [{} for _ in payload["request_ids"]]
        )

    def flush(self):
        self.events.append("flush")
        self.flushes += 1
        if self.failure == "flush":
            raise RuntimeError("injected flush failure")

    def close(self):
        self.events.append("close")
        self.closed = True


class MockConfig:
    hidden_dim = 2
    num_aux_layers = 2
    store_key_namespace = "GlmBase"

    @classmethod
    def from_env(cls):
        return cls()

    def make_store_key(self, request_id):
        return f"rtp_{self.store_key_namespace}_{request_id}"


class CaptureFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch = importlib.import_module("torch")
        cls.ext = importlib.import_module(
            "rtp_llm.cpp.models.test.libth_pywrapped_model_cache_store_integration_test"
        )

    def setUp(self):
        MockStore.instances = []
        MockStore.failure = None
        mooncake = types.ModuleType("torchspec.transfer.mooncake")
        mooncake.MooncakeConfig = MockConfig
        mooncake.EagleMooncakeStore = MockStore
        # Do not replace torchspec.utils.fp8: the successful FP8 path must run
        # the installed quantizer, not a second implementation of its algorithm.
        self.module_patch = patch.dict(sys.modules, {mooncake.__name__: mooncake})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.publishers = []
        self.addCleanup(self.close_publishers)
        self.addCleanup(self.ext.clear_comm_ops)

    def close_publishers(self):
        for publisher in self.publishers:
            publisher.close_without_gil()
        self.publishers.clear()

    def publisher(self, **kwargs):
        publisher = self.ext.CapturePublisher(**kwargs)
        self.publishers.append(publisher)
        return publisher

    def payload(self, device="cpu"):
        t = self.torch
        return [
            t.arange(30, dtype=t.float16, device=device).reshape(5, 6),
            t.arange(5, dtype=t.int32, device=device),
            t.tensor([2, 3], dtype=t.int32),
            t.tensor([100, 101], dtype=t.int64),
        ]

    def model(
        self,
        malformed=None,
        malformed_once=False,
        writer_factory=None,
        mtp=False,
        final_norm=False,
        capture_layer_ids=(0, 1),
        mtp_failure=None,
        mtp_hc_mult=1,
    ):
        t, ext = self.torch, self.ext

        class Model:
            supports_hidden_state_capture = True
            micro_batch_final_norm = True

            def __init__(self):
                self.flags = []
                self.cache_inputs = []
                self.attention_inputs = []
                self.token_ids = []
                self.writers = []
                self.mtp_hidden = None
                self.mtp_calls = []
                self.resources = None

            def initialize(self, resources):
                self.resources = resources
                return True

            def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
                return None

            def get_mtp_target_hidden_states(self, num_tokens):
                self.mtp_calls.append(num_tokens)
                if mtp_failure == "exception":
                    raise RuntimeError("injected MTP getter failure")
                if mtp_failure == "rows":
                    return self.mtp_hidden[:-1]
                if mtp_failure == "rank":
                    return self.mtp_hidden.flatten()
                if mtp_failure == "width":
                    return self.mtp_hidden[:, :-1].contiguous()
                if mtp_failure == "dtype":
                    return self.mtp_hidden.float()
                if mtp_failure == "device":
                    return self.mtp_hidden.cpu()
                if mtp_failure == "empty":
                    return self.mtp_hidden[:0]
                return self.mtp_hidden if mtp else None

            def forward(self, inputs, fmha=None):
                self.flags.append(inputs.capture_hidden_states)
                self.cache_inputs.append(inputs.attention_inputs.cache_store_inputs)
                self.attention_inputs.append(inputs.attention_inputs)
                self.token_ids.append(inputs.input_ids.clone())
                if writer_factory is not None:
                    writer = writer_factory(inputs.attention_inputs)
                    self.writers.append(writer)
                    if writer is not None:
                        writer.forward(None)
                hidden = (
                    inputs.input_ids.to(t.float16)
                    .unsqueeze(1)
                    .expand(-1, 2)
                    .contiguous()
                )
                self.mtp_hidden = (hidden + 100).repeat(1, mtp_hc_mult)
                last_hidden = hidden + 30 if inputs.capture_hidden_states else hidden
                if final_norm:
                    # Python owns final norm in both eager and micro-batch forwards.
                    # Preserve the model dtype required by the packed capture layout.
                    gamma = t.tensor([0.5, 1.5], device=hidden.device, dtype=t.float32)
                    last_hidden = t.nn.functional.rms_norm(
                        last_hidden.float(), (2,), weight=gamma, eps=1e-5
                    ).to(hidden.dtype)
                if inputs.capture_hidden_states:
                    hidden = t.cat(
                        [hidden + (layer_id + 1) * 10 for layer_id in capture_layer_ids]
                        + [last_hidden],
                        -1,
                    )
                    if not malformed_once or len(self.flags) == 1:
                        if malformed == "width":
                            hidden = hidden[:, :-1]
                        elif malformed == "rows":
                            hidden = hidden[:-1]
                        elif malformed == "rank":
                            hidden = hidden.flatten()
                        elif malformed == "dtype":
                            hidden = hidden.float()
                        elif malformed == "device":
                            hidden = hidden.cpu()
                else:
                    hidden = last_hidden
                return ext.PyModelOutputs(hidden, params_ptr=fmha)

            def forward_micro_batch(self, inputs):
                outputs = [self.forward(item) for item in inputs]
                if malformed == "count":
                    return outputs[:-1]
                if malformed == "fake_lane":
                    outputs[-1] = ext.PyModelOutputs(
                        t.empty(0, device=inputs[0].input_ids.device)
                    )
                return outputs

        return Model()

    def register_peers(self, allreduce, allgather=None):
        def unexpected(*args):
            self.fail("unexpected TP collective")

        self.ext.register_comm_ops(unexpected, allreduce, allgather or unexpected)

    def cp_peers(
        self,
        mtp=False,
        logits=False,
        lengths=(6,),
        tp_size=2,
        tp_rank=0,
        input_ids_on_cuda=False,
        capture_layer_ids=(0, 1),
        mtp_hc_mult=1,
    ):
        t = self.torch
        calls = []
        ids = t.arange(
            1,
            sum(lengths) + 1,
            dtype=t.int32,
            device="cuda" if input_ids_on_cuda else "cpu",
        )
        peers = []
        for rank in range(tp_size):
            split = self.ext.split_cp_inputs(ids, lengths, tp_size, rank)
            self.assertEqual(split["original_lengths"].tolist(), list(lengths))
            peer = self.model(
                mtp=mtp, capture_layer_ids=capture_layer_ids, mtp_hc_mult=mtp_hc_mult
            )
            output = peer.forward(
                types.SimpleNamespace(
                    input_ids=split["input_ids"],
                    capture_hidden_states=True,
                    attention_inputs=types.SimpleNamespace(cache_store_inputs=None),
                )
            )
            peers.append((output.hidden_states, peer.mtp_hidden))

        def allreduce(tensor, op, mode, dest):
            self.assertGreater(tp_size, 1)
            self.assertEqual((op, mode, dest), (0, 0, None))  # SUM, TP
            self.assertEqual(calls, [])
            calls.append("validate")
            self.assertEqual(tensor.tolist(), [1, 1, int(mtp)])
            return tensor * tp_size

        def allgather(receive, mode, send, inplace):
            self.assertGreater(tp_size, 1)
            self.assertEqual(mode, 0)
            self.assertFalse(inplace)
            self.assertEqual(calls[0], "validate")
            if mtp and calls == ["validate"]:
                self.assertEqual(send[0].shape[1], 2 * mtp_hc_mult)
                calls.append("mtp_restore")
                payloads = [peer[1] for peer in peers]
            elif logits and send[0].shape[1] == 2:
                calls.append("logits")
                payloads = [send[0]] * tp_size
            else:
                self.assertEqual(send[0].shape[1], (len(capture_layer_ids) + 1) * 2)
                calls.append("restore")
                payloads = [peer[0] for peer in peers]
            t.testing.assert_close(send[0], payloads[tp_rank])
            payloads[tp_rank] = send[0]
            receive[0].copy_(t.cat(payloads, 0))

        self.register_peers(allreduce, allgather)
        return calls


class PublisherTest(CaptureFixture):
    def test_bf16_request_slicing_and_close_without_gil(self):
        publisher = self.publisher()
        payload = self.payload()
        result = publisher.forward(*payload)
        self.assertIsNone(result["error"])
        self.torch.testing.assert_close(result["hidden"], payload[0][:, 4:])
        store = MockStore.instances[0]
        batch = store.batches[0]
        self.assertEqual(batch["request_ids"], ["rtp_GlmBase_100", "rtp_GlmBase_101"])
        for index, (start, end) in enumerate(((0, 2), (2, 5))):
            self.torch.testing.assert_close(
                batch["hidden_states"][index],
                payload[0][start:end, :4].to(self.torch.bfloat16),
            )
            self.torch.testing.assert_close(
                batch["last_hidden_states"][index],
                payload[0][start:end, 4:].to(self.torch.bfloat16),
            )
            self.assertEqual(batch["input_ids"][index].dtype, self.torch.int64)
            self.assertEqual(
                batch["input_ids"][index].tolist(), payload[1][start:end].tolist()
            )
        self.assertNotIn("hidden_states_scale", batch)
        self.assertEqual(store.events, ["poll", "put", "flush", "poll"])
        publisher.close_without_gil()
        self.assertTrue(store.closed)
        self.assertEqual(store.events[-3:], ["flush", "poll", "close"])
        events = list(store.events)
        publisher.close_without_gil()
        self.assertEqual(store.events, events)

    def test_non_owner_does_not_initialize_store_or_validate_metadata(self):
        publisher = self.publisher(owner=False)
        payload = self.payload()
        payload[2] = self.torch.tensor([-1], dtype=self.torch.int32)
        result = publisher.forward(*payload)
        self.assertIsNone(result["error"])
        self.assertEqual(MockStore.instances, [])
        self.assertEqual(result["failures"], 0)
        self.torch.testing.assert_close(result["hidden"], payload[0][:, 4:])

    def test_request_errors_remain_recoverable(self):
        publisher = self.publisher()
        for lengths, requests in (
            ([0, 5], [100, 101]),
            ([-1, 6], [100, 101]),
            ([2, 2], [100, 101]),
            ([2, 3], [-1, 101]),
            ([2, 3], [100, 100]),
            ([2, 3], [100]),
            ([], []),
        ):
            with self.subTest(lengths=lengths, requests=requests):
                payload = self.payload()
                payload[2] = self.torch.tensor(lengths, dtype=self.torch.int32)
                payload[3] = self.torch.tensor(requests, dtype=self.torch.int64)
                batches = len(MockStore.instances[0].batches)
                result = publisher.forward(*payload)
                self.assertIsNotNone(result["error"])
                self.assertEqual(len(MockStore.instances[0].batches), batches)
                self.torch.testing.assert_close(result["hidden"], payload[0][:, 4:])
                self.assertIsNone(publisher.forward(*self.payload())["error"])

    def test_metadata_rank_and_dtype_errors_are_request_local(self):
        publisher = self.publisher()
        for index, change in (
            (1, lambda x: x.reshape(1, -1)),
            (2, lambda x: x.long()),
            (3, lambda x: x.int()),
            (2, lambda x: x.reshape(1, -1)),
            (3, lambda x: x.reshape(1, -1)),
        ):
            with self.subTest(index=index, change=change):
                payload = self.payload()
                payload[index] = change(payload[index])
                self.assertIsNotNone(publisher.forward(*payload)["error"])
                result = publisher.forward(*self.payload())
                self.assertIsNone(result["error"])
                self.assertEqual(result["broken_rejections"], 0)

    def test_layout_contract_disables_future_publication(self):
        for kind in ("rank", "rows", "width", "dtype"):
            with self.subTest(kind=kind):
                publisher = self.publisher()
                payload = self.payload()
                payload[0] = {
                    "rank": payload[0].flatten(),
                    "rows": payload[0][:4],
                    "width": payload[0][:, :5],
                    "dtype": payload[0].float(),
                }[kind]
                self.assertIsNotNone(publisher.forward(*payload)["error"])
                result = publisher.forward(*self.payload())
                self.assertIsNotNone(result["error"])
                self.assertEqual(result["broken_rejections"], 1)
                self.assertEqual(MockStore.instances[-1].batches, [])

    def test_store_failures_recover_except_hard_admission_contract(self):
        for failure in ("admit", "duplicate", "flush", "async", "contract"):
            with self.subTest(failure=failure):
                publisher = self.publisher()
                MockStore.failure = failure
                self.assertIsNotNone(publisher.forward(*self.payload())["error"])
                MockStore.failure = None
                result = publisher.forward(*self.payload())
                self.assertEqual(result["error"] is not None, failure == "contract")
                self.assertEqual(
                    result["broken_rejections"], int(failure == "contract")
                )

    def test_historical_async_failure_is_not_current_request_failure(self):
        publisher = self.publisher()
        MockStore.instances[0].errors.append(RuntimeError("previous batch request 88"))
        result = publisher.forward(*self.payload())
        self.assertIsNone(result["error"])
        self.assertEqual(result["failures"], 1)
        self.assertEqual(len(MockStore.instances[0].batches), 1)

    def test_error_is_consumed_once_and_stale_error_is_not_reused(self):
        publisher = self.publisher()
        MockStore.failure = "admit"
        publisher.forward(*self.payload(), consume_error=False)
        self.assertIn("admission failure", publisher.take_error())
        self.assertIsNone(publisher.take_error())
        publisher.forward(*self.payload(), consume_error=False)
        MockStore.failure = None
        self.assertIsNone(publisher.forward(*self.payload())["error"])
        self.assertIsNone(publisher.take_error())

    def test_partial_initialization_closes_store(self):
        MockStore.failure = "init"
        with self.assertRaisesRegex(RuntimeError, "warmup failure"):
            self.publisher()
        self.assertTrue(MockStore.instances[0].closed)

    def test_shutdown_flush_failure_still_closes_store(self):
        publisher = self.publisher()
        MockStore.failure = "flush"
        publisher.close_without_gil()
        self.assertTrue(MockStore.instances[0].closed)
        self.assertIsNone(publisher.take_error())

    def test_noncontiguous_payload_and_single_token_boundary(self):
        publisher = self.publisher()
        payload = self.payload()
        payload[0] = payload[0].t().contiguous().t()
        self.assertFalse(payload[0].is_contiguous())
        self.assertIsNone(publisher.forward(*payload)["error"])
        payload = [
            payload[0][:1],
            payload[1][:1],
            self.torch.tensor([1], dtype=self.torch.int32),
            payload[3][:1],
        ]
        self.assertIsNone(publisher.forward(*payload)["error"])
        self.assertEqual(
            MockStore.instances[0].batches[-1]["input_ids"][0].tolist(), [0]
        )

    def test_fp8_real_quantizer_scales_and_bf16_last_hidden(self):
        quantizer = importlib.import_module(
            "torchspec.utils.fp8"
        ).quantize_aux_hidden_states
        payload = self.payload(device="cuda")
        expected_hidden, expected_scale = quantizer(payload[0][:, :4].contiguous(), 2)
        publisher = self.publisher(fp8=True)
        result = publisher.forward(*payload)
        self.assertIsNone(result["error"])
        batch = MockStore.instances[0].batches[0]
        self.assertEqual(batch["hidden_states"][0].dtype, self.torch.float8_e4m3fn)
        self.torch.testing.assert_close(
            self.torch.cat(batch["hidden_states"]).float(), expected_hidden.float()
        )
        self.torch.testing.assert_close(
            self.torch.cat(batch["hidden_states_scale"]), expected_scale
        )
        self.torch.testing.assert_close(
            self.torch.cat(batch["last_hidden_states"]),
            payload[0][:, 4:].to(self.torch.bfloat16),
        )
        self.torch.testing.assert_close(result["hidden"], payload[0][:, 4:])

    def test_invalid_fp8_result_is_hard_failure(self):
        module = importlib.import_module("torchspec.utils.fp8")
        payload = self.payload(device="cuda")
        quantized, scale = module.quantize_aux_hidden_states(
            payload[0][:, :4].contiguous(), 2
        )
        for invalid in (
            "not tensors",
            (quantized,),
            (quantized.float(), scale),
            (quantized, scale[:, :1]),
            (quantized, scale.half()),
        ):
            with self.subTest(invalid=type(invalid)):
                with patch.object(
                    module, "quantize_aux_hidden_states", return_value=invalid
                ):
                    publisher = self.publisher(fp8=True)
                    self.assertIsNotNone(publisher.forward(*payload)["error"])
                    self.assertEqual(
                        publisher.forward(*payload)["broken_rejections"], 1
                    )
                    self.assertEqual(MockStore.instances[-1].batches, [])

    def test_fp8_execution_failure_is_recoverable(self):
        module = importlib.import_module("torchspec.utils.fp8")
        real_quantizer = module.quantize_aux_hidden_states
        payload = self.payload(device="cuda")
        with patch.object(
            module,
            "quantize_aux_hidden_states",
            side_effect=RuntimeError("quantizer unavailable"),
        ) as quantizer:
            publisher = self.publisher(fp8=True)
            self.assertIn("quantizer unavailable", publisher.forward(*payload)["error"])
            quantizer.side_effect = real_quantizer
            self.assertIsNone(publisher.forward(*payload)["error"])


class PyWrappedModelTest(CaptureFixture):
    def test_capture_and_skip_lm_head_are_independent(self):
        for capture, skip, micro_batch in product((False, True), repeat=3):
            with self.subTest(capture=capture, skip=skip, micro_batch=micro_batch):
                model = self.model()
                result = self.ext.run_model(
                    model, capture=capture, skip_lm_head=skip, micro_batch=micro_batch
                )
                self.assertIsNone(result["error"])
                self.assertEqual(result["logits"] is None, skip)
                self.assertEqual(model.flags, [capture] * (2 if micro_batch else 1))
                self.assertEqual(len(MockStore.instances[-1].batches), int(capture))
                expected = self.torch.arange(
                    1, 6, device="cuda", dtype=self.torch.float16
                )
                expected = (
                    (expected + (30 if capture else 0)).unsqueeze(1).expand(-1, 2)
                )
                self.torch.testing.assert_close(result["hidden"], expected)
                if not skip:
                    self.torch.testing.assert_close(
                        result["logits"], expected[[1, 4]].float() * 2
                    )
                self.assertEqual(
                    [params is None for params in model.cache_inputs],
                    [skip] * len(model.flags),
                )
                self.assertTrue(MockStore.instances[-1].closed)

    def test_unconfigured_capture_does_not_create_publisher(self):
        model = self.model()
        result = self.ext.run_model(model, configured=False)
        self.assertIsNone(result["error"])
        self.assertEqual(model.flags, [False])
        self.assertEqual(MockStore.instances, [])
        self.assertEqual(tuple(result["hidden"].shape), (5, 2))

    def test_warmup_and_fake_do_not_capture(self):
        for micro_batch in (False, True):
            for kwargs in ({"warmup": True}, {"fake": True}):
                with self.subTest(micro_batch=micro_batch, kwargs=kwargs):
                    model = self.model()
                    result = self.ext.run_model(
                        model, micro_batch=micro_batch, **kwargs
                    )
                    self.assertIsNone(result["error"])
                    self.assertEqual(model.flags, [False] * (2 if micro_batch else 1))
                    self.assertEqual(MockStore.instances[-1].batches, [])

    def test_tp_non_owner_keeps_packed_forward_without_store(self):
        calls = []

        def allreduce(tensor, op, mode, dest):
            calls.append(tensor.item())
            return tensor

        self.register_peers(allreduce)
        for micro_batch in (False, True):
            with self.subTest(micro_batch=micro_batch):
                model = self.model()
                result = self.ext.run_model(
                    model, tp_rank=1, tp_size=2, micro_batch=micro_batch
                )
                self.assertIsNone(result["error"])
                self.assertTrue(all(model.flags))
                self.assertEqual(result["hidden"][:, 0].tolist(), [31, 32, 33, 34, 35])
        self.assertEqual(calls, [1, 1])
        self.assertEqual(MockStore.instances, [])

    def test_micro_batch_and_fake_lane_preserve_rows(self):
        for lengths in ([2, 3], [1, 2, 3], [3], [1]):
            with self.subTest(lengths=lengths):
                model = self.model(malformed="fake_lane" if len(lengths) == 1 else None)
                result = self.ext.run_model(model, micro_batch=True, lengths=lengths)
                self.assertIsNone(result["error"])
                self.assertEqual(model.flags, [True, True])
                self.assertEqual(
                    result["hidden"][:, 0].tolist(), list(range(31, sum(lengths) + 31))
                )
                batch = MockStore.instances[-1].batches[0]
                self.assertEqual(
                    self.torch.cat(batch["input_ids"]).tolist(),
                    list(range(1, sum(lengths) + 1)),
                )
                self.assertEqual([x.shape[0] for x in batch["hidden_states"]], lengths)

    def test_empty_micro_batch_keeps_fake_lane_without_narrow_oob(self):
        for lengths in ([], [0]):
            with self.subTest(lengths=lengths):
                model = self.model()
                result = self.ext.run_model(
                    model,
                    capture=False,
                    configured=False,
                    micro_batch=True,
                    lengths=lengths,
                )
                self.assertIsNone(result["error"])
                self.assertEqual(tuple(result["hidden"].shape), (0, 2))
                self.assertEqual(model.flags, [False, False])
                self.assertEqual([ids.numel() for ids in model.token_ids], [0, 0])
                self.assertEqual(MockStore.instances, [])

    def test_capture_and_micro_batch_do_not_repeat_final_norm(self):
        t = self.torch
        gamma = t.tensor([0.5, 1.5], dtype=t.float32, device="cuda")
        for capture, micro_batch, skip in product((False, True), repeat=3):
            with self.subTest(capture=capture, micro_batch=micro_batch, skip=skip):
                kwargs = dict(
                    capture=capture, micro_batch=micro_batch, skip_lm_head=skip
                )
                raw = t.arange(1, 6, dtype=t.float32, device="cuda")
                raw = (raw + (30 if capture else 0)).unsqueeze(1).expand(-1, 2)
                expected = (
                    raw * t.rsqrt(raw.square().mean(-1, keepdim=True) + 1e-5) * gamma
                ).to(t.float16)

                # Positive control: C++ weights alone must not normalize Python's
                # output. Enabling the real Python norm must change that output.
                unnormalized = self.ext.run_model(
                    self.model(), final_norm=True, **kwargs
                )
                self.assertIsNone(unnormalized["error"])
                t.testing.assert_close(unnormalized["hidden"], raw.to(t.float16))
                self.assertFalse(t.allclose(unnormalized["hidden"], expected))

                baseline = self.ext.run_model(self.model(final_norm=True), **kwargs)
                self.assertIsNone(baseline["error"])
                t.testing.assert_close(baseline["hidden"], expected)
                result = self.ext.run_model(
                    self.model(final_norm=True), final_norm=True, **kwargs
                )
                self.assertIsNone(result["error"])
                t.testing.assert_close(result["hidden"], expected)
                t.testing.assert_close(result["hidden"], baseline["hidden"])
                if skip:
                    self.assertIsNone(result["logits"])
                else:
                    expected_logits = (
                        expected[[1, 4]].sum(-1, keepdim=True).expand(-1, 2).float()
                    )
                    t.testing.assert_close(result["logits"], expected_logits)
                    t.testing.assert_close(result["logits"], baseline["logits"])
                if capture:
                    batch = MockStore.instances[-1].batches[0]
                    t.testing.assert_close(
                        t.cat(batch["last_hidden_states"]), expected.to(t.bfloat16)
                    )

    def test_layout_failures_are_deferred_and_consumed_once(self):
        for micro_batch in (False, True):
            kinds = ("width", "rows", "rank", "dtype", "device") + (
                ("count",) if micro_batch else ()
            )
            for kind in kinds:
                with self.subTest(micro_batch=micro_batch, kind=kind):
                    result = self.ext.run_model(
                        self.model(malformed=kind), micro_batch=micro_batch
                    )
                    self.assertIsNotNone(result["error"])
                    self.assertIsNone(result["second_error"])
                    self.assertEqual(tuple(result["hidden"].shape), (5, 2))
                    self.assertEqual(MockStore.instances[-1].batches, [])

    def test_broken_capture_rejects_later_capture_but_not_ordinary_forward(self):
        result = self.ext.run_model(
            self.model(malformed="width", malformed_once=True),
            forward_captures=[True, False, True],
        )
        first, ordinary, rejected = result["forwards"]
        self.assertIn("width", first["error"])
        self.assertIsNone(ordinary["error"])
        self.assertIn("width", rejected["error"])
        self.assertEqual(rejected["broken_rejections"], 1)
        self.assertEqual(MockStore.instances[0].batches, [])

    def test_operational_store_failure_recovers_on_same_model(self):
        model = self.model()
        original_forward = model.forward

        def forward(inputs, fmha=None):
            MockStore.failure = "admit" if not model.flags else None
            return original_forward(inputs, fmha)

        model.forward = forward
        result = self.ext.run_model(model, forward_captures=[True, True])
        failed, recovered = result["forwards"]
        self.assertIn("admission failure", failed["error"])
        self.assertIsNone(recovered["error"])
        self.assertEqual(recovered["broken_rejections"], 0)
        self.assertEqual(len(MockStore.instances[0].batches), 1)

    def test_tp_owner_publishes_after_successful_layout_consensus(self):
        for micro_batch in (False, True):
            with self.subTest(micro_batch=micro_batch):
                calls = []

                def allreduce(tensor, op, mode, dest):
                    self.assertEqual(MockStore.instances[-1].batches, [])
                    calls.append(tensor.item())
                    return tensor

                self.register_peers(allreduce)
                result = self.ext.run_model(
                    self.model(), tp_size=2, micro_batch=micro_batch
                )
                self.assertIsNone(result["error"])
                self.assertEqual(calls, [1])
                self.assertEqual(len(MockStore.instances[-1].batches), 1)
                self.assertEqual(result["hidden"][:, 0].tolist(), [31, 32, 33, 34, 35])

    def test_tp_layout_consensus_precedes_publication(self):
        for micro_batch in (False, True):
            for malformed in (
                (None, "width", "count") if micro_batch else (None, "width")
            ):
                with self.subTest(micro_batch=micro_batch, malformed=malformed):
                    calls = []

                    def allreduce(tensor, op, mode, dest):
                        calls.append(tensor.item())
                        return tensor.zero_()

                    self.register_peers(allreduce)
                    result = self.ext.run_model(
                        self.model(malformed=malformed),
                        tp_size=2,
                        micro_batch=micro_batch,
                    )
                    self.assertEqual(calls, [int(malformed is None)])
                    self.assertIn(
                        "another TP rank" if malformed is None else malformed,
                        result["error"],
                    )
                    self.assertEqual(MockStore.instances[-1].batches, [])

    def test_cp_micro_batch_is_rejected(self):
        with self.assertRaisesRegex(
            RuntimeError, "context parallel combined with layer micro-batching"
        ):
            self.ext.run_model(self.model(), micro_batch=True, cp=True, tp_size=2)
        self.assertEqual(MockStore.instances, [])

    def test_zigzag_cp_restores_original_ids_and_hidden(self):
        calls = self.cp_peers()
        model = self.model()
        result = self.ext.run_model(model, cp=True, tp_size=2, lengths=[6])
        self.assertIsNone(result["error"])
        self.assertEqual(calls, ["validate", "restore"])
        self.assertEqual(model.token_ids[0].tolist(), [1, 2, 0, 0])
        batch = MockStore.instances[0].batches[0]
        self.assertEqual(batch["input_ids"][0].tolist(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(
            batch["last_hidden_states"][0][:, 0].tolist(), [31, 32, 33, 34, 35, 36]
        )
        self.assertEqual(result["hidden"][:, 0].tolist(), [31, 32, 33, 34, 35, 36])

    def test_cp_preserves_base_mtp_prenorm_handoff(self):
        calls = self.cp_peers(mtp=True)
        model = self.model(mtp=True)
        result = self.ext.run_model(model, cp=True, tp_size=2, lengths=[6])
        self.assertIsNone(result["error"])
        self.assertEqual(calls, ["validate", "mtp_restore", "restore"])
        self.assertEqual(model.mtp_calls, [4])
        self.assertEqual(
            result["hidden"][:, 0].tolist(), [101, 102, 103, 104, 105, 106]
        )
        self.assertEqual(
            MockStore.instances[0].batches[0]["last_hidden_states"][0][:, 0].tolist(),
            [31, 32, 33, 34, 35, 36],
        )

    def test_invalid_layout_prevents_cp_restore(self):
        for malformed in (None, "width"):
            with self.subTest(malformed=malformed):
                calls = []

                def allreduce(tensor, op, mode, dest):
                    self.assertEqual((op, mode, dest), (0, 0, None))
                    calls.append(tensor[0].item())
                    return tensor.zero_()

                self.register_peers(allreduce)
                result = self.ext.run_model(
                    self.model(malformed=malformed), cp=True, tp_size=2, lengths=[6]
                )
                self.assertEqual(calls, [int(malformed is None)])
                self.assertIn(
                    "another TP rank" if malformed is None else "width", result["error"]
                )
                self.assertEqual(MockStore.instances[-1].batches, [])

    def assert_cp_batch(self, lengths, tp_size, tp_rank, input_ids_on_cuda, mtp):
        t = self.torch
        layer_ids = (1, 0)
        store_count = len(MockStore.instances)
        calls = self.cp_peers(
            mtp=mtp,
            lengths=lengths,
            tp_size=tp_size,
            tp_rank=tp_rank,
            input_ids_on_cuda=input_ids_on_cuda,
            capture_layer_ids=layer_ids,
        )
        model = self.model(mtp=mtp, capture_layer_ids=layer_ids)
        if not mtp:
            # GLM/GenericMoe has no DSv4 pre-norm getter, rather than a getter
            # whose tensor happens to be absent for this particular request.
            del type(model).get_mtp_target_hidden_states
        result = self.ext.run_model(
            model,
            cp=True,
            tp_size=tp_size,
            tp_rank=tp_rank,
            lengths=lengths,
            input_ids_on_cuda=input_ids_on_cuda,
            capture_layer_ids=layer_ids,
        )
        self.assertIsNone(result["error"])
        self.assertIsNone(result["second_error"])
        self.assertEqual(result["failures"], 0)
        self.assertEqual(model.flags, [True])
        expected_calls = ["validate"] + (["mtp_restore"] if mtp else []) + ["restore"]
        self.assertEqual(calls, expected_calls if tp_size > 1 else [])
        ids = t.arange(1, sum(lengths) + 1, dtype=t.int64, device="cuda")
        hidden = ids.to(t.float16).unsqueeze(1).expand(-1, 2)
        t.testing.assert_close(result["hidden"], hidden + (100 if mtp else 30))
        self.assertEqual(model.mtp_calls, [model.token_ids[0].numel()] if mtp else [])
        cp_info = model.attention_inputs[0].context_parallel_info
        self.assertEqual(cp_info.prefill_actual_input_lengths_cpu.tolist(), lengths)
        if lengths == [1, 3] and tp_size == 4 and tp_rank == 3:
            self.assertEqual(t.count_nonzero(model.token_ids[0]).item(), 0)
        self.assertEqual(len(MockStore.instances), store_count + int(tp_rank == 0))
        if tp_rank != 0:
            return
        store = MockStore.instances[-1]
        self.assertTrue(store.closed)
        self.assertEqual(len(store.batches), 1)
        batch = store.batches[0]
        self.assertEqual(
            batch["request_ids"],
            [f"rtp_GlmBase_{100 + i}" for i in range(len(lengths))],
        )
        for field in ("input_ids", "hidden_states", "last_hidden_states"):
            self.assertEqual([value.shape[0] for value in batch[field]], lengths)
        offset = 0
        for index, length in enumerate(lengths):
            request_hidden = hidden[offset : offset + length]
            t.testing.assert_close(
                batch["input_ids"][index], ids[offset : offset + length]
            )
            t.testing.assert_close(
                batch["hidden_states"][index],
                t.cat(
                    [request_hidden + (layer_id + 1) * 10 for layer_id in layer_ids],
                    -1,
                ).to(t.bfloat16),
            )
            t.testing.assert_close(
                batch["last_hidden_states"][index], (request_hidden + 30).to(t.bfloat16)
            )
            offset += length

    def test_cp_multi_request_padding_layer_order_and_all_ranks(self):
        cases = (([6, 9], 2), ([10, 20, 7], 4), ([1, 3], 4), ([8, 16], 4))
        for lengths, tp_size in cases:
            for tp_rank, cuda_ids, mtp in product(
                range(tp_size), (False, True), (False, True)
            ):
                with self.subTest(
                    lengths=lengths,
                    tp_size=tp_size,
                    rank=tp_rank,
                    cuda_ids=cuda_ids,
                    mtp=mtp,
                ):
                    self.assert_cp_batch(lengths, tp_size, tp_rank, cuda_ids, mtp)

    def test_cp_single_rank_restores_without_transport(self):
        for lengths, cuda_ids, mtp in product(
            ([6, 9], [1, 3], [8, 16]), (False, True), (False, True)
        ):
            with self.subTest(lengths=lengths, cuda_ids=cuda_ids, mtp=mtp):
                self.assert_cp_batch(lengths, 1, 0, cuda_ids, mtp)

    def test_cp_mtp_getter_failures_reject_before_any_restore_on_every_rank(self):
        # Only the collective boundary is substituted: each rank runs the native
        # getter validation, rejection, fallback, and owner-local deferred error.
        for tp_size in (2, 4):
            for bad_rank, failure in product(
                (0, tp_size - 1),
                (
                    "exception",
                    "rows",
                    "rank",
                    "width",
                    "dtype",
                    "device",
                    "absent",
                    "empty",
                    "no_getter",
                ),
            ):
                statuses = [[1, 1, 1] for _ in range(tp_size)]
                if failure in ("absent", "empty", "no_getter"):
                    statuses[bad_rank] = [1, 1, 0]
                else:
                    statuses[bad_rank] = [1, 0, int(failure != "exception")]
                for tp_rank in range(tp_size):
                    with self.subTest(
                        tp_size=tp_size,
                        rank=tp_rank,
                        bad_rank=bad_rank,
                        failure=failure,
                    ):
                        calls = []

                        def allreduce(tensor, op, mode, dest):
                            self.assertEqual((op, mode, dest), (0, 0, None))
                            self.assertEqual(tensor.tolist(), statuses[tp_rank])
                            calls.append("validate")
                            return tensor.new_tensor(
                                [sum(flags) for flags in zip(*statuses)]
                            )

                        # Any MTP or packed all-gather fails the test immediately.
                        self.register_peers(allreduce)
                        store_count = len(MockStore.instances)
                        local_failure = failure if tp_rank == bad_rank else None
                        model = self.model(
                            mtp=local_failure != "absent",
                            mtp_failure=local_failure,
                        )
                        if local_failure == "no_getter":
                            del type(model).get_mtp_target_hidden_states
                        result = self.ext.run_model(
                            model,
                            cp=True,
                            tp_size=tp_size,
                            tp_rank=tp_rank,
                            lengths=[6, 9],
                        )
                        self.assertEqual(calls, ["validate"])
                        self.assertIsNone(result["second_error"])
                        self.assertEqual(tuple(result["hidden"].shape), (15, 2))
                        self.torch.testing.assert_close(
                            result["hidden"], self.torch.zeros_like(result["hidden"])
                        )
                        self.assertEqual(
                            len(MockStore.instances), store_count + int(tp_rank == 0)
                        )
                        if tp_rank == 0:
                            if failure in ("absent", "empty", "no_getter"):
                                message = "presence differs"
                            elif bad_rank != 0:
                                message = "another TP rank"
                            elif failure == "exception":
                                message = "getter failed"
                            elif failure in ("width", "dtype", "device"):
                                message = failure
                            else:
                                message = "local capture rows"
                            self.assertIn(message, result["error"])
                            if bad_rank == 0 and failure == "exception":
                                self.assertIn(
                                    "injected MTP getter failure", result["error"]
                                )
                            self.assertEqual(result["failures"], 1)
                            self.assertEqual(MockStore.instances[-1].batches, [])
                        else:
                            self.assertIsNone(result["error"])
                            self.assertEqual(result["failures"], 0)

    def test_cp_mtp_expanded_prenorm_width_uses_hc_mult(self):
        t = self.torch
        for tp_rank in range(2):
            with self.subTest(rank=tp_rank):
                calls = self.cp_peers(
                    mtp=True, lengths=[6, 9], tp_rank=tp_rank, mtp_hc_mult=4
                )
                result = self.ext.run_model(
                    self.model(mtp=True, mtp_hc_mult=4),
                    cp=True,
                    tp_size=2,
                    tp_rank=tp_rank,
                    lengths=[6, 9],
                    hc_mult=4,
                )
                self.assertIsNone(result["error"])
                self.assertEqual(calls, ["validate", "mtp_restore", "restore"])
                ids = t.arange(1, 16, dtype=t.float16, device="cuda")
                expected = (ids + 100).unsqueeze(1).expand(-1, 8)
                t.testing.assert_close(result["hidden"], expected)
                if tp_rank == 0:
                    batch = MockStore.instances[-1].batches[0]
                    t.testing.assert_close(
                        t.cat(batch["last_hidden_states"]),
                        (ids + 30).unsqueeze(1).expand(-1, 2).to(t.bfloat16),
                    )

    def test_cp_reject_equal_local_global_rows_zeros_hidden_and_logits(self):
        t = self.torch
        for failure, bad_rank, tp_rank, skip, cuda_ids in product(
            ("exception", "absent", "packed"),
            range(2),
            range(2),
            (False, True),
            (False, True),
        ):
            with self.subTest(
                failure=failure,
                bad_rank=bad_rank,
                rank=tp_rank,
                skip_lm_head=skip,
                cuda_ids=cuda_ids,
            ):
                calls = []
                statuses = [[1, 1, 1], [1, 1, 1]]
                statuses[bad_rank] = {
                    "exception": [1, 0, 0],
                    "absent": [1, 1, 0],
                    "packed": [0, 1, 0],
                }[failure]

                def allreduce(tensor, op, mode, dest):
                    self.assertEqual((op, mode, dest), (0, 0, None))
                    self.assertEqual(tensor.tolist(), statuses[tp_rank])
                    calls.append("validate")
                    return tensor.new_tensor([sum(flags) for flags in zip(*statuses)])

                def allgather(receive, mode, send, inplace):
                    # Rejection skips both restores but must still join the
                    # ordinary LM-head TP gather when skip_lm_head is false.
                    self.assertFalse(skip)
                    self.assertEqual(calls, ["validate"])
                    self.assertEqual(mode, 0)
                    self.assertFalse(inplace)
                    self.assertEqual(tuple(send[0].shape), (2, 2))
                    t.testing.assert_close(send[0], t.zeros_like(send[0]))
                    receive[0].copy_(t.cat([send[0], send[0]], 0))
                    calls.append("logits")

                self.register_peers(allreduce, allgather)
                local_failure = failure if tp_rank == bad_rank else None
                model = self.model(
                    mtp=local_failure != "absent",
                    mtp_failure="exception" if local_failure == "exception" else None,
                    malformed="width" if local_failure == "packed" else None,
                )
                store_count = len(MockStore.instances)
                result = self.ext.run_model(
                    model,
                    cp=True,
                    tp_size=2,
                    tp_rank=tp_rank,
                    lengths=[1, 3],
                    skip_lm_head=skip,
                    input_ids_on_cuda=cuda_ids,
                )
                self.assertEqual(model.token_ids[0].numel(), 4)
                self.assertTrue(t.all(model.mtp_hidden != 0).item())
                self.assertEqual(tuple(result["hidden"].shape), (4, 2))
                t.testing.assert_close(result["hidden"], t.zeros_like(result["hidden"]))
                self.assertEqual(
                    calls, ["validate"] if skip else ["validate", "logits"]
                )
                if skip:
                    self.assertIsNone(result["logits"])
                else:
                    self.assertEqual(tuple(result["logits"].shape), (2, 4))
                    t.testing.assert_close(
                        result["logits"], t.zeros_like(result["logits"])
                    )
                self.assertIsNone(result["second_error"])
                self.assertEqual(
                    len(MockStore.instances), store_count + int(tp_rank == 0)
                )
                if tp_rank == 0:
                    self.assertIsNotNone(result["error"])
                    self.assertEqual(result["failures"], 1)
                    self.assertEqual(MockStore.instances[-1].batches, [])
                else:
                    self.assertIsNone(result["error"])
                    self.assertEqual(result["failures"], 0)

    def test_cp_all_ranks_without_mtp_tensor_skip_mtp_restore(self):
        for failure, tp_rank in product((None, "empty"), range(2)):
            with self.subTest(failure=failure, rank=tp_rank):
                calls = self.cp_peers(lengths=[6, 9], tp_rank=tp_rank)
                model = self.model(mtp_failure=failure)
                result = self.ext.run_model(
                    model, cp=True, tp_size=2, tp_rank=tp_rank, lengths=[6, 9]
                )
                self.assertEqual(calls, ["validate", "restore"])
                self.assertIsNone(result["error"])
                self.assertEqual(result["hidden"][:, 0].tolist(), list(range(31, 46)))

    def test_cp_packed_error_precedes_mtp_presence_disagreement(self):
        for bad_rank, tp_rank in product(range(2), repeat=2):
            with self.subTest(bad_rank=bad_rank, rank=tp_rank):
                calls = []

                def allreduce(tensor, op, mode, dest):
                    self.assertEqual((op, mode, dest), (0, 0, None))
                    self.assertEqual(
                        tensor.tolist(), [0, 1, 0] if tp_rank == bad_rank else [1, 1, 1]
                    )
                    calls.append("validate")
                    return tensor.new_tensor([1, 2, 1])

                self.register_peers(allreduce)
                store_count = len(MockStore.instances)
                model = self.model(
                    mtp=True, malformed="width" if tp_rank == bad_rank else None
                )
                result = self.ext.run_model(
                    model, cp=True, tp_size=2, tp_rank=tp_rank, lengths=[6, 9]
                )
                self.assertEqual(calls, ["validate"])
                self.assertEqual(
                    model.mtp_calls,
                    [] if tp_rank == bad_rank else [model.token_ids[0].numel()],
                )
                self.assertIsNone(result["second_error"])
                self.assertEqual(tuple(result["hidden"].shape), (15, 2))
                self.assertEqual(
                    len(MockStore.instances), store_count + int(tp_rank == 0)
                )
                if tp_rank == 0:
                    self.assertIn(
                        "width" if bad_rank == 0 else "another TP rank", result["error"]
                    )
                    self.assertEqual(MockStore.instances[-1].batches, [])
                else:
                    self.assertIsNone(result["error"])

    def test_cp_mtp_getter_failure_single_rank_is_deferred(self):
        def unexpected(*args):
            self.fail("single-rank capture must not communicate")

        self.register_peers(unexpected)
        for failure in ("exception", "rows", "rank", "width", "dtype", "device"):
            with self.subTest(failure=failure):
                result = self.ext.run_model(
                    self.model(mtp=True, mtp_failure=failure),
                    cp=True,
                    tp_size=1,
                    lengths=[6, 9],
                )
                if failure == "exception":
                    message = "injected MTP getter failure"
                elif failure in ("width", "dtype", "device"):
                    message = failure
                else:
                    message = "local capture rows"
                self.assertIn(message, result["error"])
                self.assertIsNone(result["second_error"])
                self.assertEqual(tuple(result["hidden"].shape), (15, 2))
                self.assertEqual(MockStore.instances[-1].batches, [])

    def test_capture_configuration_and_python_capability_validation(self):
        for layer_ids in ([0, 0], [-1, 1], [0, 2]):
            with self.subTest(layer_ids=layer_ids):
                with self.assertRaisesRegex(RuntimeError, "duplicate|outside"):
                    self.ext.run_model(self.model(), capture_layer_ids=layer_ids)
        model = self.model()
        model.supports_hidden_state_capture = False
        with self.assertRaisesRegex(
            RuntimeError, "does not support hidden-state capture"
        ):
            self.ext.run_model(model)
        self.assertEqual(MockStore.instances, [])

    def test_initialization_failures_do_not_leave_store_open(self):
        model = self.model()
        model.initialize = lambda resources: False
        with self.assertRaisesRegex(RuntimeError, "initialization failed"):
            self.ext.run_model(model)
        self.assertEqual(MockStore.instances, [])
        MockStore.failure = "init"
        with self.assertRaisesRegex(RuntimeError, "warmup failure"):
            self.ext.run_model(self.model())
        self.assertTrue(MockStore.instances[0].closed)

    def test_micro_batch_requires_final_norm_contract(self):
        model = self.model()
        model.micro_batch_final_norm = False
        with self.assertRaisesRegex(RuntimeError, "micro_batch_final_norm"):
            self.ext.run_model(model, micro_batch=True)
        self.assertTrue(MockStore.instances[0].closed)

    def test_micro_batch_requires_python_entry_point(self):
        model = self.model()
        del type(model).forward_micro_batch
        with self.assertRaisesRegex(
            RuntimeError, "requires Python model method forward_micro_batch"
        ):
            self.ext.run_model(model, micro_batch=True)
        self.assertEqual(MockStore.instances, [])

    def test_py_model_outputs_retains_python_attention_params(self):
        import gc
        import weakref

        class Params:
            pass

        params = Params()
        ref = weakref.ref(params)
        hidden = self.torch.ones((2, 2))
        output = self.ext.PyModelOutputs(hidden_states=hidden, params_ptr=params)
        del params
        gc.collect()
        self.assertIsNotNone(ref())
        self.torch.testing.assert_close(output.hidden_states, hidden)
        del output
        gc.collect()
        self.assertIsNone(ref())

    def test_fp8_model_publication_uses_real_quantizer(self):
        result = self.ext.run_model(self.model(), fp8=True, micro_batch=True)
        self.assertIsNone(result["error"])
        batch = MockStore.instances[0].batches[0]
        self.assertEqual(
            [x.dtype for x in batch["hidden_states"]], [self.torch.float8_e4m3fn] * 2
        )
        self.assertEqual(
            [tuple(x.shape) for x in batch["hidden_states_scale"]], [(2, 2), (3, 2)]
        )
        self.assertEqual(result["hidden"][:, 0].tolist(), [31, 32, 33, 34, 35])


class OldCacheStoreWrapperTest(CaptureFixture):
    """Load the complete legacy module; feed it native PyWrappedModel inputs."""

    def setUp(self):
        super().setUp()
        ops = types.ModuleType("rtp_llm.ops")
        ops.__path__ = []
        compute = types.ModuleType("rtp_llm.ops.compute_ops")
        for name in (
            "KVCache",
            "LayerKVCache",
            "PyAttentionInputs",
            "PyCacheStoreInputs",
        ):
            setattr(compute, name, getattr(self.ext, name))

        def write_cache_store(lengths, prefix, blocks, params, cache):
            return None

        self.write = Mock(wraps=write_cache_store)
        compute.write_cache_store = self.write
        ops.compute_ops = compute
        path = (
            Path(__file__).resolve().parents[4]
            / "rtp_llm/models_py/modules/base/common/kvcache_store.py"
        )
        spec = importlib.util.spec_from_file_location("legacy_kvcache_store_test", path)
        module = importlib.util.module_from_spec(spec)
        # Isolate package bootstrap, which otherwise loads a second copy of the
        # pybind types. Execute the whole module, including its real imports.
        with patch.dict(
            sys.modules, {"rtp_llm.ops": ops, "rtp_llm.ops.compute_ops": compute}
        ):
            spec.loader.exec_module(module)
        self.create = module.create_write_cache_store_impl

    def test_capture_skip_has_no_writer(self):
        model = self.model(writer_factory=self.create)
        self.ext.run_model(model)
        self.assertEqual(model.writers, [None])
        self.write.assert_not_called()

    def test_decode_has_no_writer(self):
        model = self.model(writer_factory=self.create)
        self.ext.run_model(model, capture=False, skip_lm_head=False, decode=True)
        self.assertEqual(model.writers, [None])
        self.write.assert_not_called()

    def test_old_five_argument_call_uses_native_host_metadata(self):
        model = self.model(writer_factory=self.create)
        self.ext.run_model(model, capture=False, skip_lm_head=False)
        self.write.assert_called_once()
        lengths, prefix, blocks, params, cache = self.write.call_args.args
        self.assertEqual(lengths.tolist(), [2, 3])
        self.assertEqual(prefix.tolist(), [0, 0])
        self.assertTrue(lengths.is_pinned())
        self.assertTrue(prefix.is_pinned())
        self.assertEqual(lengths.data_ptr(), params.input_lengths_host.data_ptr())
        self.assertEqual(prefix.data_ptr(), params.prefix_lengths_host.data_ptr())
        self.assertEqual(
            blocks.data_ptr(),
            model.attention_inputs[0].kv_cache_block_id_host.data_ptr(),
        )
        self.assertIsNone(cache)

    def test_cp_uses_original_lengths_not_local_split(self):
        calls = self.cp_peers(logits=True)
        model = self.model(writer_factory=self.create)
        result = self.ext.run_model(
            model, cp=True, tp_size=2, lengths=[6], skip_lm_head=False
        )
        self.assertIsNone(result["error"])
        self.assertEqual(calls, ["validate", "restore", "logits"])
        self.write.assert_called_once()
        lengths, prefix, blocks, params, cache = self.write.call_args.args
        self.assertEqual(lengths.tolist(), [6])
        self.assertEqual(params.input_lengths_host.tolist(), [4])
        self.assertEqual(model.attention_inputs[0].input_lengths.tolist(), [4])


if __name__ == "__main__":
    unittest.main()
