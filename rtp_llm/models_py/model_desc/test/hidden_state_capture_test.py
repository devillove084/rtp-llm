"""Capture behavior with real CPU tensors, model hooks and native I/O bindings.

The accelerator decoder and fused residual norm are replaced at construction,
not the model forward or capture implementation. EmbeddingTorch/RMSNormTorch
are the same reference modules used by the neighboring operator tests. These
tests require the normal RTP Python/native dependencies; they do not run kernels.
"""

import os
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc import generic_moe
from rtp_llm.models_py.model_desc.generic_moe import DecodeLayerOutput, GenericMoeModel
from rtp_llm.models_py.model_desc.generic_moe_mtp import GenericMoeMTPModel
from rtp_llm.models_py.model_desc.hidden_state_capture import CaptureContext
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.base.common.embedding import EmbeddingTorch
from rtp_llm.models_py.modules.base.common.norm import RMSNormTorch
from rtp_llm.models_py.modules.dsv4 import _record_tensor
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import (
    ParamsBase,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
    SparseMlaParams,
)
from rtp_llm.utils.model_weight import W


class ResidualRMSNorm(nn.Module):
    """Adapt the existing CPU reference norm to the fused norm's tuple API."""

    def __init__(self, weight, eps=1e-6):
        super().__init__()
        self.norm = RMSNormTorch(weight, eps)

    def forward(self, hidden, residual):
        combined = hidden + residual
        return self.norm(combined), combined


class TensorDecoder(nn.Module):
    """Small decoder boundary: deterministic tensors instead of attention/MoE."""

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.cmp = None
        self.self_attn = SimpleNamespace(has_indexer=layer_id == 0)
        self.calls = []

    def forward(self, hidden, residual, fmha_impl, **kwargs):
        self.calls.append((fmha_impl, kwargs))
        topk = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32, device=hidden.device)
        return DecodeLayerOutput(
            hidden + self.layer_id + 1,
            residual + (self.layer_id + 1) * 2,
            topk[: hidden.shape[0]],
        )


class LayeredCacheBoundary:
    """Observe the model's cache lifecycle without allocating CUDA staging."""

    def __init__(self, events):
        self.events = events
        self.active = False

    def begin_forward(self, fmha_impl):
        self.events.append("begin")
        self.active = True

    def before_layer(self, layer_id):
        self.events.append(("before", layer_id))

    def end_forward(self, *, failed):
        self.events.append(("end", failed))
        self.active = False


class PositionParams(ParamsBase):
    """Native params carrier with CPU positions, without an FMHA prepare kernel."""

    def __init__(self):
        super().__init__()
        self.positions_d = torch.tensor([2, 3], dtype=torch.int32, device="cpu")


class CpuMTPModel(GenericMoeMTPModel):
    """Use the real MTP forward without its CUDA-only weight/buffer constructor."""

    def __init__(self, config, parallelism, weights):
        GptModelBase.__init__(self, config, parallelism, weights, 2)
        self.embed_tokens = EmbeddingTorch(weights.get_global_weight(W.embedding))
        norm_weight = weights.get_global_weight(W.final_ln_gamma)
        self.pre_fc_norm_embedding = RMSNormTorch(norm_weight)
        self.pre_fc_norm_hidden = RMSNormTorch(norm_weight)
        self.fc = nn.Linear(8, 4, bias=False, device="cpu")
        with torch.no_grad():
            identity = torch.eye(4, device="cpu")
            self.fc.weight.copy_(torch.cat([identity, 2 * identity], dim=-1))
        self.norm = ResidualRMSNorm(norm_weight)
        self.layers = nn.ModuleList([TensorDecoder(0)])
        self._mtp_indexer_share_enabled = True
        self._mtp_indexer_role = 0
        self._mtp_shared_topk_indices = torch.full(
            (2, 2), -1, dtype=torch.int32, device="cpu"
        )


class HiddenStateCaptureTest(unittest.TestCase):
    def setUp(self):
        # Diagnostic output is an I/O boundary, not part of capture numerics.
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(torch.device("cpu"))
        stack.enter_context(patch.object(_record_tensor, "ENABLED", False))
        stack.enter_context(
            patch.dict(os.environ, {"RTP_LLM_PREFILL_CACHE_STATS": "0"})
        )

    def make_config(self, layer_ids=(1, 0), num_layers=2):
        config = ModelConfig()
        config.num_layers = num_layers
        config.vocab_size = 3
        config.hidden_size = 4
        config.layernorm_eps = 1e-6
        config.hidden_state_capture_layer_ids = list(layer_ids)
        config.attn_config.indexer_topk = 2
        return config

    def make_weights(self, dtype=torch.float32):
        weights = ModelWeights(2, "cpu", dtype)
        weights.global_weights[W.embedding] = torch.tensor(
            [[1, -2, 3, 4], [2, 3, -4, 5], [3, 1, 2, -1]],
            dtype=dtype,
            device="cpu",
        )
        weights.global_weights[W.final_ln_gamma] = torch.tensor(
            [1, 2, 0.5, 1.5], dtype=dtype, device="cpu"
        )
        return weights

    def make_model(self, layer_ids=(1, 0), dtype=torch.float32):
        # Exercise the real constructor's capture registration and ModuleList.
        # Only factories that would construct accelerator operators are replaced.
        with (
            patch.object(
                generic_moe,
                "Embedding",
                side_effect=lambda config, parallelism, weight: EmbeddingTorch(weight),
            ),
            patch.object(
                generic_moe,
                "GenericMoeDecoderLayer",
                side_effect=lambda *args, **kwargs: TensorDecoder(args[4]),
            ),
            patch.object(generic_moe, "RMSResNorm", ResidualRMSNorm),
        ):
            return GenericMoeModel(
                self.make_config(layer_ids),
                ParallelismConfig(),
                self.make_weights(dtype),
                None,
                2,
            )

    def make_inputs(self, capture=True, rows=2, prefill=True):
        attention = PyAttentionInputs()
        attention.is_prefill = prefill
        attention.kv_cache_kernel_block_id_device_by_group = [
            torch.tensor([[10]], dtype=torch.int32, device="cpu"),
            torch.tensor([[20]], dtype=torch.int32, device="cpu"),
        ]
        attention.kv_cache_layer_to_group = torch.tensor(
            [1, 0], dtype=torch.int32, device="cpu"
        )
        attention.mtp_indexer_seed_rows = torch.tensor(
            [1, 0][:rows], dtype=torch.int32, device="cpu"
        )
        inputs = PyModelInputs()
        inputs.input_ids = torch.tensor([0, 1][:rows], dtype=torch.int32, device="cpu")
        inputs.attention_inputs = attention
        inputs.capture_hidden_states = capture
        return inputs

    def make_fmha(self, inputs, params=None):
        # The attention implementation is a boundary; its metadata/output carrier
        # are real pybind objects, including GLM's SparseMlaParams.
        return SimpleNamespace(
            fmha_params=SparseMlaParams() if params is None else params,
            attn_inputs=inputs.attention_inputs,
        )

    def assert_tensor(self, actual, expected):
        self.assertIsInstance(actual, torch.Tensor)
        torch.testing.assert_close(
            actual, expected, check_dtype=True, check_device=True
        )

    def expected_final(self, combined, model):
        return torch.nn.functional.rms_norm(
            combined.float(),
            (combined.shape[-1],),
            eps=model.config.layernorm_eps,
        ).to(combined.dtype) * model.weight.get_global_weight(W.final_ln_gamma)

    def test_glm_capture_order_first_last_final_norm_dtype_device_and_metadata(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for layer_ids in ((0,), (1,), (0, 1), (1, 0)):
                with self.subTest(dtype=dtype, layer_ids=layer_ids):
                    model = self.make_model(layer_ids, dtype)
                    inputs = self.make_inputs()
                    fmha = self.make_fmha(inputs)
                    embedded = model.embed_tokens(inputs.input_ids)
                    final = self.expected_final(embedded + 9, model)
                    norm_calls = []
                    with model.norm.register_forward_hook(
                        lambda module, args, output, calls=norm_calls: calls.append(
                            output[0]
                        )
                    ):
                        output = model(inputs, fmha)
                    expected = torch.cat(
                        [embedded + (3 if i == 0 else 9) for i in layer_ids] + [final],
                        dim=-1,
                    )
                    self.assertIsInstance(output, PyModelOutputs)
                    self.assert_tensor(output.hidden_states, expected)
                    self.assertIs(output.params_ptr, fmha.fmha_params)
                    self.assertIs(fmha.pinned_mla_groups, model.pinned_mla_groups)
                    self.assertTrue(model.supports_hidden_state_capture)
                    self.assertEqual(model._capture_context.layer_ids, layer_ids)
                    self.assertEqual(len(norm_calls), 1)
                    self.assert_tensor(norm_calls[0], final)

    def test_explicit_flag_is_not_inferred_from_prefill_and_no_stale_storage(self):
        model = self.make_model()
        previous_capture = None
        for enabled, prefill, rows in (
            (False, True, 2),
            (True, True, 2),
            (False, True, 1),
            (False, False, 2),
            (True, False, 1),
            (True, True, 0),
            (False, True, 0),
        ):
            with self.subTest(enabled=enabled, prefill=prefill, rows=rows):
                inputs = self.make_inputs(enabled, rows, prefill)
                embedded = model.embed_tokens(inputs.input_ids)
                final = self.expected_final(embedded + 9, model)
                output = model(inputs, self.make_fmha(inputs)).hidden_states
                expected = (
                    torch.cat([embedded + 9, embedded + 3, final], dim=-1)
                    if enabled
                    else final
                )
                self.assert_tensor(output, expected)
                if previous_capture is not None:
                    self.assert_tensor(previous_capture[0], previous_capture[1])
                if enabled:
                    previous_capture = output, output.clone()
        self.assertIsNone(model._capture_context._captured_hidden_states)
        self.assertIs(model.capture_context(False), model._capture_context)
        self.assertIsNot(model.capture_context(True), model.capture_context(True))

    def test_disabled_capture_never_calls_layer_hook_and_keeps_final_storage(self):
        model = self.make_model((0,))
        root = model.capture_context(False)
        hidden = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device="cpu")
        residual = torch.ones_like(hidden)
        # The real layer hook rejects a missing residual if it is ever called.
        root.capture_layer(0, hidden)
        self.assertFalse(root.wants_layer(0))
        final_outputs = []
        with model.norm.register_forward_hook(
            lambda module, args, output: final_outputs.append(output[0])
        ):
            output = root.finalize(hidden, residual)
        self.assertEqual(output.hidden_states.data_ptr(), final_outputs[0].data_ptr())
        self.assertIsNone(root._captured_hidden_states)

    def test_empty_plan_and_disabled_model_still_have_distinct_capabilities(self):
        model = self.make_model(())
        inputs = self.make_inputs()
        output = model(inputs, self.make_fmha(inputs))
        self.assert_tensor(
            output.hidden_states,
            self.expected_final(model.embed_tokens(inputs.input_ids) + 9, model),
        )
        self.assertTrue(model.supports_hidden_state_capture)
        self.assertFalse(model.capture_context(True).enabled)
        model._disable_capture_context()
        self.assertFalse(model.supports_hidden_state_capture)
        self.assertIs(model.capture_context(True), CaptureContext.unsupported())
        model._init_capture_passthrough()
        self.assertTrue(model.supports_hidden_state_capture)
        self.assertFalse(model.capture_context(True).enabled)

    def test_contexts_have_independent_storage_and_report_only_missing_layers(self):
        model = self.make_model()
        first, second = model.capture_context(True), model.capture_context(True)
        hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        residual = torch.ones_like(hidden)
        first.capture_layer(99, hidden)  # No residual: an unselected hook must not run.
        first.capture_layer(0, hidden, residual)
        first.capture_layer(1, hidden + 1, residual)
        self.assert_tensor(
            first.finalize(hidden, residual).hidden_states,
            torch.cat(
                [hidden + 2, hidden + 1, self.expected_final(hidden + 1, model)], -1
            ),
        )
        with self.assertRaisesRegex(RuntimeError, r"configured layers \[1, 0\]"):
            second.finalize(hidden, residual)
        second.capture_layer(1, hidden, residual)
        with self.assertRaisesRegex(RuntimeError, r"configured layers \[0\]"):
            second.finalize(hidden, residual)
        self.assertTrue(first.wants_layer(0))
        self.assertTrue(first.wants_layer(1))
        self.assertFalse(first.wants_layer(99))
        self.assertFalse(model.capture_context(False).wants_layer(0))

    def test_auxiliary_tensor_survives_in_place_hidden_and_residual_mutation(self):
        model = self.make_model((0,))
        capture = model.capture_context(True)
        hidden = torch.arange(16, dtype=torch.float32).reshape(2, 8)[:, ::2]
        residual = torch.ones_like(hidden)
        self.assertFalse(hidden.is_contiguous())
        expected = (hidden + residual).clone()
        capture.capture_layer(0, hidden, residual)
        hidden.add_(10)
        residual.mul_(3)
        output = capture.finalize(hidden, residual).hidden_states
        self.assert_tensor(output[:, :4], expected)
        self.assert_tensor(output[:, 4:], self.expected_final(hidden + residual, model))

    def test_real_hooks_reject_missing_residual(self):
        model = self.make_model()
        for hook in (model._capture_canonical_layer, model._capture_canonical_final):
            with (
                self.subTest(hook=hook.__name__),
                self.assertRaisesRegex(ValueError, "residual"),
            ):
                hook(torch.ones(2, 4), None)

    def test_unsupported_and_passthrough_preserve_tensor_and_native_params(self):
        hidden = torch.ones(2, 4)
        params = ParamsBase()
        for root in (CaptureContext.unsupported(), CaptureContext.passthrough()):
            for requested in (False, True):
                with self.subTest(supported=root.supported, requested=requested):
                    capture = root.for_forward(requested)
                    self.assertIs(capture, root)
                    self.assertFalse(capture.enabled)
                    capture.capture_layer(0, hidden)
                    output = capture.finalize(hidden, fmha_params=params)
                    self.assertEqual(output.hidden_states.data_ptr(), hidden.data_ptr())
                    self.assertIs(output.params_ptr, params)
        self.assertFalse(CaptureContext.unsupported().supported)
        self.assertTrue(CaptureContext.passthrough().supported)

    def test_empty_context_plan_keeps_final_hook_and_none_metadata(self):
        model = self.make_model()
        hidden = torch.ones(2, 4)
        residual = torch.full_like(hidden, 2)
        for layer_ids in (None, ()):
            with self.subTest(layer_ids=layer_ids):
                root = CaptureContext.configured(
                    layer_ids,
                    model._capture_canonical_layer,
                    model._capture_canonical_final,
                )
                self.assertIs(root.for_forward(True), root)
                self.assertFalse(root.enabled)
                output = root.finalize(hidden, residual)
                self.assert_tensor(
                    output.hidden_states, self.expected_final(hidden + residual, model)
                )
                self.assertIsNone(output.params_ptr)

    def test_malformed_active_context_reports_missing_hooks_or_storage(self):
        model = self.make_model()
        hidden = torch.ones(2, 4)
        with self.assertRaisesRegex(RuntimeError, "passthrough-only model"):
            CaptureContext(ordered_layer_ids=(0,), supported=True).for_forward(True)
        with self.assertRaisesRegex(RuntimeError, "missing canonical hooks"):
            CaptureContext(enabled=True, layer_slots={0: 0}).capture_layer(0, hidden)
        with self.assertRaisesRegex(RuntimeError, "missing canonical final hook"):
            CaptureContext(enabled=True).finalize(hidden)
        with self.assertRaisesRegex(RuntimeError, "no forward-local storage"):
            CaptureContext(
                enabled=True, canonical_final=model._capture_canonical_final
            ).finalize(hidden, hidden)

    def test_layered_cache_cleanup_on_success_and_each_forward_failure(self):
        for failure in (
            None,
            "begin",
            "embed",
            "before_layer",
            "layer",
            "missing",
            "norm",
        ):
            with self.subTest(failure=failure):
                events = []
                model = self.make_model()
                if failure == "missing":
                    model.layer_num = 1
                cache = LayeredCacheBoundary(events)
                model.layered_prefill_cache = cache
                diagnostics = SimpleNamespace(finish=Mock())
                model._prefill_cache_diagnostics = diagnostics
                inputs = self.make_inputs()
                fmha = self.make_fmha(inputs)
                with model.embed_tokens.register_forward_pre_hook(
                    lambda module, args, events=events: events.append("embed")
                ):
                    if failure in ("begin", "embed", "before_layer", "layer", "norm"):
                        boundary, method = {
                            "begin": (cache, "begin_forward"),
                            "embed": (model.embed_tokens, "forward"),
                            "before_layer": (cache, "before_layer"),
                            "layer": (model.layers[1], "forward"),
                            "norm": (model.norm, "forward"),
                        }[failure]

                        original = getattr(boundary, method)

                        def fail(*args, failure=failure, original=original, **kwargs):
                            if failure == "begin":
                                original(*args, **kwargs)
                            raise RuntimeError("injected boundary failure")

                        with (
                            patch.object(boundary, method, side_effect=fail),
                            self.assertRaisesRegex(RuntimeError, "boundary failure"),
                        ):
                            model(inputs, fmha)
                    elif failure == "missing":
                        with self.assertRaisesRegex(
                            RuntimeError, r"configured layers \[1\]"
                        ):
                            model(inputs, fmha)
                    else:
                        model(inputs, fmha)
                self.assertEqual(events[-1], ("end", failure is not None))
                self.assertFalse(cache.active)
                diagnostics.finish.assert_called_once_with(cache, failure is not None)
                if failure is None:
                    self.assertEqual(
                        events,
                        [
                            "begin",
                            "embed",
                            ("before", 0),
                            ("before", 1),
                            ("end", False),
                        ],
                    )
                    self.assertIs(fmha.layered_prefill_cache, cache)
                # A failed forward must not leave capture storage/cache state for retry.
                model.layer_num = 2
                events.clear()
                retry = model(inputs, fmha)
                embedded = model.embed_tokens(inputs.input_ids)
                self.assert_tensor(
                    retry.hidden_states,
                    torch.cat(
                        [
                            embedded + 9,
                            embedded + 3,
                            self.expected_final(embedded + 9, model),
                        ],
                        dim=-1,
                    ),
                )
                self.assertEqual(events[-1], ("end", False))

    def test_fmha_preparation_and_layer_metadata_use_real_block_map_helpers(self):
        for layered in (False, True):
            with self.subTest(layered=layered):
                model = self.make_model()
                inputs = self.make_inputs()
                fmha = self.make_fmha(inputs)
                if layered:
                    model.layered_prefill_cache = LayeredCacheBoundary([])
                block_maps = []
                handles = [
                    layer.register_forward_pre_hook(
                        lambda module, args, maps=block_maps, inputs=inputs: (
                            maps.append(
                                inputs.attention_inputs.kv_cache_kernel_block_id_device.clone()
                            )
                        )
                    )
                    for layer in model.layers
                ]
                try:
                    with patch.object(
                        model, "prepare_fmha_impl", return_value=fmha
                    ) as prepare:
                        output = model(inputs)
                        prepare.assert_called_once_with(inputs)
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertIs(output.params_ptr, fmha.fmha_params)
                groups = (
                    inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group
                )
                self.assert_tensor(block_maps[0], groups[1])
                self.assert_tensor(block_maps[1], groups[0])
                for layer in model.layers:
                    self.assertIs(layer.calls[0][0], fmha)
                    self.assertFalse(layer.calls[0][1]["enable_cmp"])

    def test_trace_and_diagnostics_receive_unpacked_final_norm_and_indexer(self):
        model = self.make_model()
        inputs = self.make_inputs()
        fmha = self.make_fmha(inputs)
        diagnostics = SimpleNamespace(begin=Mock(), indexer=Mock(), finish=Mock())
        model._prefill_cache_diagnostics = diagnostics
        with (
            patch.dict(os.environ, {"RTP_LLM_PREFILL_CACHE_STATS": "1"}),
            patch.object(_record_tensor, "ENABLED", True),
            patch.object(_record_tensor, "_DBG_MAX_SEQ", 0),
            patch.object(_record_tensor, "_DBG_NAME_RE", None),
            patch.object(_record_tensor, "_TAIL_TOKENS", 0),
            patch.object(_record_tensor, "_STREAM", False),
            patch.object(_record_tensor._local, "buf", None, create=True),
            patch.object(_record_tensor, "dump") as dump,
        ):
            output = model(inputs, fmha)
            records = dict(_record_tensor._get_buf())
        embedded = model.embed_tokens(inputs.input_ids)
        self.assert_tensor(records["layer00_combined"], embedded + 3)
        self.assert_tensor(records["layer01_combined"], embedded + 9)
        self.assert_tensor(records["final_norm"], output.hidden_states[:, -4:])
        self.assertEqual(records["final_norm"].shape, (2, 4))
        self.assertEqual(model._dbg_step, 1)
        dump.assert_called_once()
        diagnostics.begin.assert_called_once_with(fmha)
        diagnostics.finish.assert_called_once_with(None, False)
        diagnostics.indexer.assert_called_once()
        self.assertEqual(diagnostics.indexer.call_args.args[0], 0)
        self.assert_tensor(
            diagnostics.indexer.call_args.args[1],
            torch.tensor([[0, 1], [1, 2]], dtype=torch.int32),
        )

    def test_ordinary_target_output_and_mtp_handoff_seed_reuse_are_unchanged(self):
        target = self.make_model()
        inputs = self.make_inputs(capture=False)
        fmha = self.make_fmha(inputs, PositionParams())
        ordinary = target(inputs, fmha).hidden_states
        inputs.capture_hidden_states = True
        packed = target(inputs, fmha).hidden_states
        self.assert_tensor(ordinary, packed[:, -4:])
        self.assertEqual(ordinary.shape, (2, 4))
        inputs.input_hiddens = ordinary
        for role, prefill in ((0, False), (1, False), (2, False), (2, True)):
            with self.subTest(role=role, prefill=prefill):
                draft = CpuMTPModel(
                    self.make_config((0,), num_layers=1),
                    ParallelismConfig(),
                    self.make_weights(),
                )
                draft.set_mtp_indexer_role(role)
                seed = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)
                draft.load_mtp_indexer_topk(seed)
                inputs.attention_inputs.is_prefill = prefill
                self.assertFalse(draft.supports_hidden_state_capture)
                observed_hidden = []
                with draft.pre_fc_norm_hidden.register_forward_pre_hook(
                    lambda module, args, observed=observed_hidden: observed.append(
                        args[0]
                    )
                ):
                    output = draft(inputs, fmha)
                self.assertEqual(observed_hidden[0].data_ptr(), ordinary.data_ptr())
                self.assertIs(output.params_ptr, fmha.fmha_params)
                self.assertEqual(output.hidden_states.shape, ordinary.shape)
                embedded = draft.embed_tokens(inputs.input_ids)
                projected = draft.fc(
                    torch.cat(
                        [
                            draft.pre_fc_norm_embedding(embedded),
                            draft.pre_fc_norm_hidden(ordinary),
                        ],
                        dim=-1,
                    )
                )
                self.assert_tensor(
                    output.hidden_states, self.expected_final(projected + 3, draft)
                )
                kwargs = draft.layers[0].calls[0][1]
                reuse = role == 2 and not prefill
                self.assertEqual(kwargs["force_reuse_topk_indices"], reuse)
                if reuse:
                    expected_topk = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32)
                    self.assert_tensor(kwargs["prev_topk_indices"], expected_topk)
                else:
                    expected_topk = seed.flip(0) if role == 1 else seed
                    self.assertIsNone(kwargs["prev_topk_indices"])
                self.assert_tensor(draft.snapshot_mtp_indexer_topk(2), expected_topk)
                # The draft never packs even if the target request flag remains set.
                inputs.capture_hidden_states = False
                ordinary_draft = draft(inputs, fmha)
                self.assert_tensor(ordinary_draft.hidden_states, output.hidden_states)
                inputs.capture_hidden_states = True


if __name__ == "__main__":
    unittest.main()
