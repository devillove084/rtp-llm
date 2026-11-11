from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.config.qwen4_exp_config import Qwen4ModelConfig as ModelConfig
from rtp_llm.models.qwen3_next.qwen3_next import Qwen3NextBase
from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_ROPE_POS_TAG,
    INDEXER_STATE_TAG,
    PLE_NGRAM_CTX_TAG,
    PLE_STATE_TAG,
    build_qwen4_exp_kv_cache_spec_descs,
)
from rtp_llm.models_py.model_desc import generic_moe, qwen3_next, qwen4_exp
from rtp_llm.models_py.model_desc.hidden_state_capture import CaptureContext
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.ple_cache_views import PLECacheViews
from rtp_llm.models_py.model_desc.test.qwen4_exp.fixtures import (
    _EPS,
    _EXPERTS,
    _HC,
    _HIDDEN,
    _Attention,
    _bare_model,
    _hc_weights,
    _hybrid_types,
    _RoutedBackend,
    _select_topk,
    _spec_config,
)
from rtp_llm.models_py.modules.hybrid.gated_residual import (
    GatedResidual,
    inject_into_residual,
)
from rtp_llm.ops import HybridAttentionType, MoeConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs
from rtp_llm.utils.model_weight import W


class Qwen4ExpDecoderLayerTest(TestCase):
    def setUp(self):
        self.hc_hidden = _HC * _HIDDEN
        self.config = ModelConfig()
        self.config.num_layers = 1
        self.config.hidden_size = _HIDDEN
        self.config.max_seq_len = 16
        self.config.layernorm_eps = _EPS
        self.config.activation_type = "SiGLU"
        self.config.hc_mult = _HC
        self.config.hybrid_attention_config.hybrid_attention_types = [
            HybridAttentionType.LINEAR
        ]
        Qwen3NextBase._parse_moe_config(
            {
                "num_experts": _EXPERTS,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 4,
                "shared_expert_intermediate_size": 0,
            },
            self.config,
        )

        self.attn_hc = _hc_weights(0)
        self.mlp_hc = _hc_weights(1)
        self.weights = {
            W.qwen4_hc_attn_norm: self.attn_hc[0],
            W.qwen4_hc_attn_mix_down: self.attn_hc[1],
            W.qwen4_hc_attn_mix_up: self.attn_hc[2],
            W.qwen4_hc_attn_inject: self.attn_hc[3],
            W.qwen4_hc_mlp_norm: self.mlp_hc[0],
            W.qwen4_hc_mlp_mix_down: self.mlp_hc[1],
            W.qwen4_hc_mlp_mix_up: self.mlp_hc[2],
            W.qwen4_hc_mlp_inject: self.mlp_hc[3],
            W.moe_gate: torch.ones(_EXPERTS, _HIDDEN),
            W.moe_w1: torch.ones(_EXPERTS, _HIDDEN, _HIDDEN),
            W.moe_w2: torch.ones(_EXPERTS, _HIDDEN, 4),
        }

    def _run(self, hyper_states):
        attention = _Attention()
        with (
            patch.object(qwen3_next, "Qwen3NextGatedDeltaNet", return_value=attention),
            patch.object(
                generic_moe.LinearFactory,
                "create_linear_from_weights",
                return_value=nn.Linear(_HIDDEN, _EXPERTS, bias=False),
            ),
            patch.object(generic_moe, "SelectTopk", return_value=_select_topk),
            patch.object(
                generic_moe.FusedMoeFactory,
                "create_fused_moe",
                return_value=_RoutedBackend(),
            ),
        ):
            layer = qwen4_exp.Qwen4ExpDecoderLayer(
                self.config, ParallelismConfig(), self.weights, 0, MoeConfig()
            )
            output = layer(hyper_states, None)
        return layer, attention, output

    def test_wiring_matches_a_hand_rolled_two_stage_reference(self):
        hyper_states = torch.randn(3, self.hc_hidden)

        _, attention, output = self._run(hyper_states)

        attn_unit = GatedResidual(*self.attn_hc, hc_mult=_HC, norm_eps=_EPS)
        mlp_unit = GatedResidual(*self.mlp_hc, hc_mult=_HC, norm_eps=_EPS)
        x, hyper, w = attn_unit(hyper_states)
        hyper = inject_into_residual(hyper, x * 3.0, w)
        x, hyper, w = mlp_unit(hyper)
        expected = inject_into_residual(hyper, torch.ones_like(x), w)

        self.assertEqual(attention.seen.shape, (3, _HIDDEN))
        torch.testing.assert_close(output, expected)

    def test_rejects_a_narrow_stream(self):
        with self.assertRaisesRegex(ValueError, "gated residual features"):
            self._run(torch.randn(3, _HIDDEN))

    def test_linear_layer_forwards_the_configured_norm_activation(self):
        # Qwen4-Exp checkpoints gate the GDN output norm with output_gate_type
        # ("sigmoid"); the layer must forward it instead of the silu default.
        self.config.linear_attn_norm_activation = "sigmoid"
        attention = _Attention()
        with (
            patch.object(
                qwen3_next, "Qwen3NextGatedDeltaNet", return_value=attention
            ) as gdn_cls,
            patch.object(
                generic_moe.LinearFactory,
                "create_linear_from_weights",
                return_value=nn.Linear(_HIDDEN, _EXPERTS, bias=False),
            ),
            patch.object(generic_moe, "SelectTopk", return_value=_select_topk),
            patch.object(
                generic_moe.FusedMoeFactory,
                "create_fused_moe",
                return_value=_RoutedBackend(),
            ),
        ):
            qwen4_exp.Qwen4ExpDecoderLayer(
                self.config, ParallelismConfig(), self.weights, 0, MoeConfig()
            )
        self.assertEqual(gdn_cls.call_args.kwargs.get("norm_activation"), "sigmoid")


class Qwen4ExpMetadataTest(TestCase):
    def test_target_transactions_require_enabled_ple_not_method_presence(self):
        from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpModel

        model = SimpleNamespace(ple_layers={}, config=SimpleNamespace(is_mtp=False))
        self.assertFalse(Qwen4ExpModel.requires_speculative_target_commit(model))
        model.ple_layers = {"1": object()}
        self.assertTrue(Qwen4ExpModel.requires_speculative_target_commit(model))
        for role in ("DECODE", "PDFUSION", "PREFILL"):
            model.parallelism_config = SimpleNamespace(role_type=role)
            self.assertEqual(
                Qwen4ExpModel.requires_speculative_target_commit(model),
                role != "PREFILL",
            )
        model.config.is_mtp = True
        self.assertFalse(Qwen4ExpModel.requires_speculative_target_commit(model))

    def test_ple_region_alias_avoids_equality_sync_but_checks_distinct_values(self):
        values = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        left = SimpleNamespace(input_lengths=values)
        alias = SimpleNamespace(input_lengths=values.view_as(values))
        different = SimpleNamespace(input_lengths=values + 1)
        with patch.object(torch, "equal", wraps=torch.equal) as equal:
            self.assertTrue(PLECacheViews._same_metadata(left, alias, "input_lengths"))
            equal.assert_not_called()
            self.assertFalse(
                PLECacheViews._same_metadata(left, different, "input_lengths")
            )
            equal.assert_called_once()


class Qwen4ExpModelConstructionTest(TestCase):
    """The real construction path: GptModelBase.__init__ then Qwen4ExpModel.

    Every other test in this file fakes the base init, which hid a startup
    regression: GptModelBase already owns `_mtp_target_hidden_states` as a plain
    attribute, so re-registering it as a buffer raised
    "attribute ... already exists" on every server start.
    """

    class _FakeModelWeights:
        def __init__(self, embedding, mixer_norm, mix_down, mix_up):
            self._globals = {
                W.embedding: embedding,
                W.qwen4_hc_mixer_norm: mixer_norm,
                W.qwen4_hc_mixer_mix_down: mix_down,
                W.qwen4_hc_mixer_mix_up: mix_up,
            }
            self.global_weights = self._globals
            self.weights: list = []

        def get_global_weight(self, name):
            return self._globals[name]

    def test_model_init_reuses_the_base_mtp_capture_attributes(self):
        hidden, vocab = _HIDDEN, 16
        hc_hidden = _HC * hidden
        config = ModelConfig()
        config.num_layers = 0
        config.hidden_size = hidden
        config.vocab_size = vocab
        config.hc_mult = _HC
        config.layernorm_eps = _EPS
        config.capture_aux_hidden_layer_ids = None
        mixer_norm, mix_down, mix_up, _ = _hc_weights(0)
        weights = self._FakeModelWeights(
            torch.zeros(vocab, hidden),
            mixer_norm,
            mix_down,
            mix_up,
        )

        model = qwen4_exp.Qwen4ExpModel(
            config, ParallelismConfig(), weights, MoeConfig(), 0
        )

        self.assertIsNone(model._mtp_target_hidden_states)
        self.assertFalse(model._capture_mtp_target_hidden)
        self.assertTrue(model.supports_hidden_state_capture)
        self.assertIs(model.capture_context(False), model._capture_context)
        self.assertEqual(model.hyper_connection_mixer.hc_hidden_size, hc_hidden)


class Qwen4ExpCanonicalCaptureTest(TestCase):
    """CPU tensor wiring, not native/FP8/Graph or training-recipe parity."""

    class _TensorOutputs:
        """Model the non-optional native MTP setter, not RTP's native ABI.

        The default read is None (undefined Tensor); assigning None is invalid.
        Optional capture fields deliberately remain independent of that setter.
        """

        def __init__(self, hidden_states):
            self.hidden_states = hidden_states
            self.captured_hidden_states = None
            self.capture_error = None
            self._mtp_target_hidden_states = None
            self.mtp_assignments = []

        @property
        def mtp_target_hidden_states(self):
            return self._mtp_target_hidden_states

        @mtp_target_hidden_states.setter
        def mtp_target_hidden_states(self, value):
            if not isinstance(value, torch.Tensor):
                raise TypeError("mtp_target_hidden_states requires torch.Tensor")
            self.mtp_assignments.append(value)
            self._mtp_target_hidden_states = value

    def setUp(self):
        # Scope the stricter output contract to this test case, including the
        # real CaptureContext.finalize; never replace torch or other test modules.
        outputs_patch = patch.dict(
            CaptureContext.finalize.__globals__, PyModelOutputs=self._TensorOutputs
        )
        outputs_patch.start()
        self.addCleanup(outputs_patch.stop)

    def test_mtp_output_setter_rejects_none_but_keeps_tensor_identity(self):
        output = self._TensorOutputs(torch.ones(1, _HIDDEN))
        self.assertIsNone(output.mtp_target_hidden_states)
        with self.assertRaisesRegex(TypeError, "requires torch.Tensor"):
            output.mtp_target_hidden_states = None
        self.assertEqual(output.mtp_assignments, [])
        wide = torch.ones(1, _HC * _HIDDEN)
        output.mtp_target_hidden_states = wide
        self.assertIs(output.mtp_target_hidden_states, wide)

    class _Block(nn.Module):
        layer_type = HybridAttentionType.LINEAR

        def __init__(self, index):
            super().__init__()
            self.index = index
            self.post = None

        def forward(self, value, *args, **kwargs):
            output = value[:, :_HIDDEN] * (0.1 + self.index * 0.03)
            gates = torch.arange(1, _HC + 1, dtype=value.dtype).expand(
                value.shape[0], -1
            )
            self.post = inject_into_residual(value, output, gates)
            return self.post

    def _model(self, layer_ids=(2, 0), dtype=torch.float32, draft=False):
        model = qwen4_exp.Qwen4ExpModel.__new__(qwen4_exp.Qwen4ExpModel)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(
            hidden_size=_HIDDEN,
            hc_mult=_HC,
            layernorm_eps=_EPS,
            hidden_state_capture_layer_ids=list(layer_ids),
            is_mtp=draft,
        )
        model.parallelism_config = SimpleNamespace(role_type="PDFUSION")
        model._disable_capture_context()
        gamma, down, up, _ = (w.to(dtype) for w in _hc_weights(19))
        weights = Qwen4ExpModelConstructionTest._FakeModelWeights(None, gamma, down, up)
        model._build_output_head(model.config, weights)
        model.layers = nn.ModuleList([self._Block(i) for i in range(3)])
        model.ple_layers = nn.ModuleDict()
        model.kv_cache = None
        model._capture_mtp_target_hidden = False
        embedding = torch.linspace(-2, 3, 3 * _HIDDEN).reshape(3, _HIDDEN).to(dtype)
        model.word_embedding = lambda _: embedding
        model._try_fp8_hyper_embedding = lambda _: None
        model._build_attn_meta = lambda *_: None
        inputs = PyModelInputs()
        inputs.input_ids = torch.arange(3, dtype=torch.int32)
        inputs.attention_inputs = PyAttentionInputs()
        inputs.capture_hidden_states = True
        return model, inputs

    @staticmethod
    def _oracle(mixer, value):
        groups = value.float().reshape(-1, _HC, _HIDDEN)
        normalized = groups * torch.rsqrt(groups.square().mean(-1, keepdim=True) + _EPS)
        normalized = (normalized.flatten(-2) * (1 + mixer.norm_gamma.float())).to(
            value.dtype
        )
        down = torch.nn.functional.linear(normalized, mixer.mix_down) / _HC
        gates = torch.sigmoid(
            torch.nn.functional.linear(torch.nn.functional.silu(down), mixer.mix_up)
        )
        return (gates * normalized).reshape(-1, _HC, _HIDDEN).mean(1)

    def _run(self, model, inputs):
        return model._forward_with_ple_prefetch(inputs, object())

    def test_canonical_order_logits_mtp_and_repeat_are_capture_invariant(self):
        for dtype in (torch.float32, torch.bfloat16):
            for mtp in (False, True):
                with self.subTest(dtype=dtype, mtp=mtp):
                    model, inputs = self._model(dtype=dtype)
                    model._capture_mtp_target_hidden = mtp
                    mixer = model.hyper_connection_mixer
                    inputs.capture_hidden_states = False
                    with patch.object(mixer, "forward", wraps=mixer.forward) as readout:
                        off = self._run(model, inputs)
                    self.assertEqual(readout.call_count, 1)
                    with patch.object(mixer, "forward", wraps=mixer.forward) as readout:
                        inputs.capture_hidden_states = True
                        on = self._run(model, inputs)
                    # K auxiliary projections (including selected last layer),
                    # plus exactly one ordinary final projection.
                    self.assertEqual(readout.call_count, 3)
                    self.assertIsNone(on.capture_error)
                    expected = torch.cat(
                        [self._oracle(mixer, model.layers[i].post) for i in (2, 0)]
                        + [self._oracle(mixer, model.layers[-1].post)],
                        dim=-1,
                    )
                    torch.testing.assert_close(
                        on.captured_hidden_states, expected, rtol=0, atol=0
                    )
                    self.assertEqual(on.captured_hidden_states.shape, (3, 3 * _HIDDEN))
                    self.assertEqual(on.captured_hidden_states.dtype, dtype)
                    torch.testing.assert_close(
                        on.hidden_states, off.hidden_states, rtol=0, atol=0
                    )
                    lm_head = (
                        torch.linspace(-1, 1, _HIDDEN * 13)
                        .reshape(_HIDDEN, 13)
                        .to(dtype)
                    )
                    torch.testing.assert_close(
                        on.hidden_states @ lm_head,
                        off.hidden_states @ lm_head,
                        rtol=0,
                        atol=0,
                    )
                    if mtp:
                        self.assertIs(
                            on.mtp_target_hidden_states, model.layers[-1].post
                        )
                        torch.testing.assert_close(
                            on.mtp_target_hidden_states,
                            off.mtp_target_hidden_states,
                            rtol=0,
                            atol=0,
                        )
                        self.assertEqual(
                            on.mtp_target_hidden_states.shape[-1], _HC * _HIDDEN
                        )
                    else:
                        self.assertIsNone(on.mtp_target_hidden_states)
                    raw_mean = model.layers[0].post.reshape(-1, _HC, _HIDDEN).mean(1)
                    self.assertFalse(
                        torch.allclose(
                            on.captured_hidden_states[:, _HIDDEN : 2 * _HIDDEN],
                            raw_mean,
                        )
                    )
                    again = self._run(model, inputs)
                    torch.testing.assert_close(
                        again.captured_hidden_states, expected, rtol=0, atol=0
                    )
                    self.assertNotEqual(
                        again.captured_hidden_states.data_ptr(),
                        on.captured_hidden_states.data_ptr(),
                    )

    def test_tensor_output_contract_preserves_finalize_in_normal_and_warmup(self):
        finalize = CaptureContext.finalize
        for warmup in (False, True):
            for mtp in (False, True):
                for mode in ("off", "capture", "capture_error"):
                    if warmup and mode == "capture_error":
                        # Active warmup already exercises the missing-layer error.
                        continue
                    with self.subTest(warmup=warmup, mtp=mtp, mode=mode):
                        model, inputs = self._model()
                        model._capture_mtp_target_hidden = mtp
                        inputs.capture_hidden_states = mode != "off"
                        embedding = model.word_embedding(inputs)
                        wide = model._initial_hyper_states(embedding)
                        if warmup:
                            model.ple_layers = nn.ModuleDict({"1": nn.Identity()})
                            inputs.attention_inputs.is_prefill = True
                            inputs.attention_inputs.is_s_padded = True
                        finalized = []

                        def record(context, hidden, residual=None):
                            output = finalize(context, hidden, residual)
                            self.assertIs(output.hidden_states, hidden)
                            self.assertEqual(output.mtp_assignments, [])
                            finalized.append(
                                (
                                    output,
                                    hidden,
                                    output.captured_hidden_states,
                                    output.capture_error,
                                )
                            )
                            return output

                        with (
                            patch.object(
                                model, "_initial_hyper_states", return_value=wide
                            ),
                            patch.object(
                                model,
                                "_capture_canonical_layer",
                                wraps=model._capture_canonical_layer,
                                side_effect=(
                                    ValueError("aux failure")
                                    if mode == "capture_error"
                                    else None
                                ),
                            ) as layer_hook,
                            patch.object(
                                CaptureContext,
                                "finalize",
                                autospec=True,
                                side_effect=record,
                            ),
                        ):
                            model._init_capture_context(
                                layer_hook, model._capture_canonical_final
                            )
                            actual = self._run(model, inputs)
                        self.assertEqual(len(finalized), 1)
                        output, hidden, payload, error = finalized[0]
                        self.assertIs(actual, output)
                        self.assertIs(actual.hidden_states, hidden)
                        self.assertIs(actual.captured_hidden_states, payload)
                        self.assertIs(actual.capture_error, error)
                        self.assertEqual(len(actual.mtp_assignments), int(mtp))
                        if mtp:
                            self.assertIs(
                                actual.mtp_target_hidden_states,
                                wide if warmup else model.layers[-1].post,
                            )
                        else:
                            self.assertIsNone(actual.mtp_target_hidden_states)
                        if warmup:
                            self.assertIs(actual.hidden_states, embedding)
                        if mode == "off":
                            self.assertIsNone(payload)
                            self.assertIsNone(error)
                        elif warmup or mode == "capture_error":
                            self.assertIsNone(payload)
                            self.assertIn(
                                (
                                    "did not visit configured layers"
                                    if warmup
                                    else "aux failure"
                                ),
                                error,
                            )
                        else:
                            self.assertIsInstance(payload, torch.Tensor)
                            self.assertIsNone(error)

    def test_deferred_snapshot_preserves_live_aliases_and_context_owners(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model, _ = self._model((0,), dtype)
                base = torch.linspace(-3, 5, 3 * _HC * _HIDDEN).reshape(3, -1).to(dtype)
                pending = (
                    torch.linspace(-1, 1, 3 * _HIDDEN).reshape(3, -1).to(dtype),
                    torch.tensor([[0.1, 0.3, 0.7, 1.2]], dtype=dtype).expand(3, -1),
                )
                originals = [tensor.clone() for tensor in (base, *pending)]
                context = model.capture_context(True)
                context.capture_layer_from(
                    0, lambda: qwen4_exp.inject_into_residual(base, *pending)
                )
                snapshot, residual = context._capture_sources[0]
                self.assertIsNone(residual)
                self.assertNotEqual(snapshot.data_ptr(), base.data_ptr())
                for tensor in pending:
                    self.assertNotEqual(snapshot.data_ptr(), tensor.data_ptr())
                torch.testing.assert_close(
                    snapshot, inject_into_residual(base, *pending), rtol=0, atol=0
                )
                # The normal consumer still materializes the original pending tuple.
                materialized = inject_into_residual(base, *pending)
                final = model.hyper_connection_mixer(materialized)[0]
                outputs = context.finalize(final)
                self.assertIs(outputs.hidden_states, final)
                torch.testing.assert_close(
                    outputs.captured_hidden_states[:, :_HIDDEN],
                    self._oracle(model.hyper_connection_mixer, materialized),
                    rtol=0,
                    atol=0,
                )
                for tensor, original in zip((base, *pending), originals):
                    torch.testing.assert_close(tensor, original, rtol=0, atol=0)
                self.assertNotEqual(
                    outputs.captured_hidden_states.data_ptr(), base.data_ptr()
                )
                base.add_(100)
                torch.testing.assert_close(
                    outputs.captured_hidden_states[:, :_HIDDEN], final, rtol=0, atol=0
                )

    def test_real_deferred_decoder_keeps_next_consumer_and_rounding(self):
        class Attention(nn.Module):
            def forward(self, hidden_states, **kwargs):
                return hidden_states * 0.3

        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model, _ = self._model((0,), dtype)
                layer = qwen4_exp.Qwen4ExpDecoderLayer.__new__(
                    qwen4_exp.Qwen4ExpDecoderLayer
                )
                nn.Module.__init__(layer)
                layer.self_attn = Attention()
                layer.mlp = nn.Tanh()
                for name, seed in (
                    ("attn_hyper_connection", 4),
                    ("mlp_hyper_connection", 5),
                ):
                    setattr(
                        layer,
                        name,
                        qwen4_exp.GatedResidual(
                            *(w.to(dtype) for w in _hc_weights(seed)),
                            hc_mult=_HC,
                            norm_eps=_EPS,
                        ),
                    )
                wide = torch.linspace(-2, 3, 3 * _HC * _HIDDEN).reshape(3, -1).to(dtype)
                base, output, gate = layer.forward_deferred(wide, None)
                pending = (output, gate)
                saved = tuple(t.clone() for t in (base, *pending))
                off_base, off_output, off_gate = layer.forward_deferred(
                    base, None, pending_injection=pending
                )
                expected = inject_into_residual(off_base, off_output, off_gate)
                context = model.capture_context(True)
                context.capture_layer_from(
                    0, lambda: qwen4_exp.inject_into_residual(base, *pending)
                )
                consumer = layer.attn_hyper_connection
                with patch.object(
                    consumer,
                    "forward_with_injection",
                    wraps=consumer.forward_with_injection,
                ) as consume:
                    on_base, on_output, on_gate = layer.forward_deferred(
                        base, None, pending_injection=pending
                    )
                for actual, original in zip(consume.call_args.args, (base, *pending)):
                    self.assertIs(actual, original)
                actual = inject_into_residual(on_base, on_output, on_gate)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    actual, layer(layer(wide, None), None), rtol=0, atol=0
                )
                for tensor, original in zip((base, *pending), saved):
                    torch.testing.assert_close(tensor, original, rtol=0, atol=0)
                final = model.hyper_connection_mixer(actual)[0]
                result = context.finalize(final)
                self.assertIsNone(result.capture_error)
                torch.testing.assert_close(
                    result.captured_hidden_states[:, :_HIDDEN],
                    self._oracle(
                        model.hyper_connection_mixer,
                        inject_into_residual(base, *pending),
                    ),
                    rtol=0,
                    atol=0,
                )

    def test_finalize_object_and_final_identity_are_preserved(self):
        model, inputs = self._model()
        model._capture_mtp_target_hidden = True
        returned = []
        finalize = CaptureContext.finalize

        def record(context, hidden, residual=None):
            output = finalize(context, hidden, residual)
            returned.append(output)
            self.assertIs(output.hidden_states, hidden)
            return output

        with (
            patch.object(CaptureContext, "finalize", autospec=True, side_effect=record),
            patch.object(
                model, "_capture_canonical_final", wraps=model._capture_canonical_final
            ) as final_hook,
        ):
            model._init_capture_context(model._capture_canonical_layer, final_hook)
            actual = self._run(model, inputs)
        final_hook.assert_called_once()
        self.assertEqual(len(returned), 1)
        self.assertIs(actual, returned[0])
        self.assertIs(actual.mtp_target_hidden_states, model.layers[-1].post)

    def test_snapshot_and_projection_errors_fail_open_but_fatal_errors_propagate(self):
        model, _ = self._model((0, 1))
        base = torch.ones(3, _HC * _HIDDEN)
        pending = (torch.ones(3, _HIDDEN), torch.ones(3, _HC))
        for site in ("snapshot", "projection"):
            for error in (
                ValueError("side readout failure"),
                MemoryError("host allocation"),
                torch.OutOfMemoryError("allocator"),
                RuntimeError("CUDA error: illegal address"),
                RuntimeError("NCCL error"),
                KeyboardInterrupt(),
            ):
                with self.subTest(site=site, error=type(error).__name__):
                    context = model.capture_context(True)
                    target = (
                        patch.object(
                            qwen4_exp, "inject_into_residual", side_effect=error
                        )
                        if site == "snapshot"
                        else patch.object(
                            model.hyper_connection_mixer, "forward", side_effect=error
                        )
                    )
                    with target:
                        if isinstance(error, ValueError):
                            context.capture_layer_from(
                                0,
                                lambda: qwen4_exp.inject_into_residual(base, *pending),
                            )
                        else:
                            with self.assertRaises(type(error)) as caught:
                                context.capture_layer_from(
                                    0,
                                    lambda: qwen4_exp.inject_into_residual(
                                        base, *pending
                                    ),
                                )
                            self.assertIs(caught.exception, error)
                            continue
                    self.assertFalse(context.wants_layer(1))
                    final = model.hyper_connection_mixer(
                        inject_into_residual(base, *pending)
                    )[0]
                    output = context.finalize(final)
                    self.assertIs(output.hidden_states, final)
                    self.assertIsNone(output.captured_hidden_states)
                    self.assertIn("side readout failure", output.capture_error)

    def test_side_failure_continues_main_forward_but_model_failure_is_not_swallowed(
        self,
    ):
        model, inputs = self._model()
        inputs.capture_hidden_states = False
        off = self._run(model, inputs)
        inputs.capture_hidden_states = True
        forward = model.hyper_connection_mixer.forward
        calls = []

        def fail_first(value):
            calls.append(value)
            if len(calls) == 1:
                raise ValueError("aux failure")
            return forward(value)

        with patch.object(
            model.hyper_connection_mixer, "forward", side_effect=fail_first
        ):
            on = self._run(model, inputs)
        self.assertEqual(len(calls), 2)
        self.assertIn("aux failure", on.capture_error)
        self.assertIsNone(on.captured_hidden_states)
        torch.testing.assert_close(on.hidden_states, off.hidden_states, rtol=0, atol=0)
        with patch.object(
            model.layers[1], "forward", side_effect=ValueError("decoder failure")
        ):
            with self.assertRaisesRegex(ValueError, "decoder failure"):
                self._run(model, inputs)
        model, inputs = self._model(())
        with patch.object(
            model.hyper_connection_mixer,
            "forward",
            side_effect=ValueError("final failure"),
        ):
            with self.assertRaisesRegex(ValueError, "final failure"):
                self._run(model, inputs)

    def test_invalid_readout_and_pack_failures_do_not_export_partial_payload(self):
        model, _ = self._model((0,))
        base = torch.ones(3, _HC * _HIDDEN)
        invalid = (
            torch.ones(3, _HIDDEN + 1),
            torch.ones(3, _HIDDEN, dtype=torch.float64),
            torch.ones(3, _HIDDEN, device="meta"),
        )
        for value in invalid:
            context = model.capture_context(True)
            with patch.object(
                model.hyper_connection_mixer,
                "forward",
                return_value=(value, base, None),
            ):
                context.capture_layer(0, base)
            output = context.finalize(torch.ones(3, _HIDDEN))
            self.assertIn("canonical readout", output.capture_error)
            self.assertIsNone(output.captured_hidden_states)
        context = model.capture_context(True)
        context.capture_layer(0, base, torch.ones_like(base))
        self.assertIn(
            "materialized Tensor without residual",
            context.finalize(torch.ones(3, _HIDDEN)).capture_error,
        )
        for error in (ValueError("pack failure"), MemoryError("pack allocation")):
            context = model.capture_context(True)
            context.capture_layer(0, base)
            with patch.object(torch, "cat", side_effect=error):
                if isinstance(error, MemoryError):
                    with self.assertRaises(MemoryError):
                        context.finalize(torch.ones(3, _HIDDEN))
                else:
                    output = context.finalize(torch.ones(3, _HIDDEN))
                    self.assertIn("pack failure", output.capture_error)
                    self.assertIsNone(output.captured_hidden_states)

    def test_empty_off_draft_and_warmup_do_not_claim_successful_capture(self):
        for ids, requested, draft in (
            ((), True, False),
            ((2, 0), False, False),
            ((2, 0), True, True),
        ):
            model, inputs = self._model(ids, draft=draft)
            inputs.capture_hidden_states = requested
            self.assertEqual(model.supports_hidden_state_capture, not draft)
            with patch.object(
                model.hyper_connection_mixer,
                "forward",
                wraps=model.hyper_connection_mixer.forward,
            ) as readout:
                outputs = self._run(model, inputs)
            self.assertEqual(readout.call_count, 1)
            self.assertIsNone(outputs.captured_hidden_states)
            self.assertIsNone(outputs.capture_error)
            self.assertIsNone(model._capture_context._capture_sources)
            self.assertIsNone(model._capture_context._captured_hidden_states)
        model, inputs = self._model()
        model.ple_layers = nn.ModuleDict({"1": nn.Identity()})
        inputs.attention_inputs.is_prefill = True
        inputs.attention_inputs.is_s_padded = True
        for requested in (False, True):
            inputs.capture_hidden_states = requested
            with patch.object(
                model.hyper_connection_mixer,
                "forward",
                side_effect=AssertionError("warmup mixer"),
            ):
                outputs = self._run(model, inputs)
            self.assertIsNone(outputs.captured_hidden_states)
            if requested:
                self.assertIn("did not visit configured layers", outputs.capture_error)
            else:
                self.assertIsNone(outputs.capture_error)


class Qwen4ExpAttentionTagRoutingTest(TestCase):
    """The PLE layer owns three cache regions, so attention must be tag-routed.

    Tags come from the real desc builder so this fails if either side renames a
    region.
    """

    def _descs(self):
        return build_qwen4_exp_kv_cache_spec_descs(
            _spec_config(), ple_layer_indices=[1], ple_conv_kernel_size=4, ngram_size=3
        )

    def test_ple_layer_resolves_its_single_attention_region(self):
        descs = self._descs()
        model = _bare_model(descs)

        self.assertEqual(len(descs[1]), 3)
        self.assertEqual(model._attention_tag(1), descs[1][0].tag)
        self.assertNotIn(model._attention_tag(1), (PLE_STATE_TAG, PLE_NGRAM_CTX_TAG))

    def test_first_linear_layer_metadata_excludes_ple_regions(self):
        descs = build_qwen4_exp_kv_cache_spec_descs(
            _spec_config(), ple_layer_indices=[0], ple_conv_kernel_size=4, ngram_size=3
        )
        model = _bare_model(descs, _hybrid_types())
        model.parallelism_config = ParallelismConfig()
        linear = PyAttentionInputs()
        linear.is_prefill = False
        linear.is_target_verify = True
        side = PyAttentionInputs()
        side.is_prefill = False
        side.is_target_verify = False
        inputs = PyModelInputs()
        # Side regions deliberately come first so primary-map order cannot
        # accidentally select the right metadata.
        inputs.attention_inputs = {
            PLE_STATE_TAG: side,
            PLE_NGRAM_CTX_TAG: side,
            descs[0][0].tag: linear,
        }
        for graph in (False, True):
            meta = model._build_attn_meta(inputs, torch.device("cpu"), graph)
            self.assertTrue(meta.is_target_verify)
            self.assertEqual(meta.is_cuda_graph, graph)

    def test_plain_layers_resolve_their_own_region(self):
        descs = self._descs()
        model = _bare_model(descs)

        for layer_idx in range(len(descs)):
            if layer_idx == 1:
                continue
            self.assertEqual(model._attention_tag(layer_idx), descs[layer_idx][0].tag)

    def test_rejects_a_layer_with_two_attention_regions(self):
        descs = self._descs()
        descs[0] = descs[0] + [descs[3][0]]
        model = _bare_model(descs)

        with self.assertRaisesRegex(RuntimeError, "exactly one attention region"):
            model._attention_tag(0)


class Qwen4ExpFmhaRoutingTest(TestCase):
    """Indexer regions must not leak into FMHA routing.

    With the indexer enabled a full-attention layer owns three regions. Without
    a filter ``_get_fmha_group_tags`` would ask the factory to build impls for
    the indexer pools and ``_layer_fmha_impl`` would hand ``self_attn`` a list
    of every region's impl instead of the attention one.
    """

    def _descs(self):
        return build_qwen4_exp_kv_cache_spec_descs(
            _spec_config(),
            ple_layer_indices=[1],
            ple_conv_kernel_size=4,
            ngram_size=3,
            indexer_head_dim=128,
            indexer_compress_ratio=4,
        )

    def _full_layer(self):
        return SimpleNamespace(layer_type=HybridAttentionType.NONE)

    def test_fmha_impls_build_only_for_the_attention_region(self):
        model = _bare_model(self._descs(), _hybrid_types())

        self.assertEqual(model._get_fmha_group_tags(), ["full"])

    def test_full_attention_layer_resolves_its_single_impl(self):
        model = _bare_model(self._descs(), _hybrid_types())
        impls = {
            "full": object(),
            INDEXER_KV_TAG: object(),
            INDEXER_STATE_TAG: object(),
        }

        chosen = model._layer_fmha_impl(self._full_layer(), impls, 3)

        self.assertIs(chosen, impls["full"])

    def test_linear_layer_gets_no_impl(self):
        model = _bare_model(self._descs(), _hybrid_types())

        chosen = model._layer_fmha_impl(
            SimpleNamespace(layer_type=HybridAttentionType.LINEAR),
            {"full": object()},
            0,
        )

        self.assertIsNone(chosen)

    def test_an_untagged_impl_passes_through(self):
        model = _bare_model(self._descs(), _hybrid_types())
        impl = object()

        self.assertIs(model._layer_fmha_impl(self._full_layer(), impl, 3), impl)

    def test_rejects_a_mapping_without_the_attention_tag(self):
        model = _bare_model(self._descs(), _hybrid_types())

        with self.assertRaisesRegex(RuntimeError, "FMHA impl for tag"):
            model._layer_fmha_impl(self._full_layer(), {INDEXER_KV_TAG: object()}, 3)

    def test_qsa_runtime_context_resolves_all_three_tag_local_regions(self):
        descs = self._descs()
        model = _bare_model(descs)
        model.config = SimpleNamespace(is_mtp=True)
        main_cache = model.kv_cache.get_layer_cache(3, "full")
        main_inputs = object()
        kv_inputs = object()
        state_inputs = object()
        attention_inputs = {
            "full": main_inputs,
            INDEXER_KV_TAG: kv_inputs,
            INDEXER_STATE_TAG: state_inputs,
            # Target and draft share a global input map; the draft layer does
            # not own the target's coordinate state in the cache topology.
            INDEXER_ROPE_POS_TAG: object(),
        }

        context = model._qsa_runtime_context(
            3, main_cache, main_inputs, attention_inputs
        )

        self.assertIs(context.main_cache, main_cache)
        self.assertIs(context.main_inputs, main_inputs)
        self.assertEqual(context.indexer_kv_cache.tag, INDEXER_KV_TAG)
        self.assertIs(context.indexer_kv_inputs, kv_inputs)
        self.assertEqual(context.indexer_state_cache.tag, INDEXER_STATE_TAG)
        self.assertIs(context.indexer_state_inputs, state_inputs)
        self.assertTrue(context.is_mtp_draft)
        self.assertIsNone(context.position_state_cache)
        self.assertIsNone(context.position_state_inputs)


class Qwen4ExpSharedHCStorageTest(TestCase):
    def test_real_residual_builder_reuses_owner_and_blocks_updates(self):
        import weakref

        from rtp_llm.model_loader.model_weight_info import ModelWeights
        from rtp_llm.model_loader.weight_manager import WeightManager

        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        config = ModelConfig()
        config.hc_mult = 4
        config.layernorm_eps = 1e-6
        weights = ModelWeights(1, "cuda", torch.bfloat16)
        for norm, down, up, inject in (
            (
                W.qwen4_hc_attn_norm,
                W.qwen4_hc_attn_mix_down,
                W.qwen4_hc_attn_mix_up,
                W.qwen4_hc_attn_inject,
            ),
            (
                W.qwen4_hc_mlp_norm,
                W.qwen4_hc_mlp_mix_down,
                W.qwen4_hc_mlp_mix_up,
                W.qwen4_hc_mlp_inject,
            ),
        ):
            for key, shape in (
                (norm, (32,)),
                (down, (7, 32)),
                (up, (32, 7)),
                (inject, (4, 32)),
            ):
                weights.set_layer_weight(
                    0, key, torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                )
        old_refs = [
            weakref.ref(weights.weights[0][key])
            for key in (
                W.qwen4_hc_attn_mix_down,
                W.qwen4_hc_attn_inject,
                W.qwen4_hc_mlp_mix_down,
                W.qwen4_hc_mlp_inject,
            )
        ]
        layer = qwen4_exp.Qwen4ExpDecoderLayer.__new__(qwen4_exp.Qwen4ExpDecoderLayer)
        nn.Module.__init__(layer)
        layer._build_residual_modules(config, weights.weights[0])
        self.assertTrue(all(ref() is None for ref in old_refs))
        for unit, down, inject in (
            (
                layer.attn_hyper_connection,
                W.qwen4_hc_attn_mix_down,
                W.qwen4_hc_attn_inject,
            ),
            (
                layer.mlp_hyper_connection,
                W.qwen4_hc_mlp_mix_down,
                W.qwen4_hc_mlp_inject,
            ),
        ):
            self.assertEqual(
                weights.weights[0][down].untyped_storage().data_ptr(),
                unit.merged_down_inject.untyped_storage().data_ptr(),
            )
            self.assertIs(weights.weights[0][inject], unit.inject)
            before = unit.merged_down_inject.clone()
            with self.assertRaisesRegex(PermissionError, "restart"):
                weights.update_layer_weight(0, down, torch.zeros_like(unit.mix_down))
            torch.testing.assert_close(unit.merged_down_inject, before, rtol=0, atol=0)
        manager = WeightManager.__new__(WeightManager)
        manager._weights = weights
        # Invalid IPC data would fail decoding if the shared-view guard were late.
        with self.assertRaisesRegex(PermissionError, "restart"):
            manager.update(
                dict(
                    name="model.layers.0." + W.qwen4_hc_attn_inject,
                    method="shm",
                    desc="invalid-ipc-description",
                )
            )

    def test_unsupported_cpu_keeps_original_storage(self):
        fixture = Qwen4ExpDecoderLayerTest()
        fixture.setUp()
        layer = qwen4_exp.Qwen4ExpDecoderLayer.__new__(qwen4_exp.Qwen4ExpDecoderLayer)
        nn.Module.__init__(layer)
        original = fixture.weights[W.qwen4_hc_attn_mix_down]
        layer._build_residual_modules(fixture.config, fixture.weights)
        self.assertIs(fixture.weights[W.qwen4_hc_attn_mix_down], original)
        self.assertFalse(getattr(original, "_rtp_requires_restart_for_update", False))


if __name__ == "__main__":
    main()
