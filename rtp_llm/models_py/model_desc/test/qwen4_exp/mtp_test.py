import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpModel
from rtp_llm.models_py.model_desc.qwen4_exp_mtp import (
    Qwen4ExpMTPInputProjection,
    Qwen4ExpMTPModel,
    normalize_mtp_decode_positions,
    normalize_mtp_prefill_positions,
)
from rtp_llm.models_py.model_desc.test.qwen4_exp.fixtures import (
    _EPS,
    _HC,
    _HIDDEN,
    _hc_weights,
)
from rtp_llm.models_py.modules.qwen4_exp.gated_residual import Qwen4GatedResidual
from rtp_llm.ops import HybridAttentionType
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyCacheStoreInputs, PyModelInputs


class _Matmul(nn.Module):
    def __init__(self, runtime_weight: torch.Tensor):
        super().__init__()
        self.runtime_weight = runtime_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.runtime_weight


def _raw_gamma_rms_norm(
    value: torch.Tensor, gamma: torch.Tensor, eps: float
) -> torch.Tensor:
    normalized = value.float() * torch.rsqrt(
        value.float().pow(2).mean(-1, keepdim=True) + eps
    )
    return (normalized * (1.0 + gamma.float())).type_as(value)


class Qwen4ExpMTPInputProjectionTest(unittest.TestCase):
    def test_prepare_normalizes_unpublished_cold_and_prefix_hit_positions(self):
        # Match updatePrefillPostDraftModelInput: shift each prompt's position
        # rows left, then retain its native tail anchor. Include a cold row
        # alongside prefix hits so both cache writers use the same RoPE origin.
        for factor in (1, 3):
            for suffix in (1, 3, 4, 8):
                with self.subTest(factor=factor, suffix=suffix):
                    prefixes = [0, 128, 256]
                    lengths = [3, suffix, 1]
                    positions = [
                        torch.arange(prefix, prefix + length, dtype=torch.int32)
                        for prefix, length in zip(prefixes, lengths)
                    ]
                    expected = torch.cat(positions).repeat_interleave(factor)
                    shifted = torch.cat(
                        [torch.cat((row[1:], row[-1:])) for row in positions]
                    ).repeat_interleave(factor)
                    groups = {}
                    for tag in ("full", "indexer_kv", "indexer_state"):
                        groups[tag] = SimpleNamespace(
                            is_prefill=True,
                            is_target_verify=False,
                            is_cuda_graph=False,
                            cache_store_inputs=None,
                            prefix_lengths_device=torch.tensor(
                                prefixes, dtype=torch.int32
                            ),
                            cu_seqlens_device=torch.tensor(
                                [0, 3, 3 + suffix, 4 + suffix], dtype=torch.int32
                            ),
                            combo_position_ids=shifted.clone(),
                        )
                    inputs = SimpleNamespace(
                        input_ids=torch.zeros(sum(lengths), dtype=torch.int32),
                        combo_position_ids=shifted.clone(),
                        attention_inputs=groups,
                    )
                    model = Qwen4ExpMTPModel.__new__(Qwen4ExpMTPModel)
                    nn.Module.__init__(model)
                    model.config = SimpleNamespace(
                        attn_config=SimpleNamespace(
                            rope_config=SimpleNamespace(index_factor=factor)
                        )
                    )
                    with patch.object(
                        Qwen4ExpModel, "prepare_fmha_impl", return_value={}
                    ):
                        model.prepare_fmha_impl(inputs)
                        torch.testing.assert_close(inputs.combo_position_ids, expected)
                        # Accepted-token prefills already use canonical rows.
                        model.prepare_fmha_impl(inputs)
                    torch.testing.assert_close(inputs.combo_position_ids, expected)
                    for value in inputs.attention_inputs.values():
                        torch.testing.assert_close(value.combo_position_ids, expected)

    def test_pd_main_and_indexer_share_normalized_cache_positions(self):
        writer = SimpleNamespace(write=lambda *_: None)
        plan = PyCacheStoreInputs()
        shifted = torch.tensor(
            [129, 130, 130, 256], dtype=torch.int32
        ).repeat_interleave(3)
        groups = {
            tag: SimpleNamespace(
                cache_store_inputs=plan,
                cache_store_writer=writer,
                is_prefill=True,
                is_target_verify=False,
                is_cuda_graph=False,
                prefix_lengths_device=torch.tensor([128, 256], dtype=torch.int32),
                cu_seqlens_device=torch.tensor([0, 3, 4], dtype=torch.int32),
                combo_position_ids=shifted if tag != "full" else shifted.clone(),
            )
            for tag in ("full", "indexer_kv", "indexer_state")
        }
        inputs = SimpleNamespace(
            input_ids=torch.zeros(4, dtype=torch.int32),
            combo_position_ids=shifted.clone(),
            attention_inputs=groups,
        )
        pointers = {
            tag: value.combo_position_ids.data_ptr() for tag, value in groups.items()
        }
        normalize_mtp_prefill_positions(inputs, 3)
        expected = torch.tensor(
            [128, 129, 130, 256], dtype=torch.int32
        ).repeat_interleave(3)
        torch.testing.assert_close(inputs.combo_position_ids, expected)
        for tag, value in groups.items():
            torch.testing.assert_close(value.combo_position_ids, expected)
            self.assertEqual(value.combo_position_ids.data_ptr(), pointers[tag])

    def test_decode_verify_and_graph_positions_are_preserved(self):
        for prefill, verify, graph in (
            (False, False, False),
            (True, True, False),
            (True, False, True),
        ):
            with self.subTest(prefill=prefill, verify=verify, graph=graph):
                shifted = torch.tensor([129, 130, 130], dtype=torch.int32)
                inputs = SimpleNamespace(
                    input_ids=torch.zeros(3, dtype=torch.int32),
                    combo_position_ids=shifted,
                    attention_inputs={
                        "full": SimpleNamespace(
                            cache_store_inputs=None,
                            is_prefill=prefill,
                            is_target_verify=verify,
                            is_cuda_graph=graph,
                        )
                    },
                )
                normalize_mtp_prefill_positions(inputs, 1)
                torch.testing.assert_close(
                    shifted, torch.tensor([129, 130, 130], dtype=torch.int32)
                )

    def test_decode_removes_target_media_offset_without_scalar_reads(self):
        for graph in (False, True):
            with self.subTest(graph=graph):
                groups = {
                    tag: SimpleNamespace(
                        is_prefill=False,
                        is_target_verify=False,
                        is_cuda_graph=graph,
                        sequence_lengths=torch.tensor([137, 260], dtype=torch.int32),
                        sequence_lengths_plus_1_device=torch.tensor(
                            [138, 261], dtype=torch.int32
                        ),
                        combo_position_ids=torch.tensor(
                            [40] * 3 + [70] * 3, dtype=torch.int32
                        ),
                    )
                    for tag in ("full", "indexer_kv", "indexer_state")
                }
                inputs = SimpleNamespace(
                    input_ids=torch.tensor([7, 8], dtype=torch.int32),
                    attention_inputs=groups,
                    combo_position_ids=groups["full"].combo_position_ids.clone(),
                )
                with patch.object(
                    torch.Tensor, "item", side_effect=AssertionError("decode readback")
                ):
                    normalize_mtp_decode_positions(inputs, 3)
                expected = torch.tensor([137] * 3 + [260] * 3, dtype=torch.int32)
                torch.testing.assert_close(inputs.combo_position_ids, expected)
                for value in groups.values():
                    torch.testing.assert_close(value.combo_position_ids, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_decode_graph_refreshes_logical_positions(self):
        lengths = torch.tensor([138, 261], dtype=torch.int32, device="cuda")
        positions = torch.zeros(6, dtype=torch.int32, device="cuda")
        anchor = SimpleNamespace(
            is_prefill=False,
            is_target_verify=False,
            is_cuda_graph=True,
            sequence_lengths_plus_1_device=lengths,
            combo_position_ids=positions,
        )
        inputs = SimpleNamespace(
            input_ids=torch.zeros(2, dtype=torch.int32, device="cuda"),
            combo_position_ids=positions,
            attention_inputs={"full": anchor},
        )
        normalize_mtp_decode_positions(inputs, 3)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            normalize_mtp_decode_positions(inputs, 3)
        for actual in ([152, 290], [200, 330]):
            lengths.copy_(torch.tensor(actual, dtype=torch.int32, device="cuda"))
            graph.replay()
            torch.testing.assert_close(positions, (lengths - 1).repeat_interleave(3))

    def test_draft_graph_does_not_prepare_unused_gdn_metadata(self):
        model = SimpleNamespace(
            layers=[SimpleNamespace(layer_type=HybridAttentionType.NONE)]
        )
        inputs = SimpleNamespace()
        with patch(
            "rtp_llm.models_py.model_desc.qwen4_exp_mtp.get_attention_inputs_value",
            return_value={"full": SimpleNamespace(is_prefill=True)},
        ):
            meta = Qwen4ExpMTPModel._build_attn_meta(
                model, inputs, torch.device("cuda"), True
            )
        self.assertTrue(meta.is_cuda_graph)
        self.assertIsNone(meta.prefill_conv1d_meta)

    def test_graph_captures_explicit_base_rope_positions(self):
        draft = SimpleNamespace(
            config=SimpleNamespace(
                attn_config=SimpleNamespace(rope_config=SimpleNamespace(index_factor=3))
            )
        )
        self.assertEqual(Qwen4ExpMTPModel.cuda_graph_position_id_len_factor(draft), 3)

    def test_target_forward_exports_precollapse_rows_to_graph_output(self):
        class Block(nn.Module):
            layer_type = HybridAttentionType.LINEAR

            def forward(self, hidden, *args, **kwargs):
                return hidden

        inputs = PyModelInputs()
        inputs.attention_inputs = PyAttentionInputs()
        inputs.input_ids = torch.arange(2, dtype=torch.int32)
        embedding = torch.arange(2 * _HIDDEN, dtype=torch.float32).reshape(2, _HIDDEN)
        wide = torch.cat([embedding + 10 * i for i in range(_HC)], dim=-1)
        gamma, down, up, _ = _hc_weights(3)
        mixer = Qwen4GatedResidual(gamma, down, up, None, hc_mult=_HC, norm_eps=_EPS)
        fields = dict(
            config=SimpleNamespace(
                hidden_size=_HIDDEN, hidden_state_capture_layer_ids=[0]
            ),
            ple_layers={},
            layers=nn.ModuleList([Block()]),
            kv_cache=None,
            _ple_target_transaction=None,
            word_embedding=lambda _: embedding,
            _try_fp8_hyper_embedding=lambda _: None,
            _build_attn_meta=lambda *_: None,
            _initial_hyper_states=lambda _: wide,
            hyper_connection_mixer=mixer,
        )
        model = Qwen4ExpModel.__new__(Qwen4ExpModel)
        nn.Module.__init__(model)
        for name, value in fields.items():
            setattr(model, name, value)
        model._init_capture_context(
            model._capture_canonical_layer, model._capture_canonical_final
        )
        for mtp in (False, True):
            model._capture_mtp_target_hidden = mtp
            inputs.capture_hidden_states = False
            off = Qwen4ExpModel.forward(model, inputs, object())
            inputs.capture_hidden_states = True
            with patch.object(mixer, "forward", wraps=mixer.forward) as readout:
                on = Qwen4ExpModel.forward(model, inputs, object())
            self.assertEqual(readout.call_count, 2)
            self.assertIsNone(on.capture_error)
            self.assertIsNone(off.capture_error)
            self.assertIsNone(off.captured_hidden_states)
            torch.testing.assert_close(
                on.hidden_states, off.hidden_states, rtol=0, atol=0
            )
            torch.testing.assert_close(
                on.captured_hidden_states,
                torch.cat([off.hidden_states] * 2, dim=-1),
                rtol=0,
                atol=0,
            )
            if mtp:
                self.assertIs(on.mtp_target_hidden_states, wide)
                self.assertIs(off.mtp_target_hidden_states, wide)
                self.assertIs(model._mtp_target_hidden_states, wide)
            else:
                self.assertIsNone(on.mtp_target_hidden_states)
                self.assertIsNone(off.mtp_target_hidden_states)
                self.assertIsNone(model._mtp_target_hidden_states)

    def test_two_projection_forward_matches_small_shape_reference(self):
        torch.manual_seed(11)
        tokens, hc_mult, hidden_size = 3, 2, 4
        eps = 1e-6
        embedding = torch.randn(tokens, hidden_size, dtype=torch.float32)
        target_hidden = torch.randn(tokens, hc_mult * hidden_size, dtype=torch.float32)
        embedding_gamma = torch.tensor(
            [0.003, -0.007, 0.011, -0.013], dtype=torch.bfloat16
        )
        hidden_gamma = torch.tensor(
            [-0.005, 0.009, -0.015, 0.017, 0.019, -0.021, 0.023, -0.025],
            dtype=torch.bfloat16,
        )
        # Runtime linear weights have already transposed checkpoint [out, in]
        # tensors and are therefore [in, out].
        embedding_fc = torch.randn(hidden_size, hidden_size)
        hidden_fc = torch.randn(hidden_size, hidden_size)
        component = Qwen4ExpMTPInputProjection(
            embedding_gamma,
            hidden_gamma,
            _Matmul(embedding_fc),
            _Matmul(hidden_fc),
            hidden_size=hidden_size,
            norm_eps=eps,
        )

        actual = component(embedding, target_hidden)
        expected_embedding = (
            _raw_gamma_rms_norm(embedding, embedding_gamma, eps) @ embedding_fc
        ).unsqueeze(1)
        expected_hidden = (
            _raw_gamma_rms_norm(
                target_hidden,
                hidden_gamma,
                eps,
            ).reshape(-1, hidden_size)
            @ hidden_fc
        ).reshape(tokens, hc_mult, hidden_size)
        expected = (expected_embedding + expected_hidden).flatten(-2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        # Pooling the branches before fc_hidden is a tempting shape workaround,
        # but it erases the released head's hyper-connection semantics.
        pooled_hidden = (
            _raw_gamma_rms_norm(
                target_hidden,
                hidden_gamma,
                eps,
            )
            .reshape(tokens, hc_mult, hidden_size)
            .mean(1)
            @ hidden_fc
        )
        wrongly_pooled = (
            (expected_embedding + pooled_hidden.unsqueeze(1))
            .expand(-1, hc_mult, -1)
            .flatten(-2)
        )
        self.assertGreater(float((actual - wrongly_pooled).abs().max()), 1e-5)

        # Folding +1 into BF16 is observably different from upstream's fp32
        # addition and would make this parity anchor fail.
        folded_embedding = embedding_gamma + torch.ones_like(embedding_gamma)
        folded_hidden = hidden_gamma + torch.ones_like(hidden_gamma)
        normalized_embedding = embedding * torch.rsqrt(
            embedding.pow(2).mean(-1, keepdim=True) + eps
        )
        hidden_groups = target_hidden.reshape(tokens, hc_mult, hidden_size)
        normalized_hidden = (
            target_hidden
            * torch.rsqrt(target_hidden.pow(2).mean(-1, keepdim=True) + eps)
        ).reshape_as(hidden_groups)
        folded_embedding_out = (
            normalized_embedding * folded_embedding.float()
        ) @ embedding_fc
        folded_hidden_out = (
            (
                normalized_hidden * folded_hidden.reshape(hc_mult, hidden_size).float()
            ).reshape(-1, hidden_size)
            @ hidden_fc
        ).reshape(tokens, hc_mult, hidden_size)
        folded = (folded_embedding_out.unsqueeze(1) + folded_hidden_out).flatten(-2)
        self.assertGreater(float((actual - folded).abs().max()), 1e-5)

    def test_mtp_hidden_norm_spans_the_hyperconnection_stream(self):
        """Unequal branch RMS distinguishes MTP norm from main HC norms."""
        hidden_size, branches, eps = 4, 4, 1e-6
        hidden = torch.arange(1, branches + 1, dtype=torch.float32)
        hidden = hidden.repeat_interleave(hidden_size).unsqueeze(0)
        embedding = torch.zeros(1, hidden_size)
        args = (
            torch.zeros(hidden_size),
            torch.zeros(branches * hidden_size),
            _Matmul(torch.eye(hidden_size)),
            _Matmul(torch.eye(hidden_size)),
        )
        actual = Qwen4ExpMTPInputProjection(
            *args, hidden_size=hidden_size, norm_eps=eps
        )(embedding, hidden)
        expected = hidden / torch.sqrt(hidden.square().mean(-1, keepdim=True) + eps)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        branch_local = hidden / torch.sqrt(
            hidden.unflatten(-1, (branches, hidden_size))
            .square()
            .mean(-1, keepdim=True)
            .expand(-1, -1, hidden_size)
            .flatten(-2)
            + eps
        )
        self.assertGreater(float((actual - branch_local).abs().max()), 0.4)

    def test_rejects_non_packed_or_mismatched_target_hidden_input(self):
        component = Qwen4ExpMTPInputProjection(
            torch.zeros(4),
            torch.zeros(8),
            nn.Identity(),
            nn.Identity(),
            hidden_size=4,
            norm_eps=1e-6,
        )
        with self.assertRaisesRegex(RuntimeError, "packed 2-D"):
            component(torch.zeros(1, 1, 4), torch.zeros(1, 1, 8))
        with self.assertRaisesRegex(RuntimeError, "row counts differ"):
            component(torch.zeros(2, 4), torch.zeros(1, 8))
        with self.assertRaisesRegex(RuntimeError, "embedding width mismatch"):
            component(torch.zeros(2, 3), torch.zeros(2, 8))
        with self.assertRaisesRegex(RuntimeError, "target hidden width mismatch"):
            component(torch.zeros(2, 4), torch.zeros(2, 4))

    def test_target_tiles_once_and_mtp_projection_is_not_retiled(self):
        target = SimpleNamespace(
            config=SimpleNamespace(hidden_size=4),
            hc_mult=2,
        )
        embedding = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        tiled = Qwen4ExpModel._initial_hyper_states(target, embedding)
        torch.testing.assert_close(tiled, embedding.repeat(1, 2))

        draft = SimpleNamespace(config=SimpleNamespace(hidden_size=4, hc_mult=2))
        projected = torch.arange(16, dtype=torch.float32).reshape(2, 8)
        actual = Qwen4ExpMTPModel._initial_hyper_states(draft, projected)
        self.assertEqual(actual.data_ptr(), projected.data_ptr())

        with self.assertRaisesRegex(RuntimeError, "projected residual width mismatch"):
            Qwen4ExpMTPModel._initial_hyper_states(draft, embedding)

    def test_target_hidden_accessor_exposes_precollapse_rows(self):
        wide = torch.arange(24, dtype=torch.float32).reshape(3, 8)
        model = SimpleNamespace(_mtp_target_hidden_states=wide)

        actual = Qwen4ExpModel.get_mtp_target_hidden_states(model, 2)
        torch.testing.assert_close(actual, wide[:2])
        torch.testing.assert_close(
            Qwen4ExpModel.get_mtp_target_hidden_states(model, -1), wide
        )
        with self.assertRaisesRegex(RuntimeError, "exceed the last forward"):
            Qwen4ExpModel.get_mtp_target_hidden_states(model, 4)


if __name__ == "__main__":
    unittest.main()
