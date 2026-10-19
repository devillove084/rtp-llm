"""CPU orchestration tests; mocked kernels do not validate CUDA numerics.

Load the owning sources, not copies of their methods. Only external GPU/runtime
imports are stubbed so these tests need neither model weights nor built extensions.
"""

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

_ROOT = Path(__file__).resolve().parents[6]
_BASE = "rtp_llm.models_py.modules.base.cuda"
_HYBRID = "rtp_llm.models_py.modules.hybrid"


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load owning source: {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _cpu_topk(logits, starts, ends, output, *args):
    output.fill_(-1)
    for row, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
        count = min(end - start, output.shape[1])
        if count:
            output[row, :count] = torch.topk(logits[row, start:end], count).indices
    # The real op writes output and returns None.


@contextmanager
def _owning_sources():
    stubs = {}

    def module(name, **attrs):
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            parent = ".".join(parts[:i])
            if parent not in stubs:
                value = types.ModuleType(parent)
                value.__path__ = [str(_ROOT.joinpath(*parts[:i]))]
                stubs[parent] = value
        stubs[name].__dict__.update(attrs)
        return stubs[name]

    ops = SimpleNamespace(
        topk_v3=Mock(),
        dsv4_persistent_topk=Mock(),
        topk_v3_tie_break=Mock(side_effect=_cpu_topk),
        dsv4_top_k_per_row_prefill=Mock(side_effect=_cpu_topk),
        cp_gather_indexer_k_quant_cache=Mock(),
    )
    compute = module(
        "rtp_llm.ops.compute_ops",
        KVCache=object,
        LayerKVCache=object,
        rtp_llm_ops=ops,
        cuda_graph_warmup_forward_enabled=Mock(return_value=False),
        cuda_graph_capture_forward_enabled=Mock(return_value=False),
    )
    module(
        "rtp_llm.ops",
        AttentionConfigs=object,
        HWKernelConfig=object,
        ParallelismConfig=object,
    )
    module(
        "rtp_llm.models_py.distributed.collective_torch",
        Group=object,
        barrier=Mock(),
        all_reduce=Mock(),
    )
    module("rtp_llm.models_py.distributed.pynccl_cp", all_gather=Mock())
    module(
        "rtp_llm.models_py.kernels.cuda.fp8_kernel",
        sgl_per_token_group_quant_fp8=Mock(),
    )
    deep = module("deep_gemm", fp8_mqa_logits=Mock(), fp8_paged_mqa_logits=Mock())
    module("flashinfer.rope")
    # This external Triton optimization is inapplicable to CPU tensors. Keep the
    # actual _write_prefill_topk Torch fallback under test.
    module(_BASE + ".indexer_launch_fusion", try_write_topk=Mock(return_value=False))
    module("rtp_llm.models_py.modules.dsv4.chunk_env")
    module(
        "rtp_llm.device.device_type",
        DeviceType=SimpleNamespace(Cuda="cuda"),
        get_device_type=lambda: "cpu",
    )
    factory = module(
        "rtp_llm.models_py.modules.factory",
        LinearFactory=SimpleNamespace(
            create_linear_from_weights=Mock(
                side_effect=lambda *a, **kw: torch.nn.Identity()
            )
        ),
    )
    module(
        "rtp_llm.models_py.modules.factory.attention.attn_factory", MlaImplBase=object
    )
    module("rtp_llm.models_py.utils.fuse_config", fuse_kernels_enabled=lambda _: False)
    names = (
        "mla_indexer_qb_w",
        "mla_indexer_qb_s",
        "mla_indexer_k_w",
        "mla_indexer_k_s",
        "mla_indexer_k_norm_w",
        "mla_indexer_k_norm_b",
        "mla_indexer_weights_proj_w",
        "rope_cos_sin_cache",
    )
    module("rtp_llm.utils.model_weight", W=SimpleNamespace(**{n: n for n in names}))
    module(_HYBRID)
    test_env = {
        key: os.environ[key] for key in ("TMPDIR", "TEST_TMPDIR") if key in os.environ
    }
    with patch.dict(sys.modules, stubs), patch.dict(os.environ, test_env, clear=True):
        _load(
            "rtp_llm.models_py.modules.dsv4.chunk_env",
            "rtp_llm/models_py/modules/dsv4/chunk_env.py",
        )
        base = _load(
            _BASE + ".indexer_op", "rtp_llm/models_py/modules/base/cuda/indexer_op.py"
        )
        stubs["rtp_llm.models_py.modules"].IndexerOp = base.IndexerOp
        stubs["rtp_llm.models_py.modules"].LayerNorm = lambda *a, **kw: (
            torch.nn.Identity()
        )
        stubs["rtp_llm.models_py.modules"].RMSNorm = torch.nn.Identity
        indexer = _load(
            _HYBRID + ".indexer", "rtp_llm/models_py/modules/hybrid/indexer.py"
        )
        attention = _load(
            _HYBRID + ".mla_attention",
            "rtp_llm/models_py/modules/hybrid/mla_attention.py",
        )
        yield SimpleNamespace(
            base=base,
            indexer=indexer,
            attention=attention,
            ops=ops,
            deep=deep,
            compute=compute,
            factory=factory,
        )


class _Token:
    def __init__(self):
        self.output = None
        self.cancel = Mock()
        self.finish = Mock(side_effect=self._finish)

    def _finish(self, output):
        self.output = output.clone()


class _Context:
    def __init__(self):
        self.calls = []
        self.tokens = []
        self.note_skip = Mock()
        self.begin_topk = Mock(side_effect=self._begin)

    def _begin(self, **kwargs):
        self.calls.append(
            {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in kwargs.items()
            }
        )
        token = _Token()
        self.tokens.append(token)
        return token


class IndexerTensorCaptureTest(unittest.TestCase):
    def setUp(self):
        self.sources = _owning_sources()
        self.s = self.sources.__enter__()
        self.addCleanup(self.sources.__exit__, None, None, None)
        self.op = self.s.base.IndexerOp(2, 128, 3, 64, layer_idx=7)
        self.context = _Context()
        self.logits = torch.arange(32, dtype=torch.float32).view(2, 16)[:, ::2]
        self.starts = torch.tensor([1, 3], dtype=torch.int32)
        self.ends = torch.tensor([5, 5], dtype=torch.int32)
        self.output = torch.empty((2, 3), dtype=torch.int32)

    def run_topk(self, context=None, max_seq_len=100):
        return self.op._run_prefill_topk(
            self.logits,
            self.starts,
            self.ends,
            self.output,
            max_seq_len,
            context=context,
            query_row_start=2,
            total_query_rows=5,
        )

    def ragged(self, context, rows=5, **input_flags):
        # Two requests: query lengths [3, 2], prefixes [2, 1]. Chunk [2, 4)
        # crosses their boundary. KV length is 8, not query length 5.
        q = torch.zeros((rows, 2, 128))
        q[:, 0, 0] = torch.arange(rows)
        scores = torch.arange(80, dtype=torch.float32).view(5, 16)[:, ::2]
        self.s.deep.fp8_mqa_logits.side_effect = lambda q, *args, **kw: scores[
            int(q[0, 0, 0]) : int(q[-1, 0, 0]) + 1
        ]
        params = SimpleNamespace(
            prefill_total_kv_tokens=8,
            prefill_max_kv_len=5,
            ks=torch.tensor([0, 0, 0, 5, 5], dtype=torch.int32)[:rows],
            ke=torch.tensor([3, 4, 5, 7, 8], dtype=torch.int32)[:rows],
            topk_indices_offset=torch.tensor([0, 0, 0, 5, 5], dtype=torch.int32)[:rows],
        )
        inputs = SimpleNamespace(
            tensor_capture_context=context,
            is_prefill=True,
            kv_cache_kernel_block_id_device=torch.zeros((2, 1), dtype=torch.int32),
            cu_kv_seqlens=torch.tensor([0, 5, 8], dtype=torch.int32),
        )
        inputs.__dict__.update(input_flags)
        cache = SimpleNamespace(
            kv_scale_base=torch.empty((2, 64, 132), dtype=torch.uint8)
        )
        with patch.dict(os.environ, {"DSV4_FP8_INDEXER_SCORE_CHUNK_ROWS": "2"}):
            result = self.op._get_topk_ragged(
                q, torch.ones((rows, 2, 1)), cache, params, inputs
            )
        return result, scores

    def test_per_row_dispatch_records_exact_arguments_and_none_return(self):
        for value, force in (("1", True), ("0", False), ("false", True)):
            with (
                self.subTest(radix=value),
                patch.dict(os.environ, {"DSV4_PREFILL_TOPK_FORCE_RADIX": value}),
            ):
                context = _Context()
                self.assertIsNone(self.run_topk(context))
                args = self.s.ops.dsv4_top_k_per_row_prefill.call_args.args
                self.assertIs(args[3], self.output)
                self.assertEqual(args[4:], (2, 16, 2, 3, force))
                self.assertEqual(
                    context.calls[0]["parameters"],
                    {
                        "num_rows": 2,
                        "stride0": 16,
                        "stride1": 2,
                        "force_radix_sort": force,
                    },
                )
                self.assertEqual(context.calls[0]["layer"], 7)
                self.assertEqual(context.calls[0]["backend"], "dsv4_per_row")
                torch.testing.assert_close(context.tokens[0].output, self.output)
        self.assertEqual(self.s.ops.dsv4_top_k_per_row_prefill.call_count, 3)
        self.s.ops.topk_v3_tie_break.assert_not_called()

    def test_v3_dispatch_clamps_actual_max_length(self):
        self.op._prefill_topk_backend = "topk_v3_tie_break"
        for supplied, expected in ((100, 8), (0, 1), (-5, 1), (5, 5)):
            with self.subTest(max_seq_len=supplied):
                context = _Context()
                self.run_topk(context, supplied)
                self.assertEqual(
                    self.s.ops.topk_v3_tie_break.call_args.args[4:], (3, expected)
                )
                self.assertEqual(
                    context.calls[0]["parameters"],
                    {"max_seq_len": expected, "stride0": 16, "stride1": 2},
                )
                torch.testing.assert_close(context.tokens[0].output, self.output)
        self.s.ops.dsv4_top_k_per_row_prefill.assert_not_called()

    def test_empty_short_exact_long_and_tied_valid_domains(self):
        self.logits = torch.tensor(
            [
                [-9.0, -8.0, -7.0, -6.0, -5.0, -4.0],
                [99.0, -2.0, 99.0, 99.0, 99.0, 99.0],
                [99.0, -3.0, -1.0, -2.0, 99.0, 99.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 99.0],
                [99.0, 1.0, 1.0, 1.0, 1.0, 99.0],
            ]
        )
        self.starts = torch.tensor([0, 1, 1, 0, 1], dtype=torch.int32)
        self.ends = torch.tensor([0, 2, 4, 5, 5], dtype=torch.int32)
        self.output = torch.empty((5, 3), dtype=torch.int32)
        self.op._run_prefill_topk(
            self.logits,
            self.starts,
            self.ends,
            self.output,
            6,
            context=self.context,
            total_query_rows=5,
        )
        self.assertEqual(self.output[0].tolist(), [-1, -1, -1])
        self.assertEqual(self.output[1].tolist(), [0, -1, -1])
        self.assertEqual(sorted(self.output[2].tolist()), [0, 1, 2])
        for row, length in enumerate((0, 1, 3, 5, 4)):
            valid = self.output[row][self.output[row] >= 0]
            self.assertEqual(valid.numel(), min(length, 3))
            self.assertTrue(torch.all(valid < length))
        torch.testing.assert_close(self.context.tokens[0].output, self.output)

    def test_radix_is_resolved_once_before_begin(self):
        def begin(**kw):
            os.environ["DSV4_PREFILL_TOPK_FORCE_RADIX"] = "0"
            return self.context._begin(**kw)

        self.context.begin_topk.side_effect = begin
        self.run_topk(self.context)
        self.assertTrue(self.s.ops.dsv4_top_k_per_row_prefill.call_args.args[-1])
        self.assertTrue(self.context.calls[0]["parameters"]["force_radix_sort"])

    def test_disabled_does_not_enter_capture_or_query_cuda(self):
        with (
            patch.object(
                self.op,
                "_begin_topk_capture",
                side_effect=AssertionError("capture entered"),
            ),
            patch.object(
                torch.cuda,
                "is_current_stream_capturing",
                side_effect=AssertionError("CUDA queried"),
            ),
            patch.object(torch.Tensor, "cpu", side_effect=AssertionError("D2H")),
            patch.object(torch.Tensor, "clone", side_effect=AssertionError("snapshot")),
        ):
            self.run_topk()
        self.s.compute.cuda_graph_capture_forward_enabled.assert_not_called()
        self.s.compute.cuda_graph_warmup_forward_enabled.assert_not_called()
        self.s.ops.dsv4_top_k_per_row_prefill.assert_called_once()

    def test_unselected_context_runs_kernel_once(self):
        self.context.begin_topk.return_value = None
        self.context.begin_topk.side_effect = None
        self.run_topk(self.context)
        self.s.ops.dsv4_top_k_per_row_prefill.assert_called_once()
        self.assertEqual(self.context.tokens, [])

    def test_kernel_exception_identity_preserved_even_if_cancel_fails(self):
        error = RuntimeError("original kernel failure")
        token = _Token()
        token.cancel.side_effect = RuntimeError("cancel failure")
        self.context.begin_topk.side_effect = None
        self.context.begin_topk.return_value = token
        self.s.ops.dsv4_top_k_per_row_prefill.side_effect = error
        with (
            self.assertLogs(self.s.base.__name__, level="WARNING"),
            self.assertRaises(RuntimeError) as raised,
        ):
            self.run_topk(self.context)
        self.assertIs(raised.exception, error)
        token.cancel.assert_called_once_with(reason="kernel_error")
        token.finish.assert_not_called()
        self.s.ops.dsv4_top_k_per_row_prefill.assert_called_once()

    def test_begin_and_skip_failure_are_fail_open_and_observable(self):
        self.context.begin_topk.side_effect = RuntimeError("begin failure")
        self.context.note_skip.side_effect = RuntimeError("report failure")
        with self.assertLogs(self.s.base.__name__, level="WARNING") as logs:
            self.run_topk(self.context)
        self.assertEqual(len(logs.output), 2)
        self.s.ops.dsv4_top_k_per_row_prefill.assert_called_once()
        self.context.note_skip.assert_called_once_with("capture_error", layer=7)

    def test_finish_failure_cancels_and_does_not_repeat_kernel(self):
        token = _Token()
        token.finish.side_effect = RuntimeError("finish failure")
        self.context.begin_topk.side_effect = None
        self.context.begin_topk.return_value = token
        with self.assertLogs(self.s.base.__name__, level="WARNING"):
            self.run_topk(self.context)
        token.cancel.assert_called_once_with(reason="capture_error")
        self.context.note_skip.assert_called_once_with("capture_error", layer=7)
        self.s.ops.dsv4_top_k_per_row_prefill.assert_called_once()
        self.assertTrue(torch.any(self.output >= 0))

    def test_missing_layer_is_not_layer_zero(self):
        self.op.layer_idx = None
        self.run_topk(self.context)
        self.context.begin_topk.assert_not_called()
        self.context.note_skip.assert_called_once_with("missing_layer", layer=None)
        self.s.ops.dsv4_top_k_per_row_prefill.assert_called_once()

    def test_missing_row_mapping_skips(self):
        self.op._run_prefill_topk(
            self.logits, self.starts, self.ends, self.output, 8, context=self.context
        )
        self.context.begin_topk.assert_not_called()
        self.context.note_skip.assert_called_once_with("missing_row_mapping", layer=7)

    def test_ops_warmup_and_capture_flags_reject_without_torch_capture(self):
        for name, reason in (
            ("cuda_graph_capture_forward_enabled", "graph"),
            ("cuda_graph_warmup_forward_enabled", "warmup"),
        ):
            with (
                self.subTest(flag=name),
                patch.object(self.s.compute, name, return_value=True),
                patch.object(
                    torch.cuda, "is_current_stream_capturing", return_value=False
                ),
            ):
                context = _Context()
                self.run_topk(context)
                context.begin_topk.assert_not_called()
                context.note_skip.assert_called_once_with(reason, layer=7)

    def test_chunks_cross_requests_preserve_prefix_and_raw_snapshot(self):
        with patch.dict(os.environ, {"DSV4_INDEXER_TOPK_CANONICALIZE": "1"}):
            result, scores = self.ragged(self.context)
        self.assertEqual([c["query_row_start"] for c in self.context.calls], [0, 2, 4])
        self.assertEqual([c["total_query_rows"] for c in self.context.calls], [5, 5, 5])
        self.assertEqual(self.s.ops.dsv4_top_k_per_row_prefill.call_count, 3)
        self.assertEqual(
            self.s.ops.cp_gather_indexer_k_quant_cache.call_args.args[1].shape[0], 8
        )
        middle = self.context.calls[1]
        self.assertEqual(middle["row_starts"].tolist(), [0, 5])
        self.assertEqual(middle["row_ends"].tolist(), [5, 7])
        self.assertEqual(middle["logits"].shape, (2, 8))
        raw = self.context.tokens[1].output
        self.assertEqual(raw.tolist(), [[4, 3, 2], [1, 0, -1]])
        self.assertEqual(result[2:4].tolist(), [[2, 3, 4], [5, 6, -1]])
        scores.fill_(-100)
        result.fill_(-9)
        self.assertEqual(raw.tolist(), [[4, 3, 2], [1, 0, -1]])
        self.assertNotEqual(middle["logits"][0, 0].item(), -100)

    def test_global_chunk_override(self):
        with patch.dict(os.environ, {"DSV4_CHUNK_TOKENS": "4"}):
            self.ragged(self.context)
        self.assertEqual([c["query_row_start"] for c in self.context.calls], [0, 4])

    def test_empty_query_skips_topk_but_preserves_existing_gather(self):
        result, _ = self.ragged(self.context, rows=0)
        self.assertEqual(result.shape, (0, 3))
        self.context.note_skip.assert_called_once_with("empty", layer=7)
        self.context.begin_topk.assert_not_called()
        self.s.deep.fp8_mqa_logits.assert_not_called()
        self.s.ops.cp_gather_indexer_k_quant_cache.assert_called_once()

    def test_input_graph_cp_decode_flags_skip_capture_not_inference(self):
        for flags, reason in (
            ({"is_cuda_graph": True}, "graph"),
            ({"prefill_cuda_graph_copy_params": object()}, "graph"),
            ({"context_parallel_info": object()}, "cp"),
            ({"is_prefill": False}, "decode"),
        ):
            with self.subTest(flags=flags):
                context = _Context()
                result, _ = self.ragged(context, **flags)
                self.assertEqual(result.shape, (5, 3))
                context.begin_topk.assert_not_called()
                context.note_skip.assert_called_once_with(reason, layer=7)

    def test_cp_entry_reports_skip_and_does_not_capture(self):
        empty = torch.empty(0, dtype=torch.int32)
        result = self.op._get_topk_ragged_cp(
            torch.empty((0, 2, 128)),
            torch.empty((0, 2, 1)),
            None,
            None,
            SimpleNamespace(tensor_capture_context=self.context),
            empty,
            empty,
            0,
            empty,
            empty,
            empty,
            empty,
        )
        self.assertEqual(result.shape, (0, 3))
        self.context.note_skip.assert_called_once_with("cp", layer=7)
        self.context.begin_topk.assert_not_called()

    def test_paged_entry_reports_skip_and_keeps_kernel(self):
        self.s.deep.fp8_paged_mqa_logits.return_value = self.logits
        result = self.op._get_topk_paged(
            torch.zeros((2, 2, 128)),
            torch.ones((2, 2, 1)),
            SimpleNamespace(kv_scale_base=torch.empty((2, 64, 132), dtype=torch.uint8)),
            SimpleNamespace(
                expanded_seq_lens=torch.tensor([4, 5]),
                kvlen_d=torch.tensor([4, 5]),
                schedule_metadata=torch.ones(1),
            ),
            SimpleNamespace(
                tensor_capture_context=self.context,
                decode_cu_seqlens_d=torch.arange(3),
                kv_cache_kernel_block_id_device=torch.zeros((2, 1), dtype=torch.int32),
            ),
        )
        self.assertEqual(result.shape, (2, 3))
        self.context.note_skip.assert_called_once_with("paged", layer=7)
        self.context.begin_topk.assert_not_called()
        self.s.ops.dsv4_persistent_topk.assert_called_once()

    def test_verify_and_draft_prefill_shape_still_route_to_paged(self):
        indexer = object.__new__(self.s.indexer.Indexer)
        torch.nn.Module.__init__(indexer)
        indexer.parallelism_config = None
        indexer.indexer_op = SimpleNamespace(
            _get_topk_paged=Mock(return_value=self.output),
            _get_topk_ragged=Mock(),
            _get_topk_ragged_cp=Mock(),
        )
        for flags in (
            {"is_prefill": False},
            {"is_prefill": True, "is_target_verify": True},
            {"is_prefill": True, "is_draft_extend": True},
        ):
            result = indexer._compute_topk(
                None, None, None, None, SimpleNamespace(**flags), None
            )
            self.assertIs(result, self.output)
        self.assertEqual(indexer.indexer_op._get_topk_paged.call_count, 3)
        indexer.indexer_op._get_topk_ragged.assert_not_called()
        indexer.indexer_op._get_topk_ragged_cp.assert_not_called()

    def test_hybrid_skip_reporting_failure_is_fail_open(self):
        self.context.note_skip.side_effect = RuntimeError("skip report failure")
        with self.assertLogs(self.s.indexer.__name__, level="WARNING"):
            self.s.indexer._note_indexer_capture_skip(
                SimpleNamespace(tensor_capture_context=self.context),
                "dense_fast_path",
                7,
            )

    def test_layer_propagates_through_real_indexer_constructor(self):
        config = SimpleNamespace(
            indexer_head_num=2,
            indexer_head_dim=128,
            indexer_topk=3,
            rope_head_dim=64,
            kernel_tokens_per_block=64,
            rope_config=SimpleNamespace(indexer_is_neox_style=True),
        )
        indexer = self.s.indexer.Indexer(
            config,
            {
                "mla_indexer_k_norm_w": torch.ones(128),
                "mla_indexer_k_norm_b": torch.zeros(128),
            },
            {"rope_cos_sin_cache": None},
            11,
            1e-5,
            None,
        )
        self.assertEqual(indexer.indexer_op.layer_idx, 11)
        self.assertIsNone(self.s.base.IndexerOp(2, 128, 3, 64).layer_idx)

    def test_dense_fast_path_keeps_cache_update_without_topk(self):
        indexer = object.__new__(self.s.indexer.Indexer)
        torch.nn.Module.__init__(indexer)
        indexer.layer_idx = 7
        indexer._get_k_bf16 = Mock(return_value=torch.zeros((1, 128)))
        indexer.indexer_op = SimpleNamespace(quant_k_only=Mock())
        result = indexer.forward(
            torch.zeros((1, 128)),
            None,
            None,
            SimpleNamespace(slot_mapping=None),
            SimpleNamespace(tensor_capture_context=self.context),
            use_fast_path=True,
        )
        self.assertIsNone(result)
        indexer.indexer_op.quant_k_only.assert_called_once()
        self.context.note_skip.assert_called_once_with("dense_fast_path", layer=7)
        self.context.begin_topk.assert_not_called()

    def test_shared_no_indexer_and_dense_reuse_paths(self):
        for reuse, forced, sparse, reason in (
            (True, False, True, "reused"),
            (False, True, True, "reused"),
            (True, False, False, "dense_fast_path"),
            (False, False, True, "no_indexer"),
        ):
            with self.subTest(reason=reason, forced=forced):
                context = _Context()
                attention = object.__new__(self.s.attention.MlaAttention)
                torch.nn.Module.__init__(attention)
                attention.layer_idx = 7
                attention.reuse_topk_indices = reuse
                attention.indexer = None
                shared = torch.ones((1, 3), dtype=torch.int32)
                result = attention._run_sparse_indexer(
                    None,
                    None,
                    None,
                    None,
                    SimpleNamespace(
                        is_sparse=lambda sparse=sparse: sparse,
                        attn_inputs=SimpleNamespace(tensor_capture_context=context),
                    ),
                    prev_topk_indices=shared,
                    force_reuse_topk_indices=forced,
                )
                self.assertIs(result, shared if reason == "reused" else None)
                context.note_skip.assert_called_once_with(reason, layer=7)
                context.begin_topk.assert_not_called()

    @contextmanager
    def recorder(self, tp_rank=0, config_overrides=None, identity_overrides=None):
        capture = _load(
            "rtp_llm.utils.tensor_capture", "rtp_llm/utils/tensor_capture.py"
        )
        with tempfile.TemporaryDirectory(
            prefix="indexer-capture-", dir=os.environ.get("TMPDIR")
        ) as directory:
            root = Path(directory)
            config = {
                "enabled": True,
                "authorized_capture": True,
                "layers": [7],
                "ops": ["indexer.topk"],
                "phases": ["prefill"],
                "rank_filter": "record_owner",
                "max_calls": 16,
                "max_bundle_bytes": 1 << 20,
                "max_total_bytes": 1 << 24,
                "max_inflight_bytes": 1 << 24,
            }
            config.update(config_overrides or {})
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            identity = {
                "schema_version": 1,
                "session_id": "test-session",
                "owner_id": "owner0",
                "replica_id": "replica0",
                "world_rank": 0,
                "dp_rank": 0,
                "exec_id": "42",
                "model_role": "target",
                "requests": [
                    {"phase": "prefill", "request_id": "r1", "q_len": 3, "kv_len": 2},
                    {"phase": "prefill", "request_id": "r2", "q_len": 2, "kv_len": 1},
                ],
            }
            identity.update(identity_overrides or {})
            with patch.dict(
                os.environ,
                {
                    "RTP_LLM_RECORD_SESSION": "test-session",
                    "RTP_LLM_RECORD_DIR": str(root),
                },
            ):
                recorder = capture.TensorCaptureRecorder(
                    str(config_path), tp_rank=tp_rank, world_rank=0
                )
                try:
                    context = recorder.for_execution(json.dumps(identity))
                    yield capture, recorder, context, root
                finally:
                    recorder.close()

    def test_real_capture_file_roundtrip_and_cpu_kernel_replay(self):
        for backend in ("dsv4_per_row", "topk_v3_tie_break"):
            with (
                self.subTest(backend=backend),
                self.recorder() as (capture, recorder, context, root),
            ):
                self.assertIsNotNone(context)
                self.op._prefill_topk_backend = backend
                with patch.dict(os.environ, {"DSV4_INDEXER_TOPK_CANONICALIZE": "1"}):
                    result, scores = self.ragged(context)
                torch.testing.assert_close(
                    scores, torch.arange(80, dtype=torch.float32).view(5, 16)[:, ::2]
                )
                original_scores = scores.clone()
                scores.fill_(-100)
                result.fill_(-9)
                self.assertTrue(recorder.close())
                self.assertEqual(recorder.stats()["completed"], 3)
                self.assertEqual(recorder.stats()["inflight_bytes"], 0)
                manifests = list(root.rglob("manifest.json"))
                self.assertEqual(len(manifests), 3)
                bundles = sorted(
                    (capture.load_capture(path) for path in manifests),
                    key=lambda item: item[0]["query_row_start"],
                )
                self.assertEqual([m["query_row_start"] for m, _ in bundles], [0, 2, 4])
                for manifest, tensors in bundles:
                    self.assertTrue(manifest["complete"])
                    self.assertEqual(manifest["layer"], 7)
                    self.assertEqual(manifest["execution_id"], "42")
                    self.assertEqual(manifest["backend"], backend)
                    self.assertEqual(manifest["total_query_rows"], 5)
                    self.assertEqual(
                        manifest["parameters"]["index_space"], "relative_to_row_start"
                    )
                    desc = manifest["tensors"]["logits"]
                    self.assertEqual(desc["original_stride"], [16, 2])
                    row_start = manifest["query_row_start"]
                    for row, (start, stop) in enumerate(
                        zip(
                            tensors["row_starts"].tolist(), tensors["row_ends"].tolist()
                        )
                    ):
                        saved = tensors["logits"][row]
                        torch.testing.assert_close(
                            saved[start:stop],
                            original_scores[row_start + row, start:stop],
                            atol=0,
                            rtol=0,
                        )
                        self.assertTrue(torch.all(saved[:start] == 0))
                        self.assertTrue(torch.all(saved[stop:] == 0))
                    # Reconstruct the source strides for replay; the file itself
                    # is contiguous. This is still a CPU mock-kernel replay.
                    logits = torch.empty_strided(desc["shape"], desc["original_stride"])
                    logits.copy_(tensors["logits"])
                    output = torch.empty_like(tensors["indices"])
                    self.op._run_prefill_topk(
                        logits,
                        tensors["row_starts"],
                        tensors["row_ends"],
                        output,
                        manifest["parameters"].get("max_seq_len", 5),
                    )
                    torch.testing.assert_close(
                        output, tensors["indices"], atol=0, rtol=0
                    )
                self.assertEqual(
                    bundles[1][1]["indices"].tolist(), [[4, 3, 2], [1, 0, -1]]
                )

    def test_real_oversize_and_layer_filter_leave_inference_unchanged(self):
        for overrides, reason in (
            ({"max_bundle_bytes": 1}, "oversize"),
            ({"layers": [8]}, "layer_filter"),
        ):
            with (
                self.subTest(reason=reason),
                self.recorder(config_overrides=overrides) as (
                    _,
                    recorder,
                    context,
                    root,
                ),
            ):
                result, _ = self.ragged(context)
                self.assertEqual(result.shape, (5, 3))
                self.assertTrue(recorder.close())
                self.assertEqual(recorder.stats()["completed"], 0)
                self.assertEqual(recorder.stats()["skip_reasons"][reason], 3)
                self.assertEqual(recorder.stats()["inflight_bytes"], 0)
                self.assertEqual(list(root.rglob("manifest.json")), [])

    def test_real_mixed_decode_or_propose_context_is_rejected(self):
        for overrides in (
            {"model_role": "draft"},
            {
                "requests": [
                    {"phase": "decode", "request_id": "r1", "q_len": 1, "kv_len": 2}
                ]
            },
        ):
            with (
                self.subTest(overrides=overrides),
                self.recorder(identity_overrides=overrides) as (
                    _,
                    recorder,
                    context,
                    root,
                ),
            ):
                self.assertIsNone(context)
                self.ragged(context)
                self.assertTrue(recorder.close())
                self.assertEqual(recorder.stats()["skip_reasons"]["invalid_context"], 1)
                self.assertEqual(list(root.rglob("manifest.json")), [])

    def test_real_non_owner_is_not_given_context(self):
        with self.recorder(tp_rank=1) as (_, recorder, context, root):
            self.assertIsNone(context)
            self.ragged(context)
            self.assertTrue(recorder.close())
            self.assertEqual(recorder.stats()["skip_reasons"]["non_owner_rank"], 1)
            self.assertEqual(list(root.rglob("manifest.json")), [])

            self.assertEqual(self.s.ops.dsv4_top_k_per_row_prefill.call_count, 3)

    def test_real_microbatch_row_mismatch_does_not_publish(self):
        with self.recorder() as (_, recorder, context, root):
            # A leaked context for the full five-row execution must not accept
            # a three-row microbatch as a complete execution.
            result, _ = self.ragged(context, rows=3)
            self.assertEqual(result.shape, (3, 3))
            self.assertTrue(recorder.close())
            self.assertEqual(recorder.stats()["completed"], 0)
            self.assertEqual(
                recorder.stats()["skip_reasons"]["invalid_topk_or_snapshot_error"], 2
            )
            self.assertEqual(list(root.rglob("manifest.json")), [])
            self.assertEqual(self.s.ops.dsv4_top_k_per_row_prefill.call_count, 2)

    def test_real_kernel_failure_releases_reservation_without_manifest(self):
        with self.recorder() as (_, recorder, context, root):
            error = RuntimeError("kernel failure")
            self.s.ops.dsv4_top_k_per_row_prefill.side_effect = error
            with self.assertRaises(RuntimeError) as raised:
                self.ragged(context)
            self.assertIs(raised.exception, error)
            self.assertTrue(recorder.close())
            self.assertEqual(recorder.stats()["skip_reasons"]["kernel_error"], 1)
            self.assertEqual(recorder.stats()["inflight_bytes"], 0)
            self.assertEqual(recorder.stats()["reserved_bytes"], 0)
            self.assertEqual(list(root.rglob("manifest.json")), [])
            self.s.ops.dsv4_top_k_per_row_prefill.assert_called_once()

    def test_before_snapshot_survives_kernel_mutating_input(self):
        before = self.logits.clone()

        def kernel(*args):
            _cpu_topk(*args)
            args[0].fill_(-100)

        self.s.ops.dsv4_top_k_per_row_prefill.side_effect = kernel
        self.run_topk(self.context)
        torch.testing.assert_close(self.context.calls[0]["logits"], before)
        self.assertTrue(torch.all(self.logits == -100))

    def test_missing_shared_indices_remains_an_error(self):
        attention = object.__new__(self.s.attention.MlaAttention)
        attention.layer_idx = 7
        attention.reuse_topk_indices = True
        with self.assertRaisesRegex(RuntimeError, "needs previous top-k"):
            attention._run_sparse_indexer(
                None, None, None, None, SimpleNamespace(is_sparse=lambda: True)
            )


if __name__ == "__main__":
    unittest.main()
