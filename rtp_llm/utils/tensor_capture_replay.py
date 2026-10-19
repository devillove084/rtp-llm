"""Validate an indexer.topk capture without a model or replay its CUDA operator.

    /opt/conda310/bin/python3 -m rtp_llm.utils.tensor_capture_replay MANIFEST
    /opt/conda310/bin/python3 -m rtp_llm.utils.tensor_capture_replay MANIFEST --device cuda

CPU mode checks the current source contract, NOT execution of the captured kernel.
This is the stable-index contract, not a promise about arbitrary older binaries:
DSV4 insertion compares actual token indices on equal scores; radix embeds the
same index priority in its key. Legacy insertion used candidate arrival slots.
A mismatch requires checking the capture/loaded library provenance; the loader
cannot establish a binary's source revision from an operator name or timestamp.
Both backends select by descending FP32 bit-order key, then ascending relative
index; long-row output order is unspecified. Short rows emit 0..length-1, -1.
V3's effective candidate length is min(row_end-row_start, max_seq_len).
Signed zeros are distinct. Valid-domain NaNs are explicitly unsupported: their
histogram/backend semantics are not a numeric maximum-set contract. Infinities
are supported. Invalid-domain snapshots must contain the capture protocol's zeros.

Reading requires no capture authorization configuration and sends no data out.
The loader verifies checksums, not authenticity: coordinated metadata/payload
rewrites cannot be detected. JSON summaries omit activation values and identities.
Exit codes: 0 validated, 1 invalid/failed, 2 unsupported, 3 CUDA unavailable.
"""

import argparse
import importlib
import json
import sys

import torch

from rtp_llm.utils.tensor_capture import load_capture


class UnsupportedCapture(ValueError):
    """The artifact cannot be checked/replayed under the supported contract."""


class ReplayUnavailable(RuntimeError):
    """CUDA or the real operator library is unavailable (not a passing replay)."""


_MAX_STORAGE_BYTES = 1024**3


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, minimum=0):
    return type(value) is int and minimum <= value <= 2**31 - 1


def _storage_elements(desc):
    # A conservative nonoverlap proof admits dense, transposed and padded views.
    # Exotic interleaved layouts not proven by this rule are explicitly refused.
    shape, strides = desc["shape"], desc["original_stride"]
    span = 1
    for stride, size in sorted(zip(strides, shape)):
        if size > 1:
            if stride < span:
                raise UnsupportedCapture("overlapping or unsupported source strides")
            span += (size - 1) * stride
    return desc["storage_offset"] + span


def _contiguous(shape, strides):
    expected = 1
    for size, stride in reversed(list(zip(shape, strides))):
        if size > 1 and stride != expected:
            return False
        expected *= max(size, 1)
    return True


def _check_schema(manifest, tensors):
    for field, expected in (
        ("op", "indexer.topk"),
        ("phase", "prefill"),
        ("invalid_logits", "zero_filled_snapshot_only"),
        ("nonfinite_policy", "valid_domain_bits_preserved_backend_semantics"),
        ("output_contract", "raw_relative_indices_minus_one_padding_backend_order"),
    ):
        _require(manifest.get(field) == expected, "unsupported " + field)
    _require(
        manifest.get("finite_values_checked") is False, "invalid finite_values_checked"
    )
    backend = manifest.get("backend")
    if backend not in ("dsv4_per_row", "topk_v3_tie_break"):
        raise UnsupportedCapture("unsupported backend: " + str(backend))
    names = {"logits", "row_starts", "row_ends", "indices"}
    _require(
        set(tensors) == names, "expected exactly logits/row_starts/row_ends/indices"
    )
    logits = tensors["logits"]
    _require(
        logits.dtype == torch.float32 and logits.ndim == 2,
        "logits must be float32 [rows, columns]",
    )
    rows, columns = logits.shape
    _require(rows > 0 and columns > 0, "capture requires nonempty logits dimensions")
    params = manifest.get("parameters")
    _require(isinstance(params, dict), "missing backend parameters")
    common = {"k", "index_space", "stride0", "stride1"}
    expected = common | (
        {"num_rows", "force_radix_sort"}
        if backend == "dsv4_per_row"
        else {"max_seq_len"}
    )
    _require(set(params) == expected, "missing/unknown backend parameters")
    k = params["k"]
    _require(_integer(k, 1), "k must be a positive int32")
    _require(params["index_space"] == "relative_to_row_start", "invalid index_space")
    for name in ("row_starts", "row_ends", "indices"):
        tensor = tensors[name]
        expected_shape = (rows, k) if name == "indices" else (rows,)
        _require(
            tensor.dtype == torch.int32 and tuple(tensor.shape) == expected_shape,
            name + " must be int32 with shape " + str(expected_shape),
        )
    for name in ("query_row_start", "query_row_count", "total_query_rows"):
        _require(_integer(manifest.get(name)), "invalid " + name)
    _require(
        manifest["query_row_count"] == rows
        and manifest["query_row_start"] + rows <= manifest["total_query_rows"],
        "inconsistent query chunk",
    )
    desc = manifest["tensors"]["logits"]
    for axis in (0, 1):
        key = "stride" + str(axis)
        _require(
            _integer(params[key]) and params[key] == desc["original_stride"][axis],
            "parameter/source " + key + " mismatch",
        )
    if backend == "dsv4_per_row":
        _require(
            _integer(params["num_rows"]) and params["num_rows"] == rows,
            "num_rows mismatch",
        )
        _require(
            type(params["force_radix_sort"]) is bool, "force_radix_sort must be boolean"
        )
    else:
        if k not in (512, 1024, 2048):
            raise UnsupportedCapture("topk_v3_tie_break supports K=512/1024/2048")
        _require(
            _integer(params["max_seq_len"], 1) and params["max_seq_len"] <= columns,
            "invalid max_seq_len",
        )
        if params["stride1"] != 1:
            raise UnsupportedCapture("topk_v3_tie_break requires source stride1=1")
    storage_bytes = 0
    for name, tensor in tensors.items():
        desc = manifest["tensors"][name]
        storage_bytes += _storage_elements(desc) * tensor.element_size()
        if name != "logits" and not _contiguous(desc["shape"], desc["original_stride"]):
            raise UnsupportedCapture(
                name + " must have contiguous source layout for these kernels"
            )
    if storage_bytes > _MAX_STORAGE_BYTES:
        raise UnsupportedCapture("reconstructed source storage exceeds 1 GiB limit")
    starts, ends = tensors["row_starts"], tensors["row_ends"]
    _require(
        bool(((starts >= 0) & (ends >= starts) & (ends <= columns)).all()),
        "invalid row bounds",
    )
    for row in range(rows):
        start, end = int(starts[row]), int(ends[row])
        _require(
            bool((logits[row, :start] == 0).all())
            and bool((logits[row, end:] == 0).all()),
            "invalid-domain logits are not zero-filled",
        )
        if bool(torch.isnan(logits[row, start:end]).any()):
            raise UnsupportedCapture(
                "valid-domain NaN: backend histogram semantics unsupported by this validator"
            )
    return params


def _score_keys(values):
    # Compare exact FP32 keys, including signed zero, without floating tolerance.
    bits = values.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return torch.where(
        (bits & 0x80000000) != 0, (~bits) & 0xFFFFFFFF, bits ^ 0x80000000
    )


def _check_indices(manifest, tensors, indices):
    params = manifest["parameters"]
    k = params["k"]
    counts = {
        "rows": indices.shape[0],
        "empty_rows": 0,
        "short_rows": 0,
        "selected_indices": 0,
        "padding_indices": 0,
        "boundary_tie_rows": 0,
    }
    for row, output in enumerate(indices):
        start, end = int(tensors["row_starts"][row]), int(tensors["row_ends"][row])
        length = end - start
        if manifest["backend"] == "topk_v3_tie_break":
            length = min(length, params["max_seq_len"])
        count = min(k, length)
        selected = output[:count].to(torch.int64)
        prefix = "row %d: " % row
        _require(
            bool(((selected >= 0) & (selected < length)).all()),
            prefix + "missing/out-of-range index",
        )
        _require(bool((output[count:] == -1).all()), prefix + "invalid -1 padding")
        _require(selected.unique().numel() == count, prefix + "duplicate index")
        if length <= k:
            _require(
                torch.equal(selected, torch.arange(length)),
                prefix + "short-row indices must be sequential",
            )
        else:
            keys = _score_keys(tensors["logits"][row, start : start + length])
            selected_keys = keys[selected]
            threshold = selected_keys.min()
            # No unselected candidate may outrank the weakest selected candidate.
            # Equal keys use relative index, NOT torch.topk's arbitrary tie order.
            last_tie = selected[selected_keys == threshold].max()
            chosen = torch.zeros(length, dtype=torch.bool)
            chosen[selected] = True
            relative = torch.arange(length)
            _require(
                not bool(((keys > threshold) & ~chosen).any()),
                prefix + "not the maximal candidate set (score)",
            )
            better_tie = (keys == threshold) & (relative < last_tie)
            _require(
                not bool((better_tie & ~chosen).any()),
                prefix
                + "not the maximal candidate set (index tie-break under current-source contract; "
                "verify capture/loaded library provenance)",
            )
            counts["boundary_tie_rows"] += int(int((keys == threshold).sum()) > 1)
        counts["empty_rows"] += int(length == 0)
        counts["short_rows"] += int(length < k)
        counts["selected_indices"] += count
        counts["padding_indices"] += k - count
    return counts


def _restore(tensor, desc, device):
    storage = torch.empty(_storage_elements(desc), dtype=tensor.dtype, device=device)
    result = storage.as_strided(
        desc["shape"], desc["original_stride"], desc["storage_offset"]
    )
    result.copy_(tensor)
    return result


def _cuda_operator(backend):
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise ReplayUnavailable("CUDA unavailable; no kernel replay performed")
    name = "dsv4_top_k_per_row_prefill" if backend == "dsv4_per_row" else backend
    try:
        ops = importlib.import_module("rtp_llm.ops").rtp_llm_ops
        operator = getattr(ops, name)
        if not callable(operator):
            raise AttributeError("operator is not callable")
    except Exception as error:
        raise ReplayUnavailable(
            "real CUDA operator unavailable: " + str(error)
        ) from error
    return operator


def replay_capture(path, device="cpu"):
    """Return a summary or raise ValueError/UnsupportedCapture/ReplayUnavailable."""
    _require(device in ("cpu", "cuda"), "device must be cpu or cuda")
    manifest, tensors = load_capture(path)
    params = _check_schema(manifest, tensors)
    counts = _check_indices(manifest, tensors, tensors["indices"])
    summary = dict(
        status="validated",
        mode="cpu_contract_validation",
        kernel_replayed=False,
        backend=manifest["backend"],
        k=params["k"],
        counts=counts,
        selection_contract="descending_fp32_bit_key_then_ascending_relative_index",
        nan_policy="unsupported_in_valid_domain",
    )
    if device == "cpu":
        return summary
    operator = _cuda_operator(manifest["backend"])
    restored = {
        name: _restore(tensor, manifest["tensors"][name], "cuda")
        for name, tensor in tensors.items()
    }
    output = restored["indices"]
    # Never let an unwritten output slot accidentally match the captured result.
    output.fill_(-2)
    args = (restored["logits"], restored["row_starts"], restored["row_ends"], output)
    if manifest["backend"] == "dsv4_per_row":
        operator(
            *args,
            params["num_rows"],
            params["stride0"],
            params["stride1"],
            params["k"],
            params["force_radix_sort"]
        )
    else:
        operator(*args, params["k"], params["max_seq_len"])
    torch.cuda.synchronize()
    actual = output.cpu()
    _check_indices(manifest, tensors, actual)
    _require(
        torch.equal(actual.sort(dim=1).values, tensors["indices"].sort(dim=1).values),
        "CUDA/capture selected sets differ",
    )
    summary.update(
        mode="cuda_kernel_replay",
        kernel_replayed=True,
        comparison="selected_sets_and_short_row_order",
        raw_order_equal=torch.equal(actual, tensors["indices"]),
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("manifest")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    try:
        summary = replay_capture(args.manifest, args.device)
    except (ValueError, RuntimeError, OSError) as error:
        if isinstance(error, ReplayUnavailable):
            status, code = "unavailable", 3
        elif isinstance(error, UnsupportedCapture):
            status, code = "unsupported", 2
        else:
            status, code = "failed", 1
        print(
            json.dumps(dict(status=status, device=args.device, error=str(error))),
            file=sys.stderr,
        )
        return code
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
