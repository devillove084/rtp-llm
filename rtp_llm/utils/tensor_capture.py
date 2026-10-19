"""ExecutionRecorder artifacts for explicitly authorized, eager prefill canaries only.

Set RTP_LLM_RECORD_DIR (absolute) and RTP_LLM_RECORD_SESSION to the existing
ExecutionRecorder values. Opt in with enabled=true AND authorized_capture=true;
this authorizes the ENTIRE batch/chunk, not individual rows. Never enable on
unapproved production traffic: activations are sensitive even with anonymous IDs.

Required enabled configuration: layers, ops=["indexer.topk"], phases=["prefill"],
rank_filter="record_owner", and positive max_calls/max_bundle_bytes/
max_total_bytes/max_inflight_bytes. No GPU work or filesystem/thread side effects
when disabled. Instances belong to models, never to a process-global singleton.

Budgets include raw payload and a conservative manifest allowance (not run.json
or interpreter/torch allocator overhead). Four times that bound is reserved for
inflight storage BEFORE copying: snapshots, D2H/contiguity temporaries, encoding
bytes and JSON. Reservations survive dequeue and are released only after owned
storage is discarded. Published and failed write attempts retain their total-byte
charge (incomplete files may remain); cancellation before writing refunds it.
This deliberately overestimates rather than truncating a bundle. Synchronous
producer-stream D2H is intentional; this is diagnostic code. Writer and reader
share a hard 1 GiB aggregate raw payload limit, regardless of configured budgets.
The bytes_published statistic counts tensors plus manifest only for complete
published bundles; it excludes run.json, incomplete files and failed IO bytes.
"""

import hashlib
import json
import logging
import math
import os
import platform
import queue
import re
import stat
import sys
import threading
import uuid
from collections import Counter
from pathlib import Path
from types import MappingProxyType
from typing import Optional

import torch

_LOG = logging.getLogger(__name__)
_VERSION = 1
_MAX_JSON = 1024 * 1024
_MAX_FILE_BYTES = 1024 * 1024 * 1024
_MAX_DIM = 2**31 - 1
_INFLIGHT_FACTOR = 4
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_DTYPES = {
    name: getattr(torch, name)
    for name in (
        "bool",
        "uint8",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
    )
    if hasattr(torch, name)
}


def _integer(value, minimum=0, maximum=_MAX_DIM):
    return type(value) is int and minimum <= value <= maximum


def _identifier(value):
    return (
        isinstance(value, str)
        and _IDENTIFIER.fullmatch(value) is not None
        and ".." not in value
    )


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _json_loads(value):
    if len(value) > _MAX_JSON:
        raise ValueError("JSON too large")
    return json.loads(
        value,
        object_pairs_hook=_json_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")),
    )


def _json_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value):
    if isinstance(value, MappingProxyType):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


def _config(path):
    with open(path, "rb") as stream:
        value = _json_loads(stream.read(_MAX_JSON + 1))
    fields = {
        "enabled",
        "authorized_capture",
        "layers",
        "ops",
        "phases",
        "rank_filter",
        "max_calls",
        "max_bundle_bytes",
        "max_total_bytes",
        "max_inflight_bytes",
    }
    if not isinstance(value, dict) or value.keys() - fields:
        raise ValueError("unknown tensor capture configuration field")
    for key in ("enabled", "authorized_capture"):
        if key in value and type(value[key]) is not bool:
            raise ValueError(key + " must be boolean")
    value.setdefault("enabled", False)
    value.setdefault("authorized_capture", False)
    if "layers" in value:
        layers = value["layers"]
        if not isinstance(layers, list) or not all(_integer(x) for x in layers):
            raise ValueError("layers must contain nonnegative integers")
        value["layers"] = sorted(set(layers))
    for key, expected in (
        ("ops", ["indexer.topk"]),
        ("phases", ["prefill"]),
        ("rank_filter", "record_owner"),
    ):
        if key in value and value[key] != expected:
            raise ValueError("unsupported " + key)
    for key in (
        "max_calls",
        "max_bundle_bytes",
        "max_total_bytes",
        "max_inflight_bytes",
    ):
        if key in value and not _integer(value[key], 1, 2**63 - 1):
            raise ValueError(key + " must be a positive integer")
    if value["enabled"]:
        if not value["authorized_capture"]:
            raise ValueError(
                "authorized_capture=true is required; authorized canaries only"
            )
        if fields - value.keys() or not value["layers"]:
            raise ValueError("enabled capture requires all selectors and budgets")
    return value


def config_enabled(path: str) -> bool:
    """Validate the capture configuration without starting a recorder or doing GPU work."""
    return _config(path)["enabled"]


def _identity(text, session, rank):
    source = _json_loads(text)
    if not isinstance(source, dict):
        raise ValueError("invalid execution identity")
    if (
        type(source.get("schema_version")) is not int
        or source["schema_version"] != _VERSION
    ):
        raise ValueError("unsupported identity version")
    if source.get("session_id") != session or source.get("model_role") != "target":
        raise ValueError("session or role mismatch")
    for key in ("owner_id", "replica_id"):
        if not _identifier(source.get(key)):
            raise ValueError("unsafe identity")
    for key in ("world_rank", "dp_rank"):
        if not _integer(source.get(key)):
            raise ValueError("invalid rank")
    if source["world_rank"] != rank:
        raise ValueError("rank mismatch")
    if (
        not isinstance(source.get("exec_id"), str)
        or re.fullmatch(r"[0-9]{1,40}", source["exec_id"]) is None
    ):
        raise ValueError("invalid execution id")
    rows = source.get("requests")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 4096:
        raise ValueError("invalid batch rows")
    requests = []
    q_total = kv_total = 0
    for row in rows:
        if not isinstance(row, dict) or row.get("phase") != "prefill":
            raise ValueError("not pure prefill")
        if type(row.get("is_fake", False)) is not bool or row.get("is_fake", False):
            raise ValueError("fake batch row")
        if (
            not isinstance(row.get("request_id"), str)
            or re.fullmatch(r"r[0-9]{1,40}", row["request_id"]) is None
        ):
            raise ValueError("request id must be recorder-anonymized")
        if not _integer(row.get("q_len"), 1) or not _integer(row.get("kv_len")):
            raise ValueError("invalid sequence lengths")
        clean = {k: row[k] for k in ("phase", "request_id", "q_len", "kv_len")}
        if "sequence_id" in row:
            if not _integer(row["sequence_id"]):
                raise ValueError("invalid sequence id")
            clean["sequence_id"] = row["sequence_id"]
        q_total += row["q_len"]
        kv_total += row["q_len"] + row["kv_len"]
        if q_total > _MAX_DIM or kv_total > _MAX_DIM:
            raise ValueError("batch too large")
        requests.append(clean)
    clean = {
        k: source[k]
        for k in (
            "schema_version",
            "session_id",
            "owner_id",
            "replica_id",
            "world_rank",
            "dp_rank",
            "exec_id",
            "model_role",
        )
    }
    clean["requests"] = requests
    return clean


def _parameters(parameters, k):
    if not isinstance(parameters, dict) or len(parameters) > 32:
        raise ValueError("invalid backend parameters")
    clean = {}
    for key, value in parameters.items():
        if not _identifier(key):
            raise ValueError("invalid parameter name")
        if type(value) in (int, bool):
            if type(value) is int and abs(value) > 2**63 - 1:
                raise ValueError("parameter too large")
        elif type(value) is float:
            if not math.isfinite(value):
                raise ValueError("nonfinite parameter")
        elif not _identifier(value):
            raise ValueError("backend parameters must be small scalars")
        clean[key] = value
    if (
        clean.get("k", k) != k
        or clean.get("index_space", "relative_to_row_start") != "relative_to_row_start"
    ):
        raise ValueError("inconsistent topk parameters")
    clean.update(k=k, index_space="relative_to_row_start")
    return clean


def _descriptor(tensor):
    if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
        raise ValueError("only dense tensors are supported")
    name = str(tensor.dtype).removeprefix("torch.")
    if name not in _DTYPES:
        raise ValueError("unsupported tensor dtype")
    return {
        "dtype": name,
        "shape": list(tensor.shape),
        "original_stride": list(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "serialization": "contiguous_row_major",
        "byte_order": "little",
        "nbytes": tensor.numel() * tensor.element_size(),
    }


def _snapshot(tensor):
    # copy=True is essential on CPU too: detach()/contiguous() may alias input.
    return tensor.detach().to(
        device="cpu", copy=True, memory_format=torch.contiguous_format
    )


def _encode_tensor(tensor):
    """Raw bit-preserving encoder; also usable for noncontiguous/scalar/empty tensors."""
    desc = _descriptor(tensor)
    cpu = tensor.detach().to(device="cpu").contiguous()
    octets = cpu.reshape(-1).view(torch.uint8).numpy()
    if sys.byteorder != "little" and tensor.element_size() > 1:
        octets = octets.reshape(-1, tensor.element_size())[:, ::-1]
    data = octets.tobytes()
    desc["sha256"] = hashlib.sha256(data).hexdigest()
    return desc, data


def _private_directory(path):
    """Never follow a directory symlink, including existing parent components."""
    path = Path(os.path.abspath(path))
    for part in reversed((path,) + tuple(path.parents)[:-1]):
        try:
            part.mkdir(mode=0o700)
        except FileExistsError:
            pass
        if not stat.S_ISDIR(part.lstat().st_mode):
            raise ValueError("capture directory is not a real directory")
    return path


def _atomic_file(path, data):
    temporary = path.with_name(path.name + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class TensorCaptureRecorder:
    def __init__(self, config_path: str, *, tp_rank: int = 0, world_rank: int = 0):
        self._config = _config(config_path)
        if not _integer(tp_rank) or not _integer(world_rank):
            raise ValueError("invalid recorder rank")
        self._tp_rank, self._world_rank = tp_rank, world_rank
        self._condition = threading.Condition()
        self._counts = Counter()
        self._reasons = Counter()
        self._closed = False
        self._failed = False
        self._thread = None
        self._sink = None
        self._owner = None
        self._pending = 0
        self._inflight = 0
        self._reserved = 0
        self._queue = queue.Queue(maxsize=min(self._config.get("max_calls", 1), 32))
        self._session = self._directory = None
        if self._config["enabled"]:
            self._session = os.environ.get("RTP_LLM_RECORD_SESSION")
            self._directory = os.environ.get("RTP_LLM_RECORD_DIR")
            if (
                not _identifier(self._session)
                or not self._directory
                or not os.path.isabs(self._directory)
            ):
                raise ValueError("valid RTP_LLM_RECORD_DIR/SESSION required")
            if ".." in Path(self._directory).parts or "\x00" in self._directory:
                raise ValueError("unsafe record directory")
            directory = Path(self._directory)
            for part in reversed((directory,) + tuple(directory.parents)[:-1]):
                try:
                    mode = part.lstat().st_mode
                except FileNotFoundError:
                    continue
                if not stat.S_ISDIR(mode):
                    raise ValueError(
                        "record directory must not contain files or symlinks"
                    )

    def note_skip(self, reason: str) -> None:
        if not _identifier(reason) or len(reason) > 64:
            reason = "invalid_reason"
        with self._condition:
            if reason not in self._reasons and len(self._reasons) >= 64:
                reason = "other"
            self._counts["skipped"] += 1
            self._reasons[reason] += 1

    def stats(self) -> dict:
        with self._condition:
            result = {
                key: self._counts[key]
                for key in (
                    "selected",
                    "completed",
                    "skipped",
                    "io_error",
                    "writer_error",
                    "bytes_published",
                )
            }
            result.update(
                inflight_bytes=self._inflight,
                reserved_bytes=self._reserved,
                pending=self._pending,
                queue_size=self._queue.qsize(),
                closed=self._closed,
                failed=self._failed,
                skip_reasons=dict(self._reasons),
            )
            return result

    def _start(self, identity):
        # Only thread creation happens under the lock, never filesystem IO.
        self._owner = identity["owner_id"]
        thread = threading.Thread(
            target=self._writer, name="tensor-capture", daemon=True
        )
        thread.start()
        self._thread = thread

    def _initialize_sink(self):
        root = _private_directory(
            Path(self._directory) / ("owner-" + self._owner) / "tensor-capture"
        )
        self._sink = root / ("sink-" + uuid.uuid4().hex)
        self._sink.mkdir(mode=0o700)
        metadata = {
            "schema_version": _VERSION,
            "format": "rtp_tensor_capture",
            "config": self._config,
            "config_sha256": hashlib.sha256(_json_bytes(self._config)).hexdigest(),
            "session_id": self._session,
            "owner_id": self._owner,
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "torch_git_version": getattr(torch.version, "git_version", None),
            "cuda_build_version": torch.version.cuda,
            "code_version": None,
            "model_provenance": None,
            "gpu": None,
            "inflight_factor": _INFLIGHT_FACTOR,
            "authorization": "explicit_entire_batch_canary",
            "budget_scope": "payload_and_manifest_bound_excluding_run_metadata",
        }
        # Do not probe availability/device_count: those can initialize CUDA.
        if torch.cuda.is_initialized():
            try:
                props = torch.cuda.get_device_properties(torch.cuda.current_device())
                metadata["gpu"] = {
                    "name": props.name,
                    "major": props.major,
                    "minor": props.minor,
                }
            except Exception:
                self.note_skip("gpu_metadata_unavailable")
        _atomic_file(self._sink / "run.json", _json_bytes(metadata))

    def for_execution(self, context_json: str) -> Optional["TensorCaptureContext"]:
        if not self._config["enabled"]:
            self.note_skip("disabled")
            return None
        if self._tp_rank != 0:
            self.note_skip("non_owner_rank")
            return None
        try:
            identity = _identity(context_json, self._session, self._world_rank)
        except Exception:
            self.note_skip("invalid_context")
            return None
        with self._condition:
            if self._closed or self._failed:
                self.note_skip("closed" if self._closed else "writer_unavailable")
                return None
            if self._owner is not None and self._owner != identity["owner_id"]:
                self.note_skip("owner_mismatch")
                return None
            if self._thread is None:
                try:
                    self._start(identity)
                except Exception:
                    self._failed = True
                    self._counts["writer_error"] += 1
                    self.note_skip("writer_start_failed")
                    return None
        return TensorCaptureContext(self, identity)

    def _reserve(self, size):
        with self._condition:
            reason = None
            if self._closed or self._failed:
                reason = "closed" if self._closed else "writer_unavailable"
            elif size > self._config["max_bundle_bytes"]:
                reason = "oversize"
            elif self._counts["selected"] >= self._config["max_calls"]:
                reason = "max_calls"
            elif self._reserved + size > self._config["max_total_bytes"]:
                reason = "total_budget"
            elif (
                self._inflight + _INFLIGHT_FACTOR * size
                > self._config["max_inflight_bytes"]
            ):
                reason = "inflight_budget"
            elif self._queue.full():
                reason = "queue_full"
            if reason:
                self.note_skip(reason)
                return None
            call = self._counts["selected"]
            self._counts["selected"] += 1
            self._reserved += size
            self._inflight += _INFLIGHT_FACTOR * size
            self._pending += 1
            return call

    def _release(self, size, retain_total=False):
        with self._condition:
            self._inflight -= _INFLIGHT_FACTOR * size
            self._pending -= 1
            if not retain_total:
                self._reserved -= size
            self._condition.notify_all()

    def _enqueue(self, manifest, tensors, size):
        with self._condition:
            if self._closed or self._failed:
                self.note_skip("closed" if self._closed else "writer_unavailable")
                return False
            try:
                self._queue.put_nowait((manifest, tensors, size))
                return True
            except queue.Full:
                self.note_skip("queue_full")
                return False

    def _write_bundle(self, manifest, tensors, size):
        call_dir = self._sink / ("call-%08d" % manifest["call_id"])
        call_dir.mkdir(mode=0o700)
        written = 0
        for name, tensor in tensors.items():
            desc, data = _encode_tensor(tensor)
            original = manifest["tensors"][name]
            desc.update(
                original_stride=original["original_stride"],
                storage_offset=original["storage_offset"],
                file=name + ".bin",
            )
            manifest["tensors"][name] = desc
            _atomic_file(call_dir / desc["file"], data)
            written += len(data)
            del data
        manifest["complete"] = True
        data = _json_bytes(manifest)
        written += len(data)
        if written > size or len(data) > _MAX_JSON:
            raise ValueError("manifest exceeded reserved bound")
        # Publication point: readers never consider .tmp or orphaned .bin files complete.
        _atomic_file(call_dir / "manifest.json", data)
        return written

    def _writer(self):
        try:
            self._initialize_sink()
        except Exception:
            with self._condition:
                self._failed = True
                self._counts["writer_error"] += 1
        while True:
            with self._condition:
                if (self._closed or self._failed) and self._pending == 0:
                    return
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            manifest, tensors, size = item
            attempted = False
            try:
                if self._failed:
                    self.note_skip("writer_unavailable")
                    continue
                # Orphaned files still consume disk; never refund a write attempt.
                attempted = True
                written = self._write_bundle(manifest, tensors, size)
                with self._condition:
                    self._counts["completed"] += 1
                    self._counts["bytes_published"] += written
            except Exception:
                with self._condition:
                    self._counts["io_error"] += 1
            finally:
                tensors.clear()
                del tensors, manifest, item
                self._release(size, retain_total=attempted)
                self._queue.task_done()

    def close(self, timeout: float = 5) -> bool:
        if (
            not isinstance(timeout, (float, int))
            or not math.isfinite(timeout)
            or timeout < 0
        ):
            raise ValueError("timeout must be finite and nonnegative")
        with self._condition:
            self._closed = True
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        with self._condition:
            done = self._pending == 0 and (thread is None or not thread.is_alive())
        _LOG.info("Tensor capture close complete=%s stats=%s", done, self.stats())
        return done


class TensorCaptureContext:
    def __init__(self, recorder, identity):
        self._recorder = recorder
        self._identity = _freeze(identity)
        self._identity_bytes = len(_json_bytes(identity))

    @property
    def identity(self):
        return self._identity

    def note_skip(self, reason: str, layer: Optional[int] = None) -> None:
        self._recorder.note_skip(reason)

    def begin_topk(
        self,
        *,
        layer: int,
        logits: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        k: int,
        backend: str,
        parameters: dict,
        query_row_start: int,
        total_query_rows: int
    ) -> Optional["_TopKCaptureToken"]:
        recorder = self._recorder
        if not _integer(layer) or layer not in recorder._config["layers"]:
            self.note_skip("layer_filter", layer)
            return None
        tensors = {}
        reserved = False
        try:
            if not _integer(k, 1) or not _identifier(backend):
                raise ValueError("invalid topk scalar")
            if not _integer(query_row_start) or not _integer(total_query_rows, 1):
                raise ValueError("invalid query chunk")
            if (
                logits.dtype != torch.float32
                or logits.ndim != 2
                or logits.shape[0] == 0
            ):
                raise ValueError("only nonempty float32 logits supported")
            rows, columns = logits.shape
            if any(
                x.dtype != torch.int32 or x.shape != (rows,)
                for x in (row_starts, row_ends)
            ):
                raise ValueError("only int32 row ranges supported")
            requests = self._identity["requests"]
            if (
                total_query_rows != sum(x["q_len"] for x in requests)
                or query_row_start + rows > total_query_rows
            ):
                raise ValueError("incomplete query mapping")
            if columns != sum(x["q_len"] + x["kv_len"] for x in requests):
                raise ValueError("incomplete key domain")
            params = _parameters(parameters, k)
            descriptors = {
                name: _descriptor(tensor)
                for name, tensor in (
                    ("logits", logits),
                    ("row_starts", row_starts),
                    ("row_ends", row_ends),
                )
            }
            payload = sum(x["nbytes"] for x in descriptors.values()) + rows * k * 4
            if payload > _MAX_FILE_BYTES:
                self.note_skip("payload_limit", layer)
                return None
            size = payload + self._identity_bytes + len(_json_bytes(params)) + 4096
            call = recorder._reserve(size)
            if call is None:
                return None
            reserved = True
            tensors["row_starts"] = _snapshot(row_starts)
            tensors["row_ends"] = _snapshot(row_ends)
            # Recorded kv_len is the cached prefix, not the final KV length.
            q_offset = key_offset = 0
            for request in requests:
                q_len, prefix = request["q_len"], request["kv_len"]
                begin = max(query_row_start, q_offset)
                end = min(query_row_start + rows, q_offset + q_len)
                for global_row in range(begin, end):
                    row = global_row - query_row_start
                    start = int(tensors["row_starts"][row])
                    stop = int(tensors["row_ends"][row])
                    if (
                        start != key_offset
                        or stop != key_offset + prefix + global_row - q_offset + 1
                    ):
                        raise ValueError("row range differs from frozen prefix mapping")
                    if not 0 <= start < stop <= columns:
                        raise ValueError("invalid row bounds")
                q_offset += q_len
                key_offset += prefix + q_len
            tensors["logits"] = _snapshot(logits)
            for row in range(rows):
                start = int(tensors["row_starts"][row])
                stop = int(tensors["row_ends"][row])
                tensors["logits"][row, :start].zero_()
                tensors["logits"][row, stop:].zero_()
            manifest = _thaw(self._identity)
            manifest.update(
                complete=False,
                execution_id=manifest["exec_id"],
                call_id=call,
                layer=layer,
                rank=recorder._world_rank,
                op="indexer.topk",
                phase="prefill",
                backend=backend,
                parameters=params,
                query_row_start=query_row_start,
                total_query_rows=total_query_rows,
                query_row_count=rows,
                invalid_logits="zero_filled_snapshot_only",
                finite_values_checked=False,
                nonfinite_policy="valid_domain_bits_preserved_backend_semantics",
                output_contract="raw_relative_indices_minus_one_padding_backend_order",
                tensors=descriptors,
            )
            return _TopKCaptureToken(recorder, manifest, tensors, size, rows, k)
        except Exception:
            tensors.clear()
            if reserved:
                recorder._release(size)
            self.note_skip("invalid_topk_or_snapshot_error", layer)
            return None


class _TopKCaptureToken:
    def __init__(self, recorder, manifest, tensors, size, rows, k):
        self._recorder, self._manifest, self._tensors = recorder, manifest, tensors
        self._size, self._rows, self._k = size, rows, k
        self._lock = threading.Lock()
        self._done = False

    def finish(self, output: torch.Tensor) -> None:
        with self._lock:
            if self._done:
                self._recorder.note_skip("token_already_finished")
                return
            self._done = True
            queued = False
            try:
                with self._recorder._condition:
                    if self._recorder._closed:
                        self._recorder.note_skip("closed")
                        return
                if output.dtype != torch.int32 or output.shape != (self._rows, self._k):
                    raise ValueError("invalid topk output")
                self._manifest["tensors"]["indices"] = _descriptor(output)
                self._tensors["indices"] = _snapshot(output)
                queued = self._recorder._enqueue(
                    self._manifest, self._tensors, self._size
                )
            except Exception:
                self._recorder.note_skip("output_snapshot_error")
            finally:
                if not queued:
                    self._tensors.clear()
                    self._recorder._release(self._size)
                self._tensors = {}
                self._manifest = None

    def cancel(self, reason: str = "kernel_error") -> None:
        with self._lock:
            if self._done:
                return
            self._done = True
            self._tensors.clear()
            self._manifest = None
            self._recorder.note_skip(reason)
            self._recorder._release(self._size)

    def __del__(self):
        if not self._done:
            self.cancel("abandoned_token")


def _read_regular(path, limit, exact=None):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_size > limit
            or (exact is not None and info.st_size != exact)
        ):
            raise ValueError("invalid capture file size/type")
        data = stream.read(limit + 1)
        if len(data) != info.st_size:
            raise ValueError("capture file changed during read")
        return data


def load_capture(path):
    """Load a complete manifest (or its directory), never pickle or source storage.

    Untrusted files are bounded to 1 MiB JSON / 1 GiB aggregate raw tensor bytes;
    tensors are returned CPU-contiguous in logical shape, not original strides.
    """
    path = Path(os.path.abspath(path))
    if path.is_dir():
        path = path / "manifest.json"
    if path.name != "manifest.json" or any(
        p.is_symlink() for p in (path,) + tuple(path.parents)
    ):
        raise ValueError("unsafe manifest path")
    manifest = _json_loads(_read_regular(path, _MAX_JSON))
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != _VERSION
        or manifest.get("complete") is not True
    ):
        raise ValueError("incomplete/unsupported capture")
    descriptors = manifest.get("tensors")
    if not isinstance(descriptors, dict) or not 1 <= len(descriptors) <= 32:
        raise ValueError("invalid tensor descriptors")
    planned = []
    total = 0
    for name, desc in descriptors.items():
        if (
            not _identifier(name)
            or not isinstance(desc, dict)
            or desc.get("file") != name + ".bin"
        ):
            raise ValueError("unsafe tensor filename")
        dtype_name = desc.get("dtype")
        dtype = _DTYPES.get(dtype_name) if isinstance(dtype_name, str) else None
        shape, stride = desc.get("shape"), desc.get("original_stride")
        if (
            dtype is None
            or not isinstance(shape, list)
            or len(shape) > 16
            or not all(_integer(x) for x in shape)
        ):
            raise ValueError("invalid dtype/shape")
        if (
            not isinstance(stride, list)
            or len(stride) != len(shape)
            or not all(_integer(x, 0, 2**63 - 1) for x in stride)
            or not _integer(desc.get("storage_offset"), 0, 2**63 - 1)
        ):
            raise ValueError("invalid source layout")
        # Even an empty tensor must have a representable logical layout.
        if math.prod(max(1, dimension) for dimension in shape) > _MAX_FILE_BYTES:
            raise ValueError("tensor logical layout exceeds loader limit")
        elements = math.prod(shape)
        expected = elements * torch.empty((), dtype=dtype).element_size()
        if (
            not _integer(desc.get("nbytes"), 0, _MAX_FILE_BYTES)
            or desc["nbytes"] != expected
        ):
            raise ValueError("tensor byte size mismatch")
        if (
            desc.get("serialization") != "contiguous_row_major"
            or desc.get("byte_order") != "little"
        ):
            raise ValueError("unsupported tensor encoding")
        checksum = desc.get("sha256")
        if (
            not isinstance(checksum, str)
            or re.fullmatch(r"[0-9a-f]{64}", checksum) is None
        ):
            raise ValueError("invalid checksum")
        total += expected
        if total > _MAX_FILE_BYTES:
            raise ValueError("capture exceeds loader limit")
        planned.append((name, desc, dtype, expected))
    tensors = {}
    for name, desc, dtype, expected in planned:
        data = _read_regular(path.parent / desc["file"], expected, exact=expected)
        if hashlib.sha256(data).hexdigest() != desc["sha256"]:
            raise ValueError("tensor checksum mismatch")
        if not expected:
            tensors[name] = torch.empty(desc["shape"], dtype=dtype)
        else:
            octets = torch.frombuffer(bytearray(data), dtype=torch.uint8)
            itemsize = torch.empty((), dtype=dtype).element_size()
            if sys.byteorder != "little" and itemsize > 1:
                octets = octets.reshape(-1, itemsize).flip(1).contiguous().reshape(-1)
            tensors[name] = octets.view(dtype).reshape(desc["shape"])
    return manifest, tensors
