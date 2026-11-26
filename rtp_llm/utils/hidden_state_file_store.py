"""Mounted-file transport for RTP capture, synchronous on the publisher worker.

A committed <key>.capture is a single immutable little-endian file: magic,
uint32 LE metadata length, capture marker JSON v1, then dense raw
payloads in _FIELDS order (no padding). Magic v2 keeps the same JSON v1
header and wraps just the tensor bytes in one checksummed, content-sized
Zstandard frame. Auxiliary and scale columns retain capture_layer_ids order.
No pickle, cast or re-quantization.

Visibility is per request, not transactional across a batch. hf3fs FuseOps
implements write, file fsync and hardLink, rejects rename flags and does not
register fsyncdir. Link a closed, fsynced file without replacing any name.
This is a publication receipt, NOT a guarantee of directory-entry, replica or
power-loss durability. A killed writer can leave an unlisted .pending-* file.
"""

import argparse
import json
import logging
import math
import os
import struct
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager, nullcontext
from itertools import chain
from pathlib import Path
from threading import Lock
from uuid import uuid4

import torch

from rtp_llm.utils.hidden_state_capture_trace import mark, span
from rtp_llm.utils.hidden_state_file_contract import (
    MAX_CAPTURE_METADATA_BYTES,
    decode_capture_metadata,
    encode_capture_metadata,
    is_store_key,
)

_MAGIC = b"RTPHS\x01"
_ZSTD_MAGIC = b"RTPHS\x02"
# The publisher defaults to a 1 GiB in-flight quota. A larger authorized
# request can still run, but never alongside another staged request.
_STAGE_BUDGET = 1 << 30
_DEFAULT_MAX_SAMPLE_BYTES = 1 << 30
_MAX_ZSTD_WINDOW_LOG = 27
_HEADER = struct.Struct("<6sI")
_FIELDS = ("hidden_states", "input_ids", "last_hidden_states", "hidden_states_scale")
_DTYPES = {str(t).removeprefix("torch."): t for t in (
    torch.bfloat16, torch.int64, torch.float32, torch.float8_e4m3fn,
)}
_SUFFIX = ".capture"


def _filename(key):
    if not is_store_key(key):
        raise ValueError(f"invalid capture key: {key!r}")
    return key + _SUFFIX


def _require_hf3fs(fd):
    # Match the mount of the opened directory, not a pathname prefix. All file
    # operations stay relative to this fd even if the mountpoint is replaced.
    with open(f"/proc/self/fdinfo/{fd}") as info:
        mount_id = next((s.split()[1] for s in info if s.startswith("mnt_id:")), None)
    with open("/proc/self/mountinfo") as mounts:
        for entry in mounts:
            if entry.split(" ", 1)[0] == mount_id:
                fs_type = entry.split(" - ", 1)[1].split()[0]
                if fs_type == "fuse.hf3fs":
                    return
                raise ValueError(f"capture root requires fuse.hf3fs, found {fs_type}")
    raise RuntimeError("cannot identify capture root mount")


class _CaptureRoot:
    def __init__(self, root, *, local_protocol=False, io_workers=2, io_depth=2):
        if type(local_protocol) is not bool:
            raise TypeError("local_protocol must be a boolean")
        if not local_protocol:
            # The publisher constructs FileCaptureStore(root); its local test
            # must opt in independently of the chosen directory name.
            env = "RTP_HIDDEN_STATE_CAPTURE_3FS_LOCAL_PROTOCOL_TEST_ONLY"
            switch = os.environ.get(env, "0")
            if switch not in ("0", "1"):
                raise ValueError(f"{env} must be 0 or 1")
            local_protocol = switch == "1"
        if any(type(n) is not int or n <= 0 for n in (io_workers, io_depth)):
            raise ValueError("io_workers and io_depth must be positive integers")
        self.io_workers = io_workers
        self.io_depth = io_depth
        if sys.byteorder != "little":
            raise RuntimeError("capture files require a little-endian host")
        self.root = Path(root).absolute()
        self.local_protocol = local_protocol
        with self._directory():
            pass  # Fail at initialization rather than admitting work to a bad root.

    @contextmanager
    def _directory(self):
        fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            if not self.local_protocol:
                _require_hf3fs(fd)
            yield fd
        finally:
            os.close(fd)


def _bytes(tensor):
    # numpy need not support BF16/FP8: expose contiguous CPU storage as bytes.
    # Even contiguous unit-size dimensions may have arbitrary strides.
    flat = tensor.as_strided((tensor.numel(),), (1,))
    return memoryview(flat.view(torch.uint8).numpy())


def _write_all(file, view):
    while len(view):
        written = file.write(view)
        if not written:
            raise OSError("short capture write")
        view = view[written:]


class CaptureBatchError(RuntimeError):
    """After drain, __cause__ belongs to failed_key; additional_errors holds the
    remaining original exceptions in request order, not formatted diagnostics.
    """

    def __init__(self, batch_id, keys, count, cause, additional_errors=()):
        self.additional_errors = tuple(additional_errors)
        self.committed_keys = tuple(keys[:count])
        self.uncommitted_keys = tuple(keys[count:])
        self.failed_key = keys[count]
        super().__init__(f"batch {batch_id}: committed {count}/{len(keys)}, "
                         f"first uncommitted key {self.failed_key}: {cause}")


def _discard(directory, temp):
    try:
        os.unlink(temp, dir_fd=directory)
    except OSError:
        # A private alias is not a failed commit. Preserve the original result.
        logging.getLogger(__name__).warning("Could not remove capture staging file %s", temp, exc_info=True)


def _stage(fields, index, key, logical, codec):
    result = {}
    with span("file.stage", [key], logical_bytes=logical, codec=codec):
        for field, values in fields.items():
            tensor = values[index].detach()
            if tensor.device.type != "cpu":
                with span("file.d2h", [key], field=field, logical_bytes=tensor.nbytes, codec=codec):
                    tensor = tensor.to("cpu", non_blocking=False, memory_format=torch.contiguous_format)
            result[field] = tensor.contiguous()
    return result


class FileCaptureStore(_CaptureRoot):
    def __init__(self, root, *, local_protocol=False, io_workers=2, io_depth=2,
                 compression="none", compression_level=1):
        if compression not in ("none", "zstd"):
            raise ValueError("compression must be 'none' or 'zstd'")
        if type(compression_level) is not int or not -131072 <= compression_level <= 22:
            raise ValueError("invalid Zstandard compression_level")
        self._put_lock = Lock()
        self.compression = compression
        self.compression_level = compression_level
        super().__init__(root, local_protocol=local_protocol, io_workers=io_workers, io_depth=io_depth)

    def _prepare(self, directory, key, marker, tensors, logical):
        """CPU-only pool task. Return ownership of a closed private file, never a key."""
        codec = self.compression
        keys = [key]
        temp = ".pending-" + uuid4().hex
        fd = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600, dir_fd=directory)
        try:
            with os.fdopen(fd, "wb") as out:
                prefix = _HEADER.pack(_MAGIC if codec == "none" else _ZSTD_MAGIC, len(marker)) + marker
                _write_all(out, prefix)
                if codec == "zstd":
                    import zstandard

                    params = zstandard.ZstdCompressionParameters.from_level(
                        self.compression_level, source_size=logical, write_checksum=True, write_content_size=True
                    )
                    if params.window_log > _MAX_ZSTD_WINDOW_LOG:
                        params = zstandard.ZstdCompressionParameters.from_level(
                            self.compression_level, source_size=logical, window_log=_MAX_ZSTD_WINDOW_LOG,
                            write_checksum=True, write_content_size=True,
                        )
                    compressor = zstandard.ZstdCompressor(compression_params=params)
                    stream = compressor.stream_writer(out, size=logical, closefd=False, write_return_read=True)
                else:
                    stream = nullcontext(out)
                with span("file.encode_write", keys, codec=codec, logical_bytes=logical):
                    with stream as payload:
                        for field, tensor in tensors.items():
                            with span("file.payload_write", keys, field=field, codec=codec, logical_bytes=tensor.nbytes):
                                _write_all(payload, _bytes(tensor))
                expected = len(prefix) + (stream.tell() if codec == "zstd" else logical)
                with span("file.fsync", keys, codec=codec, file_bytes=expected):
                    out.flush()
                    os.fsync(out.fileno())
                if os.fstat(out.fileno()).st_size != expected:
                    raise OSError("capture file size mismatch after fsync")
            return temp, expected
        except BaseException:
            _discard(directory, temp)
            raise

    def _commit(self, directory, key, prepared, logical):
        temp, size = prepared
        try:
            os.link(temp, key + _SUFFIX, src_dir_fd=directory, dst_dir_fd=directory)
        finally:
            _discard(directory, temp)
        try:
            mark("file.committed", [key], codec=self.compression, file_bytes=size, logical_bytes=logical)
        except Exception:
            # The link already succeeded. A trace failure cannot turn this key
            # into the first uncommitted key in a partial batch receipt.
            logging.getLogger(__name__).warning("Could not trace committed capture %s", key, exc_info=True)

    def put_batch(self, batch_id, request_ids, hidden_states, input_ids,
                  last_hidden_states, *, capture_metadata, hidden_states_scale=None):
        """Synchronous positional receipts. Multi-request I/O failures report committed_keys."""
        keys = list(request_ids)
        for key in keys:
            _filename(key)
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("put_batch requires nonempty unique request keys")
        fields = dict(hidden_states=hidden_states, input_ids=input_ids, last_hidden_states=last_hidden_states)
        if hidden_states_scale is not None:
            fields["hidden_states_scale"] = hidden_states_scale
        if any(len(values) != len(keys) for values in fields.values()):
            raise ValueError("put_batch requires one tensor per field per request")

        receipts, markers, sizes = [], [], []
        for i in range(len(keys)):
            receipt = {"shapes": {k: v[i].shape for k, v in fields.items()},
                       "dtypes": {k: v[i].dtype for k, v in fields.items()}}
            markers.append(encode_capture_metadata(capture_metadata, receipt))
            receipts.append(receipt)
            sizes.append(sum(v[i].nbytes for v in fields.values()))

        # C++ has one publisher worker. Keep the same staging bound if other
        # Python callers share this store instead of multiplying pools/budgets.
        with self._put_lock, self._directory() as directory:
            committed = 0
            pending = deque()
            reserved = 0
            admission_error = None

            def publish_oldest():
                nonlocal committed, reserved
                future = pending[0]
                error = future.exception()  # Keep ownership in pending until the worker is done.
                pending.popleft()
                if error is not None:
                    raise error
                self._commit(directory, keys[committed], future.result(), sizes[committed])
                reserved -= sizes[committed]
                committed += 1

            serial = len(keys) == 1 or self.io_depth == 1
            try:
                with (nullcontext(None) if serial else ThreadPoolExecutor(
                    max_workers=min(len(keys), self.io_depth, self.io_workers)
                )) as pool:
                    for i, key in enumerate(keys):
                        # Reserve the slot AND bytes before allocating D2H memory.
                        while pending and (len(pending) >= self.io_depth or reserved + sizes[i] > _STAGE_BUDGET):
                            publish_oldest()
                        try:
                            cpu = _stage(fields, i, key, sizes[i], self.compression)
                            if pool is not None:
                                pending.append(pool.submit(self._prepare, directory, key, markers[i], cpu, sizes[i]))
                                reserved += sizes[i]
                            else:
                                self._commit(directory, key, self._prepare(directory, key, markers[i], cpu, sizes[i]), sizes[i])
                                committed += 1
                            del cpu
                        except BaseException as exc:
                            # Finish the admitted prefix before reporting this later failure.
                            admission_error = exc
                            break
                    while pending:
                        publish_oldest()
                    if admission_error is not None:
                        primary, admission_error = admission_error, None
                        raise primary
            except BaseException as cause:
                # Executor exit has joined every worker. The primary failure was
                # removed from the ledger; report every remaining failure in key order.
                others = []
                for future in pending:
                    if future.cancelled():
                        continue  # Never ran, so owns neither a file nor a worker exception.
                    error = future.exception()
                    if error is None:
                        _discard(directory, future.result()[0])
                    else:
                        others.append(error)
                if admission_error is not None:
                    others.append(admission_error)
                if len(keys) == 1 or committed == len(keys):
                    raise
                raise CaptureBatchError(batch_id, keys, committed, cause, others) from cause
        return receipts


def _exact(src, count):
    data = src.read(count)  # BufferedReader retries short reads.
    if len(data) != count:
        raise ValueError("truncated capture file")
    return data


class _FrameInput:
    """Give read_to_iter a strict single-frame boundary despite its input read-ahead."""

    def __init__(self, src, encoded):
        self.src = src
        self.remaining = encoded

    def read(self, size):
        # read_to_iter stops at a completed frame *or* an empty read. Only the
        # former is success. Keep the last byte separate so a read containing
        # the frame end can never hide a trailing byte in decoder read-ahead.
        if self.remaining == 0:
            raise ValueError("truncated Zstandard frame")
        data = _exact(self.src, min(size, max(1, self.remaining - 1)))
        self.remaining -= len(data)
        return data


def _zstd_blocks(src, logical, encoded):
    import zstandard

    try:
        start = src.tell()
        header = src.read(18)  # Maximum Zstandard frame header size; parsed by the library.
        params = zstandard.get_frame_parameters(header)
        if (not header.startswith(zstandard.FRAME_HEADER) or params.content_size != logical or
                not params.has_checksum or params.dict_id or params.window_size > 1 << _MAX_ZSTD_WINDOW_LOG
                or logical > encoded * 32768):
            raise ValueError("invalid Zstandard frame size, checksum flag, dictionary or window")
        src.seek(start)
        source = _FrameInput(src, encoded)
        produced = 0
        for chunk in zstandard.ZstdDecompressor().read_to_iter(source, write_size=min(logical, 128 << 10)):
            produced += len(chunk)
            if produced > logical:
                raise ValueError("Zstandard payload exceeds tensor byte count")
            yield memoryview(chunk)
        if source.remaining or src.read(1):
            raise ValueError("trailing bytes after Zstandard frame")
        if produced != logical:
            raise ValueError("Zstandard payload length differs from tensor byte count")
    except zstandard.ZstdError as exc:
        raise ValueError(f"invalid Zstandard capture payload: {exc}") from exc


class FileCaptureReader(_CaptureRoot):
    """Limit decoded tensor bytes per sample (not compressed size or process RSS).

    Larger authorized captures require an explicit reader limit. Raising it
    does not raise the multi-sample prefetch budget: an oversized sample is
    decoded alone, just as it was staged alone by the writer.
    """

    def __init__(self, root, *, local_protocol=False, io_workers=2, io_depth=2,
                 max_sample_bytes=_DEFAULT_MAX_SAMPLE_BYTES):
        if type(max_sample_bytes) is not int or not 0 < max_sample_bytes <= sys.maxsize:
            raise ValueError("max_sample_bytes must be a positive host-sized integer")
        self.max_sample_bytes = max_sample_bytes
        super().__init__(root, local_protocol=local_protocol, io_workers=io_workers, io_depth=io_depth)

    def keys(self):
        with self._directory() as directory:
            return sorted(n[:-len(_SUFFIX)] for n in os.listdir(directory)
                          if n.endswith(_SUFFIX) and is_store_key(n[:-len(_SUFFIX)]))

    def _open(self, directory, key):
        fd = os.open(_filename(key), os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        src = os.fdopen(fd, "rb")
        try:
            magic, length = _HEADER.unpack(_exact(src, _HEADER.size))
            if magic not in (_MAGIC, _ZSTD_MAGIC) or not 0 < length <= MAX_CAPTURE_METADATA_BYTES:
                raise ValueError("invalid capture header")
            metadata = decode_capture_metadata(_exact(src, length))
            shapes, dtypes = metadata["tensor_shapes"], metadata["tensor_dtypes"]
            if set(shapes) - set(_FIELDS):
                raise ValueError("unsupported capture tensor")
            logical = sum(math.prod(shape) * _DTYPES[dtypes[k]].itemsize for k, shape in shapes.items())
            size = os.fstat(fd).st_size
            encoded = size - src.tell()
            if logical > self.max_sample_bytes:
                raise ValueError(f"{key}: {logical} decoded tensor bytes exceed max_sample_bytes="
                                 f"{self.max_sample_bytes}")
            if encoded <= 0 or (magic == _MAGIC and encoded != logical):
                raise ValueError("capture file length does not match tensor metadata")
            return src, metadata, logical, size, "none" if magic == _MAGIC else "zstd"
        except BaseException:
            src.close()
            raise

    @staticmethod
    def _decode(key, src, metadata, logical, size, codec):
        with src, span("file.decode_read", [key], logical_bytes=logical, file_bytes=size, codec=codec):
            blocks = None
            if codec == "zstd":
                blocks = _zstd_blocks(src, logical, size - src.tell())
                block = next(blocks, None)  # Validate frame header before allocating tensors.
            tensors = {}
            for field in _FIELDS:
                if field not in metadata["tensor_shapes"]:
                    continue
                tensor = torch.empty(metadata["tensor_shapes"][field], device="cpu",
                                     dtype=_DTYPES[metadata["tensor_dtypes"][field]])
                view = _bytes(tensor)
                while view:
                    if blocks is None:
                        n = src.readinto(view)
                        if not n:
                            raise ValueError(f"truncated capture tensor: {field}")
                    else:
                        if block is None:
                            raise ValueError(f"truncated compressed capture tensor: {field}")
                        n = min(len(view), len(block))
                        view[:n] = block[:n]
                        block = block[n:]
                        if not block:
                            block = next(blocks, None)
                    view = view[n:]
                tensors[field] = tensor
            if blocks is not None and block is not None:
                raise ValueError("excess decoded capture bytes")
            return dict(tensors=tensors, metadata=metadata, logical_bytes=logical, file_bytes=size, codec=codec)

    def read(self, key):
        """Read and verify one sample inline, without an executor."""
        with self._directory() as directory:
            return self._decode(key, *self._open(directory, key))

    def iter_read(self, keys):
        """Yield (key, sample) in order. Use prefetch() or close() on early exit."""
        keys = iter(keys)
        sentinel = object()
        first = next(keys, sentinel)
        if first is sentinel:
            return
        if self.io_depth == 1:
            for key in chain((first,), keys):
                yield key, self.read(key)
            return
        second = next(keys, sentinel)
        if second is sentinel:
            yield first, self.read(first)
            return
        pending = deque()
        reserved = 0
        with self._directory() as directory, ThreadPoolExecutor(max_workers=min(self.io_workers, self.io_depth)) as pool:
            for key in chain((first, second), keys):
                try:
                    plan = self._open(directory, key)
                except Exception:
                    # Preserve ordered delivery of the already admitted prefix.
                    while pending:
                        old_key, _, future = pending.popleft()
                        yield old_key, future.result()
                    raise
                src, _, logical, _, _ = plan
                try:
                    while pending and reserved + logical > _STAGE_BUDGET:
                        old_key, count, future = pending.popleft()
                        yield old_key, future.result()
                        reserved -= count
                        del future
                    pending.append((key, logical, pool.submit(self._decode, key, *plan)))
                    reserved += logical
                except BaseException:
                    src.close()  # Not submitted: this iterator still owns the fd.
                    raise
                if len(pending) == self.io_depth:
                    old_key, count, future = pending.popleft()
                    yield old_key, future.result()
                    reserved -= count
                    del future
            while pending:
                key, _, future = pending.popleft()
                yield key, future.result()
                del future

    def prefetch(self, keys):
        """Context-managed ordered iterator; exit joins all futures even after break."""
        return closing(self.iter_read(keys))


def main():
    parser = argparse.ArgumentParser(description="List capture keys, or verify/read samples in argument order")
    parser.add_argument("root", help="existing hf3fs directory")
    parser.add_argument("keys", nargs="*")
    parser.add_argument("--all", action="store_true", help="read all committed keys instead of listing them")
    parser.add_argument("--io-workers", type=int, default=2)
    parser.add_argument("--io-depth", type=int, default=2)
    parser.add_argument("--max-sample-bytes", type=int, default=_DEFAULT_MAX_SAMPLE_BYTES,
                        help="decoded tensor byte limit per sample (default 1 GiB); raise for authorized larger samples")
    parser.add_argument("--local-protocol", action="store_true", help="test only: allow a non-hf3fs directory")
    args = parser.parse_intermixed_args()
    reader = FileCaptureReader(args.root, local_protocol=args.local_protocol, io_workers=args.io_workers,
                               io_depth=args.io_depth, max_sample_bytes=args.max_sample_bytes)
    if args.all and args.keys:
        parser.error("--all and explicit keys are mutually exclusive")
    if not args.keys and not args.all:
        print(json.dumps(reader.keys()))
        return
    with reader.prefetch(reader.keys() if args.all else args.keys) as samples:
        for key, sample in samples:
            del sample["tensors"]
            print(json.dumps(dict(key=key, **sample)))


if __name__ == "__main__":
    main()
