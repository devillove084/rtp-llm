"""Standalone CPU file-protocol tests (no TorchSpec checkout or reference fixture).

  python3 -m rtp_llm.utils.test.hidden_state_file_store_test -v
  python3 -m rtp_llm.utils.test.hidden_state_file_store_test write EMPTY_DIRECTORY [none|zstd]
  python3 -m rtp_llm.utils.test.hidden_state_file_store_test read DIRECTORY

Standalone write/read use only the RTP runtime. The reader takes no fixture parameters or receipts.
"""

import errno
import hashlib
from contextlib import nullcontext
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from threading import Barrier, Event, Lock
from pathlib import Path
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class _BlockLegacyImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in ("torchspec", "mooncake"):
            raise AssertionError(f"standalone capture attempted to import {fullname}")
        return None


if __name__ == "__main__":
    if any(name.partition(".")[0] in ("torchspec", "mooncake") for name in sys.modules):
        raise AssertionError("standalone capture interpreter already loaded a legacy dependency")
    sys.meta_path.insert(0, _BlockLegacyImports())

import torch

from rtp_llm.utils.hidden_state_file_contract import (
    decode_capture_metadata, is_store_key, make_store_key, quantize_aux_hidden_states,
)
from rtp_llm.utils.hidden_state_file_store import CaptureBatchError, FileCaptureReader, FileCaptureStore


_LAYOUTS = ((19, 2, 31, 7, 12), (17, 1, 9))
_TOKENS, _WIDTH = 4, 13


def _fixture(layers, fp8, seed=120, quantize=quantize_aux_hidden_states):
    rng = torch.Generator().manual_seed(seed)
    aux = torch.randn(_TOKENS, len(layers), _WIDTH, generator=rng)
    aux *= torch.tensor(layers)[None, :, None]
    aux[0, 1].zero_()  # Exercise the per-token/layer scale floor.
    aux = aux.flatten(1).bfloat16()
    fields = {
        "hidden_states": aux,
        "input_ids": torch.arange(_TOKENS, dtype=torch.int64) + 65536,
        "last_hidden_states": torch.randn(_WIDTH, _TOKENS, generator=rng).bfloat16().T,
    }
    if fp8:
        fields["hidden_states"], fields["hidden_states_scale"] = quantize(aux, len(layers))
    return aux, fields


def _cases():
    for layers in _LAYOUTS:
        for fp8 in (False, True):
            yield f"rtp_{'fp8' if fp8 else 'bf16'}_{len(layers)}", layers, fp8


def _put(store, keys, layers, fields):
    return store.put_batch(batch_id="test-batch", request_ids=keys,
                           capture_metadata={"capture_layer_ids": list(layers)},
                           **{k: [v] * len(keys) for k, v in fields.items()})


def _raw(tensor):
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def _inventory(fields):
    return {k: {"bytes": v.nbytes, "sha256": hashlib.sha256(_raw(v)).hexdigest()}
            for k, v in fields.items()}


def _write_directory(root, codec):
    store = FileCaptureStore(root, local_protocol=True, compression=codec, compression_level=1)
    for key, layers, fp8 in _cases():
        _, fields = _fixture(layers, fp8)
        _put(store, [key], layers, fields)


def _read_directory(root):
    reader = FileCaptureReader(root, local_protocol=True)
    manifest = {}
    for key in reader.keys():
        sample = reader.read(key)
        manifest[key] = {"fields": _inventory(sample.pop("tensors")), **sample}
    return manifest


def _child(mode, root, *args, check=True):
    # Bazel's launcher can add runfiles to sys.path without exporting them.
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(os.path.abspath(p) for p in sys.path)}
    env.pop("RTP_TEST_TORCHSPEC_ARCHIVE", None)
    return subprocess.run([sys.executable, os.path.abspath(__file__), mode, str(root), *args],
                          env=env, cwd=root, capture_output=True, text=True, timeout=60, check=check)


class HiddenStateFileStoreTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.store = FileCaptureStore(self.root, local_protocol=True)
        self.reader = FileCaptureReader(self.root, local_protocol=True)

    def test_reference_payload_and_byte_inventory(self):
        for key, layers, fp8 in _cases():
            with self.subTest(key=key):
                aux, fields = _fixture(layers, fp8)
                receipts = _put(self.store, [key], layers, fields)
                self.assertEqual(receipts, [{"shapes": {k: v.shape for k, v in fields.items()},
                                             "dtypes": {k: v.dtype for k, v in fields.items()}}])
                sample = self.reader.read(key)
                metadata = {"version": 1, "capture_layer_ids": list(layers),
                            "tensor_shapes": {k: list(v.shape) for k, v in fields.items()},
                            "tensor_dtypes": {k: str(v.dtype).removeprefix("torch.") for k, v in fields.items()}}
                self.assertEqual(sample["metadata"], metadata)
                tensors = sample["tensors"]
                self.assertEqual(set(tensors), set(fields))
                for k, expected in fields.items():
                    self.assertEqual(tensors[k].device.type, "cpu")
                    self.assertEqual(tensors[k].dtype, expected.dtype)
                    self.assertEqual(tensors[k].shape, expected.shape)
                    self.assertEqual(_raw(tensors[k]), _raw(expected), k)

                # Verify the actual wire order and byte counts independently of read().
                data = (self.root / (key + ".capture")).read_bytes()
                magic, marker_bytes = struct.unpack_from("<6sI", data)
                self.assertEqual(magic, b"RTPHS\x01")
                offset = 10 + marker_bytes
                self.assertEqual(json.loads(data[10:offset]), metadata)
                self.assertEqual(data[offset:], b"".join(_raw(v) for v in fields.values()))
                sizes = {"input_ids": _TOKENS * 8, "last_hidden_states": _TOKENS * _WIDTH * 2,
                         "hidden_states": _TOKENS * _WIDTH * len(layers) * (1 if fp8 else 2)}
                if fp8:
                    sizes["hidden_states_scale"] = _TOKENS * len(layers) * 4
                    codes = tensors["hidden_states"].float().reshape(_TOKENS, len(layers), _WIDTH)
                    restored = codes * tensors["hidden_states_scale"].unsqueeze(-1)
                    torch.testing.assert_close(restored.flatten(1), aux.float(), rtol=0.063, atol=1e-5)
                self.assertEqual({k: v.nbytes for k, v in tensors.items()}, sizes)
                self.assertEqual(sample["logical_bytes"], sum(sizes.values()))
                self.assertEqual(sample["file_bytes"], offset + sum(sizes.values()))
                self.assertEqual(sample["file_bytes"], len(data))

    def test_pipeline_lossless_codec_and_dtype(self):
        layers = _LAYOUTS[0]
        import zstandard

        for fp8 in (False, True):
            _, reference = _fixture(layers, fp8)
            # Give the compressor a useful frame, not just tiny random fields.
            fields = {k: v.repeat((256,) + (1,) * (v.ndim - 1)) for k, v in reference.items()}
            sizes = {}
            for codec in ("none", "zstd"):
                with self.subTest(codec=codec, fp8=fp8):
                    key = f"rtp_{codec}_{int(fp8)}"
                    store = FileCaptureStore(self.root, local_protocol=True, io_workers=2, io_depth=2,
                                             compression=codec, compression_level=1)
                    _put(store, [key], layers, fields)
                    reader = FileCaptureReader(self.root, local_protocol=True, io_workers=2, io_depth=2)
                    sample = reader.read(key)
                    self.assertEqual(_inventory(sample["tensors"]), _inventory(fields))
                    self.assertEqual(sample["metadata"]["capture_layer_ids"], list(layers))
                    self.assertEqual(sample["logical_bytes"], sum(v.nbytes for v in fields.values()))
                    self.assertEqual(sample["codec"], codec)
                    path = self.root / (key + ".capture")
                    wire = path.read_bytes()
                    sizes[codec] = len(wire)
                    self.assertEqual(sample["file_bytes"], len(wire))
                    magic, meta_size = struct.unpack_from("<6sI", wire)
                    self.assertEqual(magic, b"RTPHS\x02" if codec == "zstd" else b"RTPHS\x01")
                    payload = wire[10 + meta_size:]
                    if codec == "zstd":
                        payload = zstandard.ZstdDecompressor().decompress(payload)
                        # An interrupted copy can have every tensor byte but lack the frame checksum.
                        path.write_bytes(wire[:-1])
                        with self.assertRaises(ValueError):
                            reader.read(key)
                    self.assertEqual(payload, b"".join(_raw(v) for v in fields.values()))
            self.assertLess(sizes["zstd"], sizes["none"])

    def test_write_window_and_worker_cap(self):
        layers = _LAYOUTS[1]
        _, fields = _fixture(layers, True)
        submit = ThreadPoolExecutor.submit
        for workers, depth in ((2, 3), (3, 2)):
            with self.subTest(workers=workers, depth=depth):
                store = FileCaptureStore(self.root, local_protocol=True, io_workers=workers, io_depth=depth)
                keys = [f"rtp_w{workers}_{i}" for i in range(8)]
                held, release, overflow = Event(), Event(), Event()
                lock = Lock()
                active = submitted = 0

                def track_submit(pool, *args, **kwargs):
                    nonlocal submitted
                    with lock:
                        submitted += 1
                        if not release.is_set() and submitted > depth:
                            overflow.set()
                    return submit(pool, *args, **kwargs)

                prepare = store._prepare

                def hold_writes(*args, **kwargs):
                    nonlocal active
                    with lock:
                        active += 1
                        if active == min(workers, depth):
                            held.set()
                        elif active > min(workers, depth):
                            overflow.set()
                    try:
                        if not release.wait(5):
                            raise AssertionError("writer gate was not released")
                        return prepare(*args, **kwargs)
                    finally:
                        with lock:
                            active -= 1

                with ThreadPoolExecutor(1) as caller, patch.object(ThreadPoolExecutor, "submit", track_submit), \
                     patch.object(store, "_prepare", side_effect=hold_writes):
                    # The coordinator is not one of the store's submitted writes.
                    task = submit(caller, _put, store, keys, layers, fields)
                    try:
                        self.assertTrue(held.wait(5), "expected two overlapping writes")
                        self.assertFalse(overflow.wait(0.15), "exceeded worker or outstanding-work bound")
                    finally:
                        release.set()
                    self.assertEqual(len(task.result(timeout=10)), len(keys))
                self.assertFalse(overflow.is_set())

    def test_put_drains_prepares_without_publishing_failed_suffix(self):
        layers = _LAYOUTS[1]
        _, fields = _fixture(layers, False)
        store = FileCaptureStore(self.root, local_protocol=True, io_workers=2, io_depth=2,
                                 compression="zstd", compression_level=1)
        rendezvous, failed, release, finished = Barrier(2), Event(), Event(), Event()
        prepare = store._prepare
        keys = ["rtp_batch_0", "rtp_batch_1", "rtp_batch_2"]

        def gate(directory, key, *args):
            if key == keys[0]:
                return prepare(directory, key, *args)
            rendezvous.wait(5)
            if key == keys[1]:
                failed.set()
                raise OSError("injected prepare failure")
            if not release.wait(5):
                raise AssertionError("timed out waiting for writer release")
            staged = prepare(directory, key, *args)
            finished.set()
            return staged

        with patch.object(store, "_prepare", side_effect=gate), ThreadPoolExecutor(1) as caller:
            task = caller.submit(_put, store, keys, layers, fields)
            try:
                self.assertTrue(failed.wait(5))
                with self.assertRaises(FutureTimeout):
                    task.result(timeout=0.15)
            finally:
                release.set()
            with self.assertRaises(CaptureBatchError) as failure:
                task.result(timeout=5)
        self.assertIsInstance(failure.exception.__cause__, OSError)
        self.assertEqual(failure.exception.failed_key, keys[1])
        self.assertEqual(failure.exception.committed_keys, tuple(keys[:1]))
        self.assertEqual(failure.exception.uncommitted_keys, tuple(keys[1:]))
        self.assertTrue(finished.is_set())
        self.assertEqual(self.reader.keys(), keys[:1])
        self.assertEqual(_inventory(self.reader.read(keys[0])["tensors"]), _inventory(fields))
        self.assertFalse(list(self.root.glob(".pending-*")))

    def test_failed_publication_preserves_later_fatal_future(self):
        layers = _LAYOUTS[0]
        _, original = _fixture(layers, False)
        _, replacement = _fixture(layers, False, seed=121)
        for duplicate in (True, False):
            with self.subTest(duplicate=duplicate), tempfile.TemporaryDirectory() as root:
                store = FileCaptureStore(root, local_protocol=True, io_workers=2, io_depth=2)
                reader = FileCaptureReader(root, local_protocol=True)
                first, later = "rtp_fatal_0", "rtp_fatal_1"
                if duplicate:
                    _put(store, [first], layers, original)
                failed = Event()
                prepare, link = store._prepare, os.link
                fatal = torch.distributed.DistError("injected worker fatal")

                def publish(*args, **kwargs):
                    try:
                        if duplicate:
                            return link(*args, **kwargs)  # Real attempt to overwrite first.
                        raise OSError(errno.ENOSPC, "injected publish error")
                    except OSError:
                        failed.set()
                        raise

                def stage(directory, key, *args):
                    if key == later:
                        if not failed.wait(5):
                            raise AssertionError("head did not reach publication")
                        raise fatal
                    return prepare(directory, key, *args)

                with patch.object(store, "_prepare", side_effect=stage), patch("os.link", side_effect=publish):
                    with self.assertRaises(CaptureBatchError) as raised:
                        _put(store, [first, later], layers, replacement)
                error = raised.exception
                self.assertIsInstance(error.__cause__, FileExistsError if duplicate else OSError)
                self.assertEqual(error.__cause__.errno, errno.EEXIST if duplicate else errno.ENOSPC)
                self.assertEqual(error.failed_key, first)
                self.assertEqual(error.committed_keys, ())
                self.assertEqual(error.uncommitted_keys, (first, later))
                self.assertEqual(tuple(error.additional_errors), (fatal,))
                self.assertEqual(reader.keys(), [first] if duplicate else [])
                if duplicate:
                    self.assertEqual(_inventory(reader.read(first)["tensors"]), _inventory(original))
                self.assertFalse(list(Path(root).glob(".pending-*")))

    def test_reader_limits_logical_bytes_per_sample_before_allocation(self):
        for invalid in (0, -1, True, sys.maxsize + 1):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "max_sample_bytes"):
                FileCaptureReader(self.root, local_protocol=True, max_sample_bytes=invalid)
        layers = _LAYOUTS[0]
        _, fields = _fixture(layers, True)
        fields = {k: v.repeat((64,) + (1,) * (v.ndim - 1)) for k, v in fields.items()}
        logical = sum(t.nbytes for t in fields.values())
        for codec in ("none", "zstd"):
            with self.subTest(codec=codec):
                key = f"rtp_{codec}_0"
                store = FileCaptureStore(self.root, local_protocol=True, compression=codec)
                _put(store, [key], layers, fields)
                reader = FileCaptureReader(self.root, local_protocol=True, max_sample_bytes=logical)
                sample = reader.read(key)
                self.assertEqual(_inventory(sample["tensors"]), _inventory(fields))
                self.assertEqual(sample["logical_bytes"], logical)
                if codec == "zstd":
                    self.assertLess(sample["file_bytes"], logical - 1)
                limited = FileCaptureReader(self.root, local_protocol=True, max_sample_bytes=logical - 1)
                with patch.object(torch, "empty", side_effect=AssertionError("oversize sample allocated")):
                    with self.assertRaisesRegex(ValueError, "max_sample_bytes"):
                        limited.read(key)
                    with limited.prefetch([key, key]) as stream, self.assertRaisesRegex(ValueError, "max_sample_bytes"):
                        next(stream)

    def test_reader_stops_and_drains_prefetch(self):
        layers = _LAYOUTS[1]
        _, fields = _fixture(layers, True)
        keys = [f"rtp_read_{i}" for i in range(8)]
        _put(self.store, keys, layers, fields)
        for interrupt in (False, True):
            with self.subTest(interrupt=interrupt):
                reader = FileCaptureReader(self.root, local_protocol=True, io_workers=2, io_depth=2)
                entered, release, finished, stopping = Event(), Event(), Event(), Event()
                decode = reader._decode
                visited = []

                def gated(key, src, *plan):
                    visited.append(key)
                    if key == keys[0]:
                        if not entered.wait(5):
                            raise AssertionError("second read did not overlap first")
                        if interrupt:
                            src.close()  # The injected worker owns this handle.
                            raise KeyboardInterrupt("interrupted read")
                    if key == keys[1]:
                        entered.set()
                        if not release.wait(5):
                            raise AssertionError("reader gate was not released")
                    result = decode(key, src, *plan)
                    if key == keys[1]:
                        finished.set()
                    return result

                def consume():
                    expected = self.assertRaisesRegex(KeyboardInterrupt, "interrupted read") if interrupt else nullcontext()
                    with expected, reader.prefetch(keys) as stream:
                        if interrupt:
                            stopping.set()
                        key, sample = next(stream)
                        self.assertEqual(key, keys[0])
                        self.assertEqual(_inventory(sample["tensors"]), _inventory(fields))
                        stopping.set()  # Leave without consuming the rest of the stream.

                with patch.object(reader, "_decode", side_effect=gated), ThreadPoolExecutor(1) as caller:
                    task = caller.submit(consume)
                    try:
                        self.assertTrue(entered.wait(5))
                        self.assertTrue(stopping.wait(5))
                        with self.assertRaises(FutureTimeout):
                            task.result(timeout=0.15)
                    finally:
                        release.set()
                    task.result(timeout=5)
                self.assertTrue(finished.is_set())
                self.assertCountEqual(keys[:2], visited)

    def test_fresh_reader_needs_only_committed_files(self):
        for codec in ("none", "zstd"):
            with self.subTest(codec=codec):
                root = self.root / codec
                root.mkdir()
                with tempfile.TemporaryDirectory() as source:
                    _child("write", Path(source), codec)
                    for key, _, _ in _cases():
                        shutil.copyfile(Path(source, key + ".capture"), root / (key + ".capture"))
                manifest = json.loads(_child("read", root).stdout)
                self.assertEqual(set(manifest), {key for key, _, _ in _cases()})
                for key, layers, fp8 in _cases():
                    _, fields = _fixture(layers, fp8)
                    self.assertEqual(manifest[key]["fields"], _inventory(fields))
                    self.assertEqual(manifest[key]["metadata"]["capture_layer_ids"], list(layers))
                    self.assertEqual(manifest[key]["codec"], codec)

    def test_v1_key_and_marker_contract(self):
        key = "rtp_A9_9223372036854775807"
        self.assertEqual(make_store_key("A9", 2**63 - 1), key)
        self.assertTrue(is_store_key(key))
        for invalid in ("rtp_A9_01", "rtp_A9_9223372036854775808", "rtp_A/9_1"):
            self.assertFalse(is_store_key(invalid), invalid)

        marker = (b'{"version":1,"capture_layer_ids":[7,2],"tensor_shapes":{"hidden_states":[1,4],'
                  b'"input_ids":[1],"last_hidden_states":[1,2]},"tensor_dtypes":{"hidden_states":"bfloat16",'
                  b'"input_ids":"int64","last_hidden_states":"bfloat16"}}')
        self.assertEqual(decode_capture_metadata(marker), json.loads(marker))
        for invalid in (marker.replace(b'"version":1', b'"version":1,"version":1'),
                        marker.replace(b'[7,2]', b'[7,7]'), marker.replace(b'[1,4]', b'[1,2]')):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                decode_capture_metadata(invalid)

    def test_e4m3fn_known_codes_and_scales(self):
        # Exact E4M3FN codes; each token/layer has its own max(abs(x))/448 scale.
        aux = torch.tensor([[-448, -224, 0, 448, 1, -2, 3.5, 7],
                            [224, 112, 56, 0, -14, 7, 0, -1.75]], dtype=torch.bfloat16)
        raw, scale = quantize_aux_hidden_states(aux, 2)
        self.assertEqual(raw.shape, aux.shape)
        self.assertEqual(raw.dtype, torch.float8_e4m3fn)
        self.assertEqual(_raw(raw), bytes.fromhex("fe f6 00 7e 68 f0 76 7e 7e 76 6e 00 fe 76 00 e6"))
        torch.testing.assert_close(scale, torch.tensor([[1., 1/64], [0.5, 1/32]]), atol=0, rtol=0)

    def test_interrupted_before_publication(self):
        child = _child("interrupt", self.root, check=False)
        self.assertEqual(child.returncode, 73, child.stderr)
        self.assertTrue(any(p.stat().st_size > 0 for p in self.root.glob(".pending-*")))
        self.assertEqual(self.reader.keys(), [])
        with self.assertRaises(FileNotFoundError):
            self.reader.read("rtp_crash_1")
        _, fields = _fixture(_LAYOUTS[0], False)
        _put(self.store, ["rtp_crash_1"], _LAYOUTS[0], fields)
        self.assertEqual(self.reader.keys(), ["rtp_crash_1"])
        self.assertEqual(_inventory(self.reader.read("rtp_crash_1")["tensors"]), _inventory(fields))

    def test_duplicate_never_overwrites_first_publication(self):
        layers = _LAYOUTS[1]
        _, original = _fixture(layers, True)
        _, different = _fixture(layers, True, seed=121)
        with self.assertRaises(ValueError):
            _put(self.store, ["rtp_dup_1", "rtp_dup_1"], layers, original)
        self.assertEqual(self.reader.keys(), [])
        _put(self.store, ["rtp_dup_1"], layers, original)
        path = self.root / "rtp_dup_1.capture"
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            _put(self.store, ["rtp_dup_1"], layers, different)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(_inventory(self.reader.read("rtp_dup_1")["tensors"]), _inventory(original))
        self.assertEqual(list(self.root.iterdir()), [path])


if __name__ == "__main__":
    if len(sys.argv) in (3, 4) and sys.argv[1] == "write":
        _write_directory(sys.argv[2], sys.argv[3] if len(sys.argv) == 4 else "none")
    elif len(sys.argv) == 3 and sys.argv[1] == "read":
        print(json.dumps(_read_directory(sys.argv[2]), sort_keys=True))
    elif len(sys.argv) == 3 and sys.argv[1] == "interrupt":
        store = FileCaptureStore(sys.argv[2], local_protocol=True)
        _, fields = _fixture(_LAYOUTS[0], True)
        # Simulate process death after fsync/close, before the atomic hard link.
        with patch("os.link", side_effect=lambda *args, **kwargs: os._exit(73)):
            _put(store, ["rtp_crash_1"], _LAYOUTS[0], fields)
    else:
        unittest.main()
