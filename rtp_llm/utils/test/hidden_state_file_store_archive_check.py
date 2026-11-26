"""Opt-in compatibility check against RTP_TEST_TORCHSPEC_ARCHIVE, never a runtime dependency."""

import importlib.util
import json
import os
import shutil
import struct
import tempfile
import unittest
import zipfile
import zipimport
from functools import lru_cache
from pathlib import Path

from rtp_llm.utils.hidden_state_file_contract import quantize_aux_hidden_states
from rtp_llm.utils.test.hidden_state_file_store_test import (
    _LAYOUTS, _cases, _child, _fixture, _inventory, _raw,
)


@lru_cache(maxsize=1)
def _reference():
    # Load only these two standalone modules; never put the legacy package on
    # sys.path (including in this test process). No editable install or checkout.
    location = os.environ.get("RTP_TEST_TORCHSPEC_ARCHIVE")
    if not location:
        raise RuntimeError("reference target requires RTP_TEST_TORCHSPEC_ARCHIVE (pinned TorchSpec git archive --format=zip)")
    path = Path(location).resolve(strict=True)
    with zipfile.ZipFile(path) as archive:
        if archive.comment != b"1b844317442f97464ed3d79677bbcb1a9f20c53e":
            raise ValueError("expected TorchSpec git archive of 1b844317442f97464ed3d79677bbcb1a9f20c53e")

    def load(package, name):
        spec = zipimport.zipimporter(str(path / package)).find_spec(name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load("torchspec/utils", "fp8"), load("torchspec/transfer/mooncake", "capture_metadata")


class HiddenStateFileStoreArchiveCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _reference()  # Missing or wrong fixture is a setup error, never a skip.

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_bidirectional_wire_compatibility(self):
        fp8_ref, schema = _reference()
        for codec in ("none", "zstd"):
            root = self.root / codec
            root.mkdir()
            with tempfile.TemporaryDirectory() as source:
                _child("write", Path(source), codec)
                expected = {}
                for key, layers, fp8 in _cases():
                    _, fields = _fixture(layers, fp8, quantize=fp8_ref.quantize_aux_hidden_states)
                    path = root / (key + ".capture")
                    shutil.copyfile(Path(source) / path.name, path)
                    wire = path.read_bytes()
                    magic, length = struct.unpack_from("<6sI", wire)
                    self.assertEqual(magic, b"RTPHS\x01" if codec == "none" else b"RTPHS\x02")
                    # New writer -> reference schema decoder.
                    marker = wire[10:10 + length]
                    layout = {"shapes": {k: v.shape for k, v in fields.items()},
                              "dtypes": {k: v.dtype for k, v in fields.items()}}
                    ref_marker = schema.encode_capture_metadata({"capture_layer_ids": list(layers)}, layout)
                    meta = schema.decode_capture_metadata(ref_marker)
                    self.assertEqual(schema.decode_capture_metadata(marker), meta)
                    expected[key] = (meta, _inventory(fields))

                    # Reference schema encoder + raw reference tensors -> standalone reader.
                    ref_key = key.replace("rtp_", "rtp_ref", 1)
                    payload = b"".join(_raw(v) for v in fields.values())
                    if codec == "zstd":
                        import zstandard
                        payload = zstandard.ZstdCompressor(level=1, write_checksum=True).compress(payload)
                    (root / (ref_key + ".capture")).write_bytes(
                        struct.pack("<6sI", magic, len(ref_marker)) + ref_marker + payload)
                    expected[ref_key] = expected[key]
            # The writer is gone; this new interpreter gets just the artifact directory.
            manifest = json.loads(_child("read", root).stdout)
            self.assertEqual(set(manifest), set(expected))
            for key, (meta, inventory) in expected.items():
                with self.subTest(codec=codec, key=key):
                    self.assertEqual(manifest[key]["codec"], codec)
                    self.assertEqual(manifest[key]["metadata"], meta)
                    self.assertEqual(manifest[key]["fields"], inventory)

    def test_local_quantizer_matches_pinned_raw_and_scales(self):
        for layers in _LAYOUTS:
            with self.subTest(layers=layers):
                aux, _ = _fixture(layers, False)
                actual = quantize_aux_hidden_states(aux, len(layers))
                reference = _reference()[0].quantize_aux_hidden_states(aux, len(layers))
                self.assertEqual(len(actual), 2)
                for tensor, ref in zip(actual, reference):
                    self.assertEqual(tensor.dtype, ref.dtype)
                    self.assertEqual(tensor.shape, ref.shape)
                    self.assertEqual(_raw(tensor), _raw(ref))


if __name__ == "__main__":
    unittest.main()
