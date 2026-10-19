import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import torch

from rtp_llm.utils import tensor_capture as capture


class TensorCaptureTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="tensor-capture-test-",
            dir=os.environ.get("TMPDIR", os.environ.get("TEST_TMPDIR")),
        )
        self.root = Path(self.temporary.name)
        self.record_dir = self.root / "records"
        self.config_path = self.root / "config.json"
        self.config = dict(
            enabled=True,
            authorized_capture=True,
            layers=[0, 1, 0],
            ops=["indexer.topk"],
            phases=["prefill"],
            rank_filter="record_owner",
            max_calls=100,
            max_bundle_bytes=1000000,
            max_total_bytes=10000000,
            max_inflight_bytes=10000000,
        )
        self.identity = dict(
            schema_version=1,
            session_id="canary-session",
            owner_id="123-456",
            replica_id="dp0",
            world_rank=0,
            dp_rank=0,
            exec_id="123456789",
            model_role="target",
            requests=[dict(phase="prefill", request_id="r1", q_len=2, kv_len=1)],
        )
        self.environment = mock.patch.dict(
            os.environ,
            RTP_LLM_RECORD_DIR=str(self.record_dir),
            RTP_LLM_RECORD_SESSION="canary-session",
        )
        self.environment.start()
        self.recorders = []

    def tearDown(self):
        for recorder in self.recorders:
            self.assertTrue(recorder.close(2))
        self.environment.stop()
        self.temporary.cleanup()

    def recorder(self, config=None, **kwargs):
        self.config_path.write_text(
            json.dumps(self.config if config is None else config)
        )
        recorder = capture.TensorCaptureRecorder(str(self.config_path), **kwargs)
        self.recorders.append(recorder)
        return recorder

    def context(self, recorder=None, identity=None):
        recorder = recorder or self.recorder()
        context = recorder.for_execution(
            json.dumps(self.identity if identity is None else identity)
        )
        self.assertIsNotNone(context)
        return context

    def begin(self, context, **kwargs):
        values = dict(
            layer=0,
            logits=torch.tensor([[1.0, 2.0, 999.0], [3.0, 4.0, 5.0]]),
            row_starts=torch.tensor([0, 0], dtype=torch.int32),
            row_ends=torch.tensor([2, 3], dtype=torch.int32),
            k=2,
            backend="dsv4_per_row",
            parameters={"num_rows": 2, "force_radix_sort": False},
            query_row_start=0,
            total_query_rows=2,
        )
        values.update(kwargs)
        return context.begin_topk(**values)

    def output(self):
        return torch.tensor([[1, 0], [2, 1]], dtype=torch.int32)

    def manifests(self):
        return sorted(
            self.record_dir.glob("owner-*/tensor-capture/sink-*/call-*/manifest.json")
        )

    def test_disabled_has_no_environment_files_threads_or_cuda(self):
        with mock.patch.dict(
            os.environ, {"RTP_LLM_RECORD_DIR": "", "RTP_LLM_RECORD_SESSION": ""}
        ), mock.patch.object(
            threading.Thread, "start", side_effect=AssertionError("thread")
        ), mock.patch.object(
            torch.cuda, "is_initialized", side_effect=AssertionError("cuda")
        ):
            recorder = self.recorder({})
            self.assertIsNone(recorder.for_execution("not json"))
            self.assertTrue(recorder.close())
        self.assertFalse(self.record_dir.exists())
        self.assertEqual(recorder.stats()["skip_reasons"], {"disabled": 1})

    def test_config_enabled_validates_without_recorder_side_effects(self):
        with mock.patch.dict(
            os.environ, RTP_LLM_RECORD_DIR="", RTP_LLM_RECORD_SESSION=""
        ), mock.patch.object(
            threading.Thread, "start", side_effect=AssertionError("thread")
        ), mock.patch.object(
            torch.cuda, "is_initialized", side_effect=AssertionError("cuda")
        ):
            for value, expected in (
                ({}, False),
                ({"enabled": False}, False),
                (self.config, True),
            ):
                self.config_path.write_text(json.dumps(value))
                self.assertIs(capture.config_enabled(str(self.config_path)), expected)
            self.config_path.write_text('{"enabled": 1}')
            with self.assertRaises(ValueError):
                capture.config_enabled(str(self.config_path))
        self.assertFalse(self.record_dir.exists())

    def test_strict_configuration(self):
        changes = [
            {"enabled": 1},
            {"authorized_capture": 1},
            {"authorized_capture": False},
            {"layers": [-1]},
            {"layers": [True]},
            {"layers": []},
            {"layers": "0"},
            {"ops": ["indexer.mqa_logits"]},
            {"ops": []},
            {"phases": ["decode"]},
            {"rank_filter": "all"},
            {"unexpected": 1},
            {"max_calls": 0},
            {"max_bundle_bytes": -1},
            {"max_total_bytes": 1.5},
            {"max_inflight_bytes": True},
        ]
        for change in changes:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    self.recorder(dict(self.config, **change))
        for field in (
            "authorized_capture",
            "max_calls",
            "rank_filter",
            "phases",
            "layers",
        ):
            with self.subTest(missing=field):
                config = dict(self.config)
                config.pop(field)
                with self.assertRaises(ValueError):
                    self.recorder(config)
        self.config_path.write_text('{"enabled":false,"enabled":true}')
        with self.assertRaises(ValueError):
            capture.TensorCaptureRecorder(str(self.config_path))
        self.assertFalse(self.record_dir.exists())
        recorder = self.recorder()
        self.assertEqual(recorder._config["layers"], [0, 1])

    def test_required_environment(self):
        for directory, session in (
            ("", "s"),
            ("relative", "s"),
            (str(self.record_dir), ""),
            (str(self.record_dir), "../s"),
            (str(self.root / ".." / "bad"), "s"),
        ):
            with self.subTest(directory=directory, session=session):
                with mock.patch.dict(
                    os.environ,
                    RTP_LLM_RECORD_DIR=directory,
                    RTP_LLM_RECORD_SESSION=session,
                ):
                    with self.assertRaises(ValueError):
                        self.recorder()
        self.assertFalse(self.record_dir.exists())

    def test_record_environment_file_or_symlink_rejected(self):
        blocked = self.root / "not-a-directory"
        blocked.write_text("file")
        link = self.root / "directory-link"
        link.symlink_to(self.root)
        for path in (blocked, blocked / "child", link / "child"):
            with self.subTest(path=path):
                with mock.patch.dict(os.environ, RTP_LLM_RECORD_DIR=str(path)):
                    with self.assertRaises(ValueError):
                        self.recorder()

    def test_owner_rank_only_and_identity_validation(self):
        nonowner = self.recorder(tp_rank=1)
        self.assertIsNone(nonowner.for_execution(json.dumps(self.identity)))
        self.assertEqual(nonowner.stats()["skip_reasons"]["non_owner_rank"], 1)
        self.assertFalse(self.record_dir.exists())
        recorder = self.recorder()
        changes = [
            {"schema_version": 2},
            {"schema_version": True},
            {"session_id": "other"},
            {"owner_id": "../escape"},
            {"owner_id": "a/b"},
            {"replica_id": ""},
            {"world_rank": 1},
            {"world_rank": True},
            {"dp_rank": -1},
            {"exec_id": 1},
            {"exec_id": "../../x"},
            {"model_role": "draft"},
            {"requests": []},
        ]
        for change in changes:
            with self.subTest(change=change):
                self.assertIsNone(
                    recorder.for_execution(json.dumps(dict(self.identity, **change)))
                )
        for change in (
            {"is_fake": True},
            {"is_fake": 0},
            {"phase": "decode"},
            {"request_id": "customer-id"},
            {"q_len": 0},
            {"q_len": True},
            {"kv_len": -1},
            {"sequence_id": "s1"},
        ):
            identity = copy.deepcopy(self.identity)
            identity["requests"][0].update(change)
            self.assertIsNone(recorder.for_execution(json.dumps(identity)))
        self.assertFalse(self.record_dir.exists())
        self.context(recorder)
        changed_owner = dict(self.identity, owner_id="another-owner")
        self.assertIsNone(recorder.for_execution(json.dumps(changed_owner)))
        self.assertEqual(recorder.stats()["skip_reasons"]["owner_mismatch"], 1)

    def test_owner_can_have_nonzero_world_rank_and_all_rows_must_be_legal(self):
        recorder = self.recorder(world_rank=4)
        identity = copy.deepcopy(self.identity)
        identity.update(world_rank=4, dp_rank=2, replica_id="dp2")
        context = self.context(recorder, identity)
        self.begin(context).finish(self.output())
        self.assertTrue(recorder.close(2))
        manifest, _ = capture.load_capture(self.manifests()[0])
        self.assertEqual((manifest["rank"], manifest["dp_rank"]), (4, 2))
        other = self.recorder()
        identity = copy.deepcopy(self.identity)
        identity["requests"].append(
            dict(phase="prefill", request_id="r2", q_len=1, kv_len=0, is_fake=True)
        )
        self.assertIsNone(other.for_execution(json.dumps(identity)))
        identity["requests"][-1].update(is_fake=False, phase="decode")
        self.assertIsNone(other.for_execution(json.dumps(identity)))

    def test_frozen_identity_unknown_fields_not_exported(self):
        identity = copy.deepcopy(self.identity)
        identity["prompt"] = "SECRET-PROMPT"
        identity["requests"][0]["trace_id"] = "SECRET-TRACE"
        identity["requests"][0]["sequence_id"] = 7
        context = self.context(identity=identity)
        identity["requests"][0]["q_len"] = 999
        with self.assertRaises(TypeError):
            context.identity["requests"][0]["q_len"] = 999
        self.begin(context).finish(self.output())
        self.assertTrue(context._recorder.close(2))
        text = self.manifests()[0].read_text()
        self.assertNotIn("SECRET", text)
        self.assertNotIn("trace_id", text)
        self.assertEqual(json.loads(text)["requests"][0]["sequence_id"], 7)

    def test_mask_snapshot_before_after_alias_and_layout(self):
        context = self.context()
        base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        logits = base[1:3, 1::2]
        logits[0, 2] = float("nan")
        logits[0, 0] = float("nan")
        original = logits.clone()
        starts, ends = torch.tensor([0, 0], dtype=torch.int32), torch.tensor(
            [2, 3], dtype=torch.int32
        )
        token = self.begin(context, logits=logits, row_starts=starts, row_ends=ends)
        self.assertIsNotNone(token)
        torch.testing.assert_close(logits, original, equal_nan=True)
        logits.fill_(99)
        starts.fill_(99)
        ends.fill_(99)
        backing = torch.zeros((2, 4), dtype=torch.int32)
        output = backing[:, 1::2]
        output.copy_(self.output())
        token.finish(output)
        output.fill_(77)
        token.finish(output)
        token.cancel()
        self.assertTrue(context._recorder.close(2))
        manifest, tensors = capture.load_capture(self.manifests()[0])
        original[0, 2] = 0
        torch.testing.assert_close(tensors["logits"], original, equal_nan=True)
        torch.testing.assert_close(tensors["indices"], self.output())
        self.assertEqual(manifest["tensors"]["logits"]["original_stride"], [6, 2])
        self.assertEqual(manifest["tensors"]["logits"]["storage_offset"], 7)
        self.assertEqual(manifest["tensors"]["indices"]["original_stride"], [4, 2])
        self.assertFalse(manifest["finite_values_checked"])
        self.assertEqual(manifest["parameters"]["index_space"], "relative_to_row_start")
        self.assertEqual(context._recorder.stats()["inflight_bytes"], 0)
        self.assertEqual(context._recorder.stats()["completed"], 1)
        for path in self.record_dir.rglob("*"):
            if path.is_file():
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            elif path.name.startswith("sink-") or path.name.startswith("call-"):
                self.assertEqual(path.stat().st_mode & 0o777, 0o700)

    def test_prefix_chunk_across_requests(self):
        identity = copy.deepcopy(self.identity)
        identity["requests"].append(
            dict(phase="prefill", request_id="r2", q_len=3, kv_len=2)
        )
        context = self.context(identity=identity)
        logits = torch.arange(24, dtype=torch.float32).reshape(3, 8)
        token = self.begin(
            context,
            logits=logits,
            query_row_start=1,
            total_query_rows=5,
            row_starts=torch.tensor([0, 3, 3], dtype=torch.int32),
            row_ends=torch.tensor([3, 6, 7], dtype=torch.int32),
        )
        self.assertIsNotNone(token)
        token.finish(torch.zeros((3, 2), dtype=torch.int32))
        self.assertTrue(context._recorder.close(2))
        manifest, tensors = capture.load_capture(self.manifests()[0].parent)
        self.assertEqual(manifest["query_row_start"], 1)
        self.assertEqual(manifest["total_query_rows"], 5)
        expected = logits.clone()
        expected[0, 3:] = 0
        expected[1, :3] = 0
        expected[1, 6:] = 0
        expected[2, :3] = 0
        expected[2, 7:] = 0
        torch.testing.assert_close(tensors["logits"], expected)

    def test_reject_mapping_dtypes_parameters_and_output(self):
        context = self.context()
        changes = [
            {"query_row_start": 1},
            {"total_query_rows": 3},
            {"query_row_start": -1},
            {"logits": torch.ones(2, 2)},
            {"logits": torch.ones(2, 3, dtype=torch.float16)},
            {"logits": torch.empty(0, 3)},
            {"row_starts": torch.zeros(2, dtype=torch.int64)},
            {"row_starts": torch.tensor([0, 1], dtype=torch.int32)},
            {"row_ends": torch.tensor([1, 3], dtype=torch.int32)},
            {"row_ends": torch.tensor([2, 4], dtype=torch.int32)},
            {"row_ends": torch.tensor([-1, 3], dtype=torch.int32)},
            {"row_ends": torch.tensor([2], dtype=torch.int32)},
            {"k": 0},
            {"k": True},
            {"backend": "../bad"},
            {"parameters": {"k": 3}},
            {"parameters": {"index_space": "absolute"}},
            {"parameters": {"x": float("nan")}},
            {"parameters": {"x": [1, 2]}},
            {"parameters": {"x": "business text"}},
        ]
        for change in changes:
            with self.subTest(keys=list(change)):
                self.assertIsNone(self.begin(context, **change))
                self.assertEqual(context._recorder.stats()["inflight_bytes"], 0)
        self.assertIsNone(self.begin(context, layer=8))
        self.begin(context).finish(torch.zeros(2, 2))
        self.begin(context).finish(torch.zeros(2, 1, dtype=torch.int32))
        self.assertEqual(context._recorder.stats()["pending"], 0)
        self.assertEqual(self.manifests(), [])

    def test_zero_prefix_padding_and_cancel(self):
        identity = copy.deepcopy(self.identity)
        identity["requests"][0].update(q_len=1, kv_len=0)
        context = self.context(identity=identity)
        token = self.begin(
            context,
            logits=torch.tensor([[-3.0]]),
            row_starts=torch.tensor([0], dtype=torch.int32),
            row_ends=torch.tensor([1], dtype=torch.int32),
            total_query_rows=1,
        )
        self.assertIsNotNone(token)
        token.finish(torch.tensor([[0, -1]], dtype=torch.int32))
        self.assertTrue(context._recorder.close(2))
        _, tensors = capture.load_capture(self.manifests()[0])
        self.assertEqual(tensors["indices"].tolist(), [[0, -1]])
        other = self.context()
        token = self.begin(other)
        token.cancel()
        token.cancel()
        token.finish(self.output())
        self.assertEqual(other._recorder.stats()["pending"], 0)
        self.assertEqual(other._recorder.stats()["reserved_bytes"], 0)
        self.assertEqual(other._recorder.stats()["skip_reasons"]["kernel_error"], 1)
        token = self.begin(other)
        del token
        self.assertEqual(other._recorder.stats()["pending"], 0)

    def test_raw_encoding_bfloat_float8_integers_noncontiguous_empty_scalar(self):
        examples = [
            torch.arange(24, dtype=torch.int32).reshape(4, 6)[1:, ::2],
            torch.empty((2, 0, 3), dtype=torch.float32),
            torch.tensor(42, dtype=torch.int64),
            torch.tensor([True, False]),
            torch.tensor([-128, 127], dtype=torch.int8),
            torch.tensor([0, 32767, -32768, -1], dtype=torch.int16).view(
                torch.bfloat16
            ),
        ]
        for name in capture._DTYPES:
            if name.startswith("float8"):
                examples.append(
                    torch.arange(256, dtype=torch.uint8).view(capture._DTYPES[name])
                )
        for number, tensor in enumerate(examples):
            with self.subTest(dtype=tensor.dtype, shape=tensor.shape):
                desc, data = capture._encode_tensor(tensor)
                self.assertEqual(desc["original_stride"], list(tensor.stride()))
                self.assertEqual(desc["storage_offset"], tensor.storage_offset())
                self.assertEqual(desc["nbytes"], tensor.numel() * tensor.element_size())
                directory = self.root / ("raw-%d" % number)
                directory.mkdir()
                desc["file"] = "value.bin"
                (directory / "value.bin").write_bytes(data)
                (directory / "manifest.json").write_text(
                    json.dumps(
                        dict(schema_version=1, complete=True, tensors={"value": desc})
                    )
                )
                _, restored = capture.load_capture(directory)
                expected = tensor.contiguous().reshape(-1).view(torch.uint8)
                self.assertTrue(
                    torch.equal(
                        restored["value"].reshape(-1).view(torch.uint8), expected
                    )
                )
                self.assertEqual(list(restored["value"].shape), list(tensor.shape))
        desc, data = capture._encode_tensor(
            torch.tensor([0x01020304], dtype=torch.int32)
        )
        self.assertEqual(data, b"\x04\x03\x02\x01")
        self.assertEqual(desc["sha256"], hashlib.sha256(data).hexdigest())

    def test_loader_rejects_corruption_incomplete_paths_and_bounds(self):
        context = self.context()
        self.begin(context).finish(self.output())
        context._recorder.close(2)
        path = self.manifests()[0]
        original = json.loads(path.read_text())
        tensor = path.parent / "logits.bin"
        data = tensor.read_bytes()
        tensor.write_bytes(b"x" + data[1:])
        with self.assertRaisesRegex(ValueError, "checksum"):
            capture.load_capture(path)
        tensor.write_bytes(data + b"x")
        with self.assertRaises(ValueError):
            capture.load_capture(path)
        tensor.write_bytes(data)
        mutations = [
            lambda m: m.update(complete=False),
            lambda m: m.update(schema_version=99),
            lambda m: m["tensors"]["logits"].update(file="../logits.bin"),
            lambda m: m["tensors"]["logits"].update(file=str(tensor)),
            lambda m: m["tensors"]["logits"].update(dtype="pickle"),
            lambda m: m["tensors"]["logits"].update(shape=[-1]),
            lambda m: m["tensors"]["logits"].update(shape=[True, 3]),
            lambda m: m["tensors"]["logits"].update(shape=[2**40, 3]),
            lambda m: m["tensors"]["logits"].update(shape=[2, 4]),
            lambda m: m["tensors"]["logits"].update(original_stride=[-1, 1]),
            lambda m: m["tensors"]["logits"].update(storage_offset=-1),
            lambda m: m["tensors"]["logits"].update(byte_order="big"),
            lambda m: m["tensors"]["logits"].update(nbytes=2**40),
            lambda m: m["tensors"]["logits"].update(sha256="wrong"),
        ]
        for mutate in mutations:
            manifest = copy.deepcopy(original)
            mutate(manifest)
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                capture.load_capture(path)
        path.write_text(json.dumps(original))
        tensor.unlink()
        outside = self.root / "outside.bin"
        outside.write_bytes(data)
        tensor.symlink_to(outside)
        with self.assertRaises((OSError, ValueError)):
            capture.load_capture(path)
        alias = self.root / "alias"
        alias.symlink_to(path.parent)
        with self.assertRaises(ValueError):
            capture.load_capture(alias)

    def test_budgets_reserved_before_copy(self):
        for field, reason in (
            ("max_bundle_bytes", "oversize"),
            ("max_total_bytes", "total_budget"),
            ("max_inflight_bytes", "inflight_budget"),
        ):
            with self.subTest(field=field):
                context = self.context(self.recorder(dict(self.config, **{field: 1})))
                with mock.patch.object(
                    capture,
                    "_snapshot",
                    side_effect=AssertionError("copied before budget"),
                ):
                    self.assertIsNone(self.begin(context))
                self.assertEqual(context._recorder.stats()["skip_reasons"][reason], 1)
                self.assertEqual(context._recorder.stats()["selected"], 0)
        context = self.context(self.recorder(dict(self.config, max_calls=1)))
        token = self.begin(context)
        self.assertIsNone(self.begin(context))
        token.cancel()
        self.assertIsNone(self.begin(context))
        self.assertEqual(context._recorder.stats()["skip_reasons"]["max_calls"], 2)

    def test_total_and_inflight_budget_lifetime(self):
        context = self.context()
        recorder = context._recorder
        token = self.begin(context)
        size = recorder.stats()["reserved_bytes"]
        recorder._config["max_total_bytes"] = size
        self.assertIsNone(self.begin(context))
        token.cancel()
        recorder._config["max_total_bytes"] = 10 * size
        recorder._config["max_inflight_bytes"] = capture._INFLIGHT_FACTOR * size
        token = self.begin(context)
        self.assertIsNone(self.begin(context))
        entered, release = threading.Event(), threading.Event()
        original = recorder._write_bundle

        def blocked(*args):
            entered.set()
            self.assertTrue(release.wait(2))
            return original(*args)

        with mock.patch.object(recorder, "_write_bundle", side_effect=blocked):
            try:
                token.finish(self.output())
                self.assertTrue(entered.wait(2))
                self.assertEqual(recorder.stats()["queue_size"], 0)
                self.assertEqual(
                    recorder.stats()["inflight_bytes"], capture._INFLIGHT_FACTOR * size
                )
                self.assertIsNone(self.begin(context))
                self.assertFalse(recorder.close(0.01))
            finally:
                release.set()
            self.assertTrue(recorder.close(2))
        self.assertEqual(recorder.stats()["inflight_bytes"], 0)
        self.assertEqual(recorder.stats()["reserved_bytes"], size)
        self.assertEqual(recorder.stats()["completed"], 1)
        # A successfully published bundle permanently consumes total budget.
        other = self.context()
        first = self.begin(other)
        other._recorder._config["max_total_bytes"] = other._recorder.stats()[
            "reserved_bytes"
        ]
        first.finish(self.output())
        with other._recorder._condition:
            self.assertTrue(
                other._recorder._condition.wait_for(
                    lambda: other._recorder._pending == 0, 2
                )
            )
        self.assertIsNone(self.begin(other))
        self.assertEqual(other._recorder.stats()["skip_reasons"]["total_budget"], 1)

    def test_bounded_queue_and_enqueue_race(self):
        context = self.context()
        recorder = context._recorder
        self.assertEqual(recorder._queue.maxsize, 32)
        entered, release = threading.Event(), threading.Event()
        original = recorder._write_bundle

        def blocked(*args):
            entered.set()
            self.assertTrue(release.wait(5))
            return original(*args)

        with mock.patch.object(recorder, "_write_bundle", side_effect=blocked):
            try:
                self.begin(context).finish(self.output())
                self.assertTrue(entered.wait(2))
                late_token = self.begin(context)
                for _ in range(32):
                    token = self.begin(context)
                    self.assertIsNotNone(token)
                    token.finish(self.output())
                self.assertIsNone(self.begin(context))
                late_token.finish(self.output())
                self.assertEqual(recorder.stats()["skip_reasons"]["queue_full"], 2)
                self.assertEqual(recorder.stats()["pending"], 33)
            finally:
                release.set()
            self.assertTrue(recorder.close(5))
        self.assertEqual(recorder.stats()["completed"], 33)
        self.assertEqual(recorder.stats()["inflight_bytes"], 0)

    def test_thread_and_io_failure_fail_open_without_manifest(self):
        recorder = self.recorder()
        with mock.patch.object(
            threading.Thread, "start", side_effect=RuntimeError("no threads")
        ):
            self.assertIsNone(recorder.for_execution(json.dumps(self.identity)))
        self.assertIsNone(recorder.for_execution(json.dumps(self.identity)))
        self.assertEqual(recorder.stats()["writer_error"], 1)
        context = self.context()
        original = capture._atomic_file

        def fail(path, data):
            if path.name == "row_ends.bin":
                raise OSError("disk full")
            return original(path, data)

        with mock.patch.object(capture, "_atomic_file", side_effect=fail):
            self.begin(context).finish(self.output())
            self.assertTrue(context._recorder.close(2))
        self.assertEqual(self.manifests(), [])
        self.assertEqual(context._recorder.stats()["io_error"], 1)
        self.assertEqual(context._recorder.stats()["bytes_published"], 0)
        self.assertGreater(context._recorder.stats()["reserved_bytes"], 0)
        self.assertEqual(context._recorder.stats()["inflight_bytes"], 0)
        self.assertTrue(list(self.record_dir.rglob("row_starts.bin")))

    def test_manifest_publish_failure_keeps_incomplete_and_total_charge(self):
        context = self.context()
        recorder = context._recorder
        token = self.begin(context)
        size = recorder.stats()["reserved_bytes"]
        recorder._config["max_total_bytes"] = size
        replace = os.replace

        def fail_manifest(source, destination):
            if destination.name == "manifest.json":
                raise OSError("manifest rename failed")
            return replace(source, destination)

        with mock.patch.object(os, "replace", side_effect=fail_manifest):
            token.finish(self.output())
            with recorder._condition:
                self.assertTrue(
                    recorder._condition.wait_for(lambda: recorder._pending == 0, 2)
                )
        self.assertEqual(self.manifests(), [])
        self.assertEqual(list(self.record_dir.rglob("*.tmp")), [])
        self.assertEqual(recorder.stats()["io_error"], 1)
        self.assertEqual(recorder.stats()["bytes_published"], 0)
        self.assertEqual(recorder.stats()["reserved_bytes"], size)
        self.assertIsNone(self.begin(context))
        self.assertEqual(recorder.stats()["skip_reasons"]["total_budget"], 1)

    def test_snapshot_errors_and_close_during_copy(self):
        context = self.context()
        recorder = context._recorder
        with mock.patch.object(
            capture, "_snapshot", side_effect=RuntimeError("copy failed")
        ):
            self.assertIsNone(self.begin(context))
        self.assertEqual(recorder.stats()["pending"], 0)
        token = self.begin(context)
        with mock.patch.object(
            capture, "_snapshot", side_effect=RuntimeError("output copy failed")
        ):
            token.finish(self.output())
        self.assertEqual(recorder.stats()["pending"], 0)
        entered, release = threading.Event(), threading.Event()
        original = capture._snapshot
        result = []

        def blocked(tensor):
            if tensor.dtype == torch.float32:
                entered.set()
                self.assertTrue(release.wait(2))
            return original(tensor)

        with mock.patch.object(capture, "_snapshot", side_effect=blocked):
            worker = threading.Thread(target=lambda: result.append(self.begin(context)))
            try:
                worker.start()
                self.assertTrue(entered.wait(2))
                self.assertFalse(recorder.close(0.01))
                self.assertGreater(recorder.stats()["inflight_bytes"], 0)
            finally:
                release.set()
                worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertIsNotNone(result[0])
        result[0].finish(self.output())
        self.assertTrue(recorder.close(2))
        self.assertIsNone(self.begin(context))
        self.assertIsNone(recorder.for_execution(json.dumps(self.identity)))
        self.assertEqual(recorder.stats()["inflight_bytes"], 0)
        self.assertEqual(self.manifests(), [])

    def test_two_model_instances_do_not_overwrite_and_metadata(self):
        with mock.patch.object(
            torch.cuda, "is_initialized", return_value=False
        ), mock.patch.object(
            torch.cuda, "is_available", side_effect=AssertionError("cuda probe")
        ), mock.patch.object(
            torch.cuda, "get_device_properties", side_effect=AssertionError("cuda init")
        ):
            contexts = [self.context(), self.context()]
            for context in contexts:
                self.begin(context).finish(self.output())
            for context in contexts:
                self.assertTrue(context._recorder.close(2))
        paths = self.manifests()
        self.assertEqual(len(paths), 2)
        self.assertNotEqual(paths[0].parent.parent, paths[1].parent.parent)
        for path in paths:
            metadata = json.loads((path.parent.parent / "run.json").read_text())
            self.assertEqual(metadata["owner_id"], "123-456")
            self.assertIsNone(metadata["model_provenance"])
            self.assertIsNone(metadata["gpu"])
            self.assertEqual(
                metadata["config_sha256"],
                hashlib.sha256(capture._json_bytes(metadata["config"])).hexdigest(),
            )

    def test_capture_directory_symlink_refused(self):
        self.record_dir.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        (self.record_dir / "owner-123-456").symlink_to(outside)
        recorder = self.recorder()
        recorder.for_execution(json.dumps(self.identity))
        self.assertTrue(recorder.close(2))
        self.assertEqual(recorder.stats()["writer_error"], 1)
        self.assertEqual(list(outside.iterdir()), [])

    def test_close_is_bounded_during_sink_initialization(self):
        recorder = self.recorder()
        entered, release = threading.Event(), threading.Event()
        original = recorder._initialize_sink

        def blocked():
            entered.set()
            self.assertTrue(release.wait(2))
            return original()

        with mock.patch.object(recorder, "_initialize_sink", side_effect=blocked):
            try:
                self.context(recorder)
                self.assertTrue(entered.wait(2))
                self.assertFalse(recorder.close(0.01))
                self.assertTrue(recorder._thread.daemon)
            finally:
                release.set()
            self.assertTrue(recorder.close(2))

    def test_sink_init_failure_drains_queued_payload(self):
        recorder = self.recorder()
        entered, release = threading.Event(), threading.Event()

        def blocked_failure():
            entered.set()
            self.assertTrue(release.wait(2))
            raise OSError("metadata failed")

        with mock.patch.object(
            recorder, "_initialize_sink", side_effect=blocked_failure
        ):
            try:
                context = self.context(recorder)
                self.assertTrue(entered.wait(2))
                self.begin(context).finish(self.output())
                self.assertGreater(recorder.stats()["inflight_bytes"], 0)
            finally:
                release.set()
            self.assertTrue(recorder.close(2))
        self.assertEqual(recorder.stats()["inflight_bytes"], 0)
        self.assertEqual(recorder.stats()["reserved_bytes"], 0)
        self.assertTrue(recorder.stats()["failed"])
        self.assertEqual(self.manifests(), [])

    def test_concurrent_selection_limits_and_exact_bundle_boundary(self):
        recorder = self.recorder(dict(self.config, max_calls=5))
        context = self.context(recorder)
        token = self.begin(context)
        size = recorder.stats()["reserved_bytes"]
        token.cancel()
        recorder._config["max_bundle_bytes"] = size
        barrier = threading.Barrier(9)
        failures = []

        def select():
            try:
                barrier.wait(2)
                selected = self.begin(context)
                if selected is not None:
                    selected.finish(self.output())
            except BaseException as error:
                failures.append(error)

        workers = [threading.Thread(target=select) for _ in range(8)]
        for worker in workers:
            worker.start()
        barrier.wait(2)
        for worker in workers:
            worker.join(2)
            self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        self.assertTrue(recorder.close(2))
        stats = recorder.stats()
        self.assertEqual(stats["selected"], 5)
        self.assertEqual(stats["completed"], 4)
        self.assertEqual(stats["inflight_bytes"], 0)
        self.assertEqual(stats["reserved_bytes"], 4 * size)
        self.assertEqual(stats["skip_reasons"]["max_calls"], 4)

    def test_writer_reader_payload_limit_before_reservation_or_copy(self):
        context = self.context()
        recorder = context._recorder
        # logits=24, ranges=8+8, indices=16. All individual files fit 55.
        payload = 56
        with mock.patch.object(
            capture, "_MAX_FILE_BYTES", payload - 1
        ), mock.patch.object(recorder, "_reserve") as reserve, mock.patch.object(
            capture, "_snapshot"
        ) as snapshot:
            self.assertIsNone(self.begin(context))
            reserve.assert_not_called()
            snapshot.assert_not_called()
        self.assertEqual(recorder.stats()["selected"], 0)
        self.assertEqual(recorder.stats()["skip_reasons"]["payload_limit"], 1)
        with mock.patch.object(capture, "_MAX_FILE_BYTES", payload):
            token = self.begin(context)
            self.assertIsNotNone(token)
            token.finish(self.output())
            self.assertTrue(recorder.close(2))
            path = self.manifests()[0]
            manifest, tensors = capture.load_capture(path)
            self.assertEqual(
                sum(d["nbytes"] for d in manifest["tensors"].values()), payload
            )
            torch.testing.assert_close(tensors["indices"], self.output())
        self.assertEqual(recorder.stats()["completed"], 1)
        self.assertEqual(
            recorder.stats()["bytes_published"], payload + path.stat().st_size
        )
        with mock.patch.object(
            capture, "_MAX_FILE_BYTES", payload - 1
        ), mock.patch.object(
            capture, "_read_regular", wraps=capture._read_regular
        ) as read:
            with self.assertRaisesRegex(ValueError, "capture exceeds loader limit"):
                capture.load_capture(path)
            # Reject the aggregate before reading ANY tensor payload.
            self.assertEqual(read.call_count, 1)
            self.assertEqual(read.call_args.args[0], path)

    def test_same_token_finish_finish_and_finish_cancel_races(self):
        for first, second in (
            ("finish", "finish"),
            ("finish", "cancel"),
            ("cancel", "finish"),
        ):
            with self.subTest(first=first, second=second):
                context = self.context()
                recorder = context._recorder
                token = self.begin(context)
                entered, release = threading.Event(), threading.Event()
                second_started, second_done = threading.Event(), threading.Event()
                errors = []
                target = capture if first == "finish" else recorder
                name = "_snapshot" if first == "finish" else "_release"
                original = getattr(target, name)

                def blocked(*args, **kwargs):
                    entered.set()
                    if not release.wait(5):
                        raise AssertionError("test did not release token operation")
                    return original(*args, **kwargs)

                def invoke(operation, competing=False):
                    try:
                        if competing:
                            second_started.set()
                        if operation == "finish":
                            token.finish(self.output())
                        else:
                            token.cancel()
                    except BaseException as error:
                        errors.append(error)
                    finally:
                        if competing:
                            second_done.set()

                with mock.patch.object(target, name, side_effect=blocked) as operation:
                    workers = [
                        threading.Thread(target=invoke, args=(first,)),
                        threading.Thread(target=invoke, args=(second, True)),
                    ]
                    try:
                        workers[0].start()
                        self.assertTrue(entered.wait(2))
                        workers[1].start()
                        self.assertTrue(second_started.wait(2))
                        self.assertFalse(second_done.is_set())
                        self.assertEqual(recorder.stats()["pending"], 1)
                    finally:
                        release.set()
                        for worker in workers:
                            if worker.ident is not None:
                                worker.join(5)
                    self.assertTrue(all(not worker.is_alive() for worker in workers))
                    self.assertEqual(operation.call_count, 1)
                self.assertEqual(errors, [])
                self.assertTrue(recorder.close(2))
                stats = recorder.stats()
                self.assertEqual(stats["selected"], 1)
                self.assertEqual(stats["completed"], int(first == "finish"))
                self.assertEqual(stats["pending"], 0)
                self.assertEqual(stats["inflight_bytes"], 0)
                if first == "cancel":
                    self.assertEqual(stats["reserved_bytes"], 0)
                    self.assertEqual(stats["bytes_published"], 0)

    def test_close_during_output_snapshot_retains_reservation(self):
        context = self.context()
        recorder = context._recorder
        token = self.begin(context)
        before = recorder.stats()
        entered, release = threading.Event(), threading.Event()
        original = capture._snapshot
        errors = []

        def blocked(tensor):
            snapshot = original(tensor)
            entered.set()
            if not release.wait(5):
                raise AssertionError("test did not release output snapshot")
            return snapshot

        def finish():
            try:
                token.finish(self.output())
            except BaseException as error:
                errors.append(error)

        with mock.patch.object(capture, "_snapshot", side_effect=blocked):
            worker = threading.Thread(target=finish)
            try:
                worker.start()
                self.assertTrue(entered.wait(2))
                self.assertFalse(recorder.close(0.01))
                self.assertEqual(recorder.stats()["pending"], 1)
                self.assertEqual(
                    recorder.stats()["inflight_bytes"], before["inflight_bytes"]
                )
                self.assertEqual(
                    recorder.stats()["reserved_bytes"], before["reserved_bytes"]
                )
                self.assertTrue(token._tensors)
            finally:
                release.set()
                worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(recorder.close(2))
        self.assertEqual(token._tensors, {})
        self.assertEqual(recorder.stats()["pending"], 0)
        self.assertEqual(recorder.stats()["inflight_bytes"], 0)
        self.assertEqual(recorder.stats()["reserved_bytes"], 0)
        self.assertEqual(self.manifests(), [])

    def test_reader_rejects_manifest_symlink_fifo_and_duplicate_keys(self):
        context = self.context()
        self.begin(context).finish(self.output())
        self.assertTrue(context._recorder.close(2))
        path = self.manifests()[0]
        original = path.read_text()
        alias = self.root / "alias-manifest"
        alias.mkdir()
        (alias / "manifest.json").symlink_to(path)
        with self.assertRaisesRegex(ValueError, "unsafe manifest path"):
            capture.load_capture(alias)
        for marker, replacement in (
            ('"complete":true', '"complete":false,"complete":true'),
            ('"dtype":"float32"', '"dtype":"float16","dtype":"float32"'),
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, original)
                path.write_text(original.replace(marker, replacement, 1))
                with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                    capture.load_capture(path)
        path.write_text(original)
        # A subprocess timeout keeps a missing O_NONBLOCK regression from hanging UT.
        fifo_dir = self.root / "fifo"
        fifo_dir.mkdir()
        fifo = fifo_dir / "manifest.json"
        os.mkfifo(fifo, 0o600)
        program = (
            "import sys\n"
            "from rtp_llm.utils.tensor_capture import load_capture\n"
            "try:\n"
            "    load_capture(sys.argv[1])\n"
            "except ValueError as error:\n"
            "    assert 'size/type' in str(error), str(error)\n"
            "else:\n"
            "    raise AssertionError('FIFO accepted')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", program, str(fifo)],
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(
        os.environ.get("RTP_LLM_TEST_TENSOR_CAPTURE_CUDA") == "1",
        "opt in with RTP_LLM_TEST_TENSOR_CAPTURE_CUDA=1",
    )
    def test_cuda_nondefault_stream_noncontiguous_owned_snapshot(self):
        self.assertTrue(
            torch.cuda.is_available(), "CUDA opt-in requires an available GPU"
        )
        context = self.context()
        producer = torch.cuda.Stream()
        self.assertNotEqual(
            producer.cuda_stream, torch.cuda.default_stream().cuda_stream
        )
        with torch.cuda.stream(producer):
            # Allocation, producer writes, snapshots and subsequent overwrites are
            # deliberately on the same nondefault stream, without a pre-copy sync.
            backing = torch.empty((2, 6), dtype=torch.float32, device="cuda")
            logits = backing[:, ::2]
            logits.fill_(7)
            logits[0, 2] = 99
            ranges = torch.empty((2, 4), dtype=torch.int32, device="cuda")
            starts, ends = ranges[:, 0], ranges[:, 2]
            starts.zero_()
            ends[0] = 2
            ends[1] = 3
            token = self.begin(context, logits=logits, row_starts=starts, row_ends=ends)
            self.assertIsNotNone(token)
            logits.fill_(101)
            ranges.fill_(102)
            output_backing = torch.empty((2, 4), dtype=torch.int32, device="cuda")
            output = output_backing[:, ::2]
            output[:, 0] = 1
            output[:, 1] = 0
            token.finish(output)
            output.fill_(103)
        producer.synchronize()
        self.assertTrue(context._recorder.close(5))
        manifest, tensors = capture.load_capture(self.manifests()[0])
        torch.testing.assert_close(
            tensors["logits"], torch.tensor([[7.0, 7.0, 0.0], [7.0, 7.0, 7.0]])
        )
        torch.testing.assert_close(
            tensors["row_starts"], torch.tensor([0, 0], dtype=torch.int32)
        )
        torch.testing.assert_close(
            tensors["row_ends"], torch.tensor([2, 3], dtype=torch.int32)
        )
        torch.testing.assert_close(
            tensors["indices"], torch.tensor([[1, 0], [1, 0]], dtype=torch.int32)
        )
        self.assertEqual(manifest["tensors"]["logits"]["original_stride"], [6, 2])
        self.assertEqual(manifest["tensors"]["indices"]["original_stride"], [4, 2])
        self.assertTrue(all(t.device.type == "cpu" for t in tensors.values()))
        self.assertEqual(
            logits.cpu().tolist(), [[101.0, 101.0, 101.0], [101.0, 101.0, 101.0]]
        )
        self.assertEqual(output.cpu().tolist(), [[103, 103], [103, 103]])

    def test_stats_snapshot_is_independent_and_reasons_bounded(self):
        recorder = self.recorder({})
        for i in range(100):
            recorder.note_skip("reason_%d" % i)
        recorder.note_skip("request data must not appear")
        stats = recorder.stats()
        self.assertLessEqual(len(stats["skip_reasons"]), 65)
        stats["skip_reasons"].clear()
        self.assertEqual(recorder.stats()["skipped"], 101)
        self.assertTrue(recorder.stats()["skip_reasons"])


if __name__ == "__main__":
    unittest.main()
