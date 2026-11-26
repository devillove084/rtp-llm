import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

from rtp_llm.utils import hidden_state_capture_trace as trace


class _TraceFixture(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.directory = Path(tmp.name)
        self.output = self.directory / "timeline.json"
        self.recorder = trace._Recorder(self.directory)
        recorder_patch = patch.object(trace, "_recorder", self.recorder)
        recorder_patch.start()
        self.addCleanup(recorder_patch.stop)
        self.addCleanup(self.recorder.close)

    def events(self):
        trace.close()
        trace.convert(self.directory, self.output)
        return json.loads(self.output.read_text())["traceEvents"]


class HiddenStateCaptureTraceTest(_TraceFixture):
    def test_disabled_never_touches_arguments_or_store(self):
        with patch.object(trace, "_recorder", None), patch.object(trace, "_event", side_effect=AssertionError):
            self.assertFalse(trace.enabled())
            trace.start()
            trace.mark("disabled", None)
            with trace.span("disabled", None):
                pass
            self.assertFalse(trace.instrument_mooncake_store(None))
            trace.close()
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_deferred_submit_uses_origin_tid_and_original_batch(self):
        keys = ["first", "second"]
        submit_tid, submit_ns = threading.get_native_id(), time.monotonic_ns()
        with ThreadPoolExecutor(1) as pool:
            pool.submit(trace.mark, "publisher.submit", keys, ts_ns=submit_ns, origin_tid=submit_tid).result()
            pool.submit(trace.mark, "publisher.dequeue", keys).result()
        keys.clear()
        submit, dequeue = self.events()
        self.assertEqual((submit["tid"], submit["ts"]), (submit_tid, submit_ns / 1000))
        self.assertNotEqual(dequeue["tid"], submit_tid)
        self.assertEqual(submit["args"], {"keys": ["first", "second"]})

    def test_forward_does_not_wait_for_writer_and_overflow_is_visible(self):
        writing, release = threading.Event(), threading.Event()
        tid = threading.get_native_id()
        real_open = tempfile.NamedTemporaryFile
        writer_tids = []

        def slow_open(**kwargs):
            writer_tids.append(threading.get_native_id())
            writing.set()
            if not release.wait(5):
                raise AssertionError("producer waited for file I/O")
            return real_open(**kwargs)

        with patch.object(tempfile, "NamedTemporaryFile", slow_open), patch.object(trace, "_LIMIT", 2):
            trace.start()
            try:
                self.assertTrue(writing.wait(3))
                keys = ["a", "b"]
                trace.mark("forward", keys, ts_ns=1234567, step=3)
                keys.clear()
                trace.mark("second", "single")
                trace.mark("dropped", [])
            finally:
                release.set()
        first, second, loss = self.events()
        self.assertEqual(len(writer_tids), 1)
        self.assertNotEqual(writer_tids[0], tid)
        self.assertEqual((first['ph'], first['s'], first['ts'], first['tid']), ("i", "t", 1234.567, tid))
        self.assertEqual(first['args'], dict(keys=["a", "b"], step=3))
        self.assertEqual(second['args']['keys'], ["single"])
        self.assertEqual((loss['name'], loss['args']['dropped']), ("trace.queue_full", 1))
        trace.close()
        trace.mark("late", [])
        self.assertEqual(len(self.events()), 3)

    def test_host_worker_adapter_is_instance_local_and_preserves_failure(self):
        error = ValueError("native failure")
        receipt = object()

        class Manager:
            def _put_with_retry(self, *, request_ids, stage, batch_id):
                if stage == 'ready':
                    raise error
                return receipt

        manager = Manager()
        store = SimpleNamespace(_async_put_manager=manager)
        self.assertTrue(trace.instrument_mooncake_store(store))
        self.assertTrue(trace.instrument_mooncake_store(store))
        self.assertNotIn("_put_with_retry", vars(Manager()))
        keys = ["a", "b"]
        with ThreadPoolExecutor(1) as pool:
            worker_tid = pool.submit(threading.get_native_id).result()
            trace.mark("step_stop", keys)
            self.assertIs(pool.submit(manager._put_with_retry, stage="data", request_ids=keys, batch_id="B").result(), receipt)
            with self.assertRaises(ValueError) as raised:
                pool.submit(manager._put_with_retry, stage="ready", request_ids=keys, batch_id="B").result()
            self.assertIs(raised.exception, error)
        events = self.events()
        self.assertEqual(events[0]["args"]["reason"], "missing_submit_hook")
        stop, data, ready = [e for e in events if e["name"] != "mooncake.instrumentation_unavailable"]
        self.assertNotEqual(stop['tid'], worker_tid)
        self.assertLessEqual(stop['ts'], data['ts'])
        self.assertLessEqual(data['ts'] + data['dur'], ready['ts'])
        for event, stage in ((data, 'data'), (ready, 'ready')):
            self.assertEqual((event['pid'], event['tid'], event['ph']), (os.getpid(), worker_tid, "X"))
            self.assertEqual(event['name'], f"mooncake.{stage}_put")
            self.assertGreaterEqual(event['dur'], 0)
            self.assertEqual((event['args']['keys'], event['args']['batch_id']), (keys, "B"))
        self.assertEqual(ready['args']['error_type'], 'ValueError')

    def test_unsupported_host_or_gpu_direct_has_no_fake_put_or_d2h(self):
        for store in [SimpleNamespace(), SimpleNamespace(_gpu_direct_available=True, _gpu_send_buffer=object())]:
            before = vars(store).copy()
            self.assertFalse(trace.instrument_mooncake_store(store))
            self.assertEqual(vars(store), before)
        markers = self.events()
        self.assertEqual([e["name"] for e in markers], ["mooncake.instrumentation_unavailable"] * 2)
        self.assertEqual([e["args"]["reason"] for e in markers], ["missing_host_put_hook", "gpu_direct"])

    def test_concurrent_first_use(self):
        gate = threading.Barrier(4)

        def producer(index):
            gate.wait(timeout=5)
            for seq in range(50):
                trace.mark("batch", [index], seq=seq)

        with ThreadPoolExecutor(4) as pool:
            list(pool.map(producer, range(4)))
        events = self.events()
        self.assertEqual(len({e["tid"] for e in events}), 4)
        self.assertEqual(len(list(self.directory.glob("capture-*.jsonl"))), 1)
        for index in range(4):
            self.assertEqual([e["args"]["seq"] for e in events if e["args"]["keys"] == [index]], list(range(50)))

    def test_unwritable_directory_does_not_break_producer_or_close(self):
        self.recorder.directory = self.output
        self.output.touch()
        with self.assertLogs(trace.__name__, level="ERROR"):
            trace.mark("first", [])
            trace.close()
        trace.mark("late", [])
        self.assertEqual(self.output.read_bytes(), b"")

    def test_fork_atexit_and_cli(self):
        env = dict(os.environ, RTP_HIDDEN_STATE_CAPTURE_TRACE_DIR=str(self.directory),
                   PYTHONPATH=os.pathsep.join(os.path.abspath(p) for p in sys.path))
        code = '''
import os
from rtp_llm.utils import hidden_state_capture_trace as t
t.mark("parent", ["p"])
child = os.fork()
if child == 0:
    t.mark("child", ["c"])
    t.close()
    os._exit(0)
assert os.waitpid(child, 0)[1] == 0
'''
        subprocess.run([sys.executable, "-B", "-c", code], env=env, check=True, timeout=10)
        subprocess.run([sys.executable, "-B", "-m", trace.__name__, str(self.directory), str(self.output)],
                       env=env, check=True, timeout=10)
        events = json.loads(self.output.read_text())["traceEvents"]
        self.assertCountEqual([event['name'] for event in events], ['child', 'parent'])
        self.assertEqual(len({event['pid'] for event in events}), 2)
        self.assertEqual(len(list(self.directory.glob('capture-*.jsonl'))), 2)


@unittest.skipUnless(os.environ.get("RTP_TEST_TORCHSPEC_ARCHIVE"),
                     "optional AsyncPutManager integration: set RTP_TEST_TORCHSPEC_ARCHIVE")
class PinnedMooncakeTraceTest(_TraceFixture):
    def test_pinned_worker_wait_and_put_per_chunk_after_profiler_stop(self):
        archive = Path(os.environ["RTP_TEST_TORCHSPEC_ARCHIVE"]).resolve(strict=True)
        with ZipFile(archive) as zipped:
            self.assertEqual(zipped.comment, b"1b844317442f97464ed3d79677bbcb1a9f20c53e")
        # Use normal zip imports in a fresh interpreter. Do not modify the test
        # runner's sys.path or patch any loaded TorchSpec module's __path__.
        env = dict(os.environ, RTP_TEST_TORCHSPEC_ARCHIVE=str(archive), CUDA_VISIBLE_DEVICES="",
                   PYTHONPATH=os.pathsep.join([str(archive), *(os.path.abspath(p) for p in sys.path)]))
        subprocess.run([sys.executable, "-B", __file__, "PinnedMooncakeTraceTest.check_worker"],
                       env=env, check=True, timeout=90)

    def check_worker(self):
        from importlib.util import find_spec

        archive = Path(os.environ["RTP_TEST_TORCHSPEC_ARCHIVE"])
        spec = find_spec("torchspec")
        self.assertIsNotNone(spec)
        self.assertTrue(Path(spec.origin).is_relative_to(archive), spec.origin)

        def check_origins():
            for name, module in sys.modules.copy().items():
                if name == "torchspec" or name.startswith("torchspec."):
                    self.assertTrue(Path(module.__file__).is_relative_to(archive), name)

        check_origins()
        import torch
        from torchspec.transfer.mooncake.buffers import AsyncPutManager
        check_origins()

        calls = []

        class RawStore:
            def is_exist(self, key):
                return 0

            def batch_put_from(self, keys, ptrs, sizes):
                calls.append((list(keys), threading.get_native_id(), time.monotonic_ns()))
                return [0] * len(keys)

        release = threading.Event()
        fence_times = []

        class CopyFence:
            def synchronize(self):
                if not release.wait(timeout=10):
                    raise TimeoutError("worker not released after profiler stop")
                fence_times.append(time.monotonic_ns())

        manager = AsyncPutManager(RawStore())
        chunks = [["a", "b"], ["c"], ["d"]]

        def submit(i, fences):
            keys = chunks[i]
            ptrs = list(range(100, 100 + len(keys)))
            data = [k + "_hs" for k in keys]
            ready = [k + "_ready" for k in keys]
            manager.submit_batch(
                data, ptrs, [4] * len(keys), owner_buffer_ptrs=ptrs, ready_keys=ready,
                batch_id=f"B-chunk-{i}", request_ids=keys, wait_events=fences,
                ready_buffer_ptrs=ptrs, ready_sizes=[1] * len(keys), capture_source_keys=data + ready,
            )

        try:
            store = SimpleNamespace(_async_put_manager=manager)
            self.assertTrue(trace.instrument_mooncake_store(store))
            self.assertTrue(trace.instrument_mooncake_store(store))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]):
                submit(0, [CopyFence()])
            trace.mark("profiler.stopped", chunks[0])
            release.set()
            manager.drain()
            submit(1, [])  # CPU-only staging has no fence and must emit no D2H event.
            manager.drain()
            error = RuntimeError("copy fence failed")

            class FailedFence:
                def synchronize(self):
                    raise error

            submit(2, [FailedFence()])
            with self.assertRaises(RuntimeError) as raised:
                manager.drain()
            self.assertIs(raised.exception, error)
        finally:
            release.set()
            manager.shutdown()
        events = self.events()
        for i, keys in enumerate(chunks):
            chunk = [e for e in events if e["args"].get("batch_id") == f"B-chunk-{i}"]
            self.assertEqual([e["args"]["keys"] for e in chunk], [keys] * len(chunk))
            self.assertEqual([e["name"] for e in chunk],
                             (["mooncake.d2h_fence_handoff", "mooncake.d2h_fence_wait"] if i != 1 else [])
                             + (["mooncake.data_put", "mooncake.ready_put"] if i != 2 else []))
        stop = next(e for e in events if e["name"] == "profiler.stopped")
        handoff, wait, _, failed_wait = [e for e in events if "d2h_fence" in e["name"]]
        self.assertEqual(failed_wait["args"]["error_type"], "RuntimeError")
        self.assertEqual(handoff["tid"], stop["tid"])
        self.assertEqual(wait["tid"], calls[0][1])
        self.assertEqual(len(fence_times), 1)
        self.assertLessEqual(handoff["ts"], wait["ts"])
        self.assertLessEqual(wait["ts"], fence_times[0] / 1000)
        self.assertLessEqual(fence_times[0] / 1000, wait["ts"] + wait["dur"])
        puts = [e for e in events if e["name"] in ("mooncake.data_put", "mooncake.ready_put")]
        self.assertEqual(len(calls), len(puts))
        self.assertLessEqual(wait["ts"] + wait["dur"], puts[0]["ts"])
        for event, (native_keys, tid, timestamp) in zip(puts, calls):
            suffix = "_hs" if event["name"] == "mooncake.data_put" else "_ready"
            self.assertEqual(native_keys, [k + suffix for k in event["args"]["keys"]])
            self.assertEqual(event["tid"], tid)
            self.assertNotEqual(stop["tid"], tid)
            self.assertLessEqual(stop["ts"], event["ts"])
            self.assertLessEqual(event["ts"], timestamp / 1000)
            self.assertLessEqual(timestamp / 1000, event["ts"] + event["dur"])


if __name__ == "__main__":
    unittest.main()
