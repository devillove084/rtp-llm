"""Opt-in host timeline for hidden-state capture (independent of step profiling).

Set RTP_HIDDEN_STATE_CAPTURE_TRACE_DIR before import; use a fresh directory per
run on one host. mark()/span() snapshot the batch keys; pass only immutable,
JSON-serializable metadata, never tensors. Explicit timestamps must use
time.monotonic_ns() (CLOCK_MONOTONIC on Linux), not wall time or CUDA time.

Producers only enqueue. Use start() during setup to keep writer startup off
the forward path; otherwise the first event starts it. Call close() after all
publisher/store workers drain, especially in workers which use os._exit(). Normal interpreter exit drains automatically. An abrupt exit
can lose the queued tail. Merge the resulting per-process JSONL files with:

  python3 -m rtp_llm.utils.hidden_state_capture_trace DIR timeline.json
"""

import contextlib
import json
import logging
import os
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

_DIR = os.getenv("RTP_HIDDEN_STATE_CAPTURE_TRACE_DIR")
_LIMIT = 8192
_NO_SPAN = contextlib.nullcontext()
_CATEGORY = "hidden_state_capture"


def _event(name, keys, start, args, origin_tid=None, **phase):
    args["keys"] = [keys] if isinstance(keys, str) else list(keys)
    return dict(
        name=name, cat=_CATEGORY, pid=os.getpid(), tid=threading.get_native_id() if origin_tid is None else origin_tid,
        ts=start / 1000, args=args, **phase,
    )


class _Recorder:
    def __init__(self, directory):
        self.directory = directory
        self.changed = threading.Condition()
        self.pending = deque()
        self.dropped = 0
        self.thread = None
        self.closed = False

    def start(self):
        with self.changed:
            if self.thread is not None or self.closed:
                return
            thread = threading.Thread(target=self._write, name="capture-trace", daemon=True)
            try:
                thread.start()
            except RuntimeError:
                self.closed = True
                return
            self.thread = thread

    def enqueue(self, event):
        if self.thread is None:
            self.start()
        with self.changed:
            if self.closed:
                return
            if len(self.pending) >= _LIMIT:
                self.dropped += 1
                return
            self.pending.append(event)
            if len(self.pending) == 1:
                self.changed.notify()

    def _write(self):
        try:
            Path(self.directory).mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.directory,
                prefix=f"capture-{os.getpid()}-", suffix=".jsonl", delete=False,
            ) as stream:
                while True:
                    with self.changed:
                        self.changed.wait_for(lambda: self.pending or self.closed)
                        if not self.pending:
                            break
                        events, self.pending = self.pending, deque()
                        dropped, self.dropped = self.dropped, 0
                    if dropped:
                        events.append(_event(
                            "trace.queue_full", [], time.monotonic_ns(), {"dropped": dropped}, ph="i", s="t"
                        ))
                    for event in events:
                        try:
                            line = json.dumps(event, allow_nan=False, separators=(",", ":"))
                        except (ValueError, TypeError, RecursionError):
                            logging.getLogger(__name__).warning("Skipping non-JSON capture trace event: %s", event["name"])
                            continue
                        stream.write(line + "\n")
                    stream.flush()
        except OSError:
            with self.changed:
                self.closed = True
                self.pending.clear()
            logging.getLogger(__name__).exception("Capture trace writer stopped")

    def close(self):
        with self.changed:
            self.closed = True
            self.changed.notify_all()
            thread = self.thread
        if thread is not None:
            thread.join()


def _reset():
    global _recorder
    _recorder = _Recorder(_DIR) if _DIR else None


_reset()


def start():
    """Optional: start the writer during setup rather than at the first event."""
    if _recorder is not None:
        _recorder.start()


def enabled():
    """Guard expensive metadata construction at call sites. No environment lookup."""
    return _recorder is not None and not _recorder.closed


def mark(name, keys, ts_ns=None, *, origin_tid=None, **args):
    """For a deferred mark pass the originating native TID and monotonic timestamp."""
    if _recorder is None or _recorder.closed:
        return
    _recorder.enqueue(_event(name, keys, time.monotonic_ns() if ts_ns is None else ts_ns,
                             args, origin_tid=origin_tid, ph="i", s="t"))


@contextlib.contextmanager
def _span(name, keys, args):
    # Snapshot before yielding: callers may reuse the batch list inside the span.
    start_ns = time.monotonic_ns()
    event = _event(name, keys, start_ns, args, ph="X")
    try:
        yield
    except BaseException as error:
        args["error_type"] = type(error).__name__
        raise
    finally:
        end = time.monotonic_ns()
        if event["pid"] == os.getpid():
            event["dur"] = (end - start_ns) / 1000
            _recorder.enqueue(event)


def span(name, keys, **args):
    """Time a synchronous host scope on the calling thread, including failures."""
    if _recorder is None or _recorder.closed:
        return _NO_SPAN
    return _span(name, keys, args)


def close():
    """Drain and permanently stop this process's recorder. Idempotent."""
    if _recorder is not None:
        _recorder.close()


if _DIR:
    import atexit

    atexit.register(close)
    if hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=_reset)


def _install(manager, name, required_kwargs, wrapper):
    import inspect
    import weakref

    original = getattr(manager, name, None)
    if getattr(original, "_rtp_hidden_state_trace", False) is True:
        return True
    if not inspect.ismethod(original):
        return False
    try:
        params = inspect.signature(original).parameters
        if not all(k in params and params[k].kind == inspect.Parameter.KEYWORD_ONLY for k in required_kwargs):
            return False
        # Avoid a cycle keeping the executor/native store alive after shutdown.
        ref = weakref.WeakMethod(original)

        def observed(*args, **kwargs):
            return wrapper(ref(), args, kwargs)

        observed._rtp_hidden_state_trace = True
        setattr(manager, name, observed)
    except (AttributeError, TypeError, ValueError):
        return False
    return True


def _observe_put(method, args, kwargs):
    stage = kwargs.get("stage")
    if stage in ("data", "ready") and kwargs.get("request_ids"):
        with span(f"mooncake.{stage}_put", kwargs["request_ids"], batch_id=kwargs.get("batch_id")):
            return method(*args, **kwargs)
    return method(*args, **kwargs)


class _CopyFence:
    # The pinned host worker only calls synchronize() on the submitted events.
    def __init__(self, fence, keys, batch_id, index):
        self.fence = fence
        self.keys = keys
        self.batch_id = batch_id
        self.index = index

    def synchronize(self):
        with span("mooncake.d2h_fence_wait", self.keys, batch_id=self.batch_id, fence_index=self.index):
            return self.fence.synchronize()


def _observe_submit(method, args, kwargs):
    fences = kwargs.get("wait_events")
    keys = kwargs.get("request_ids")
    if enabled() and keys and isinstance(fences, (list, tuple)) and fences:
        keys = tuple(keys)
        batch_id = kwargs.get("batch_id")
        # Eagle records copy_done on the copy stream before this call. This is
        # the host handoff, NOT the instant a copy started or finished on GPU.
        kwargs["wait_events"] = [_CopyFence(f, keys, batch_id, i) for i, f in enumerate(fences)]
        mark("mooncake.d2h_fence_handoff", keys, batch_id=batch_id, fences=len(fences))
    return method(*args, **kwargs)


def instrument_mooncake_store(store):
    """Install on a dedicated store after warmup, before the first submit.

    True means host data/ready PUT scopes are observable (including retries and
    status validation). D2H coverage is separate: only real fence handoff and
    host synchronize calls are observed, never extra waits or GPU DMA times.
    CPU-only staging has no copy fence. Unsupported hooks emit a capability
    marker rather than invented stage events. No TorchSpec import/class patch.
    """
    if not enabled():
        return False
    if getattr(store, "_gpu_direct_available", False) and getattr(store, "_gpu_send_buffer", None) is not None:
        mark("mooncake.instrumentation_unavailable", [], reason="gpu_direct", scope="host_put_and_d2h_fence")
        return False
    manager = getattr(store, "_async_put_manager", None)
    if not _install(manager, "_put_with_retry", ("request_ids", "stage", "batch_id"), _observe_put):
        mark("mooncake.instrumentation_unavailable", [], reason="missing_host_put_hook", scope="host_put_and_d2h_fence")
        return False
    if not _install(manager, "submit_batch", ("request_ids", "wait_events", "batch_id"), _observe_submit):
        mark("mooncake.instrumentation_unavailable", [], reason="missing_submit_hook", scope="d2h_fence")
    return True


def convert(directory, output):
    """Stream shards into a Chrome trace. Only convert after all writers drain."""
    shards = sorted(Path(directory).glob("capture-*.jsonl"))
    if not shards:
        raise ValueError(f"no capture trace shards in {directory}")
    if Path(output).suffix == ".jsonl":
        raise ValueError("choose a .json timeline, not a shard")
    with open(output, "w", encoding="utf-8") as dest:
        dest.write('{"traceEvents":[')
        separator = ""
        for shard in shards:
            with shard.open(encoding="utf-8") as source:
                for line in source:
                    # A killed process may have left a partial final record.
                    if not line.endswith("\n"):
                        break
                    dest.write(separator + line.rstrip("\n"))
                    separator = ",\n"
        dest.write("]}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge hidden-state capture trace shards (same host/run)")
    parser.add_argument("directory")
    parser.add_argument("output")
    options = parser.parse_args()
    convert(options.directory, options.output)
