"""CPU regression for the native publisher. Run in a fresh process with GPUs hidden.

For optional pinned FP8 parity, run //rtp_llm/utils/test:hidden_state_file_store_archive_check.
Set TEST_UNDECLARED_OUTPUTS_DIR to retain the verified Chrome timeline.
"""
import errno
import hashlib
import importlib
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class FilePublisherCpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_available():
            raise RuntimeError('run the publisher CPU tests with CUDA_VISIBLE_DEVICES empty')
        kernel = 'rtp_llm.utils.hidden_state_file_fp8_kernel'
        cls.kernel_imports = []

        class BlockStoreDependencies:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == kernel:
                    cls.kernel_imports.append(fullname)
                    raise AssertionError('CPU publisher imported the file FP8 kernel')
                if fullname.split('.')[0] in ('torchspec', 'mooncake'):
                    raise AssertionError(f'file publisher imported {fullname}')
                return None

        if kernel in sys.modules or any(n.split('.')[0] in ('torchspec', 'mooncake') for n in sys.modules):
            raise RuntimeError('file publisher regression requires a fresh process')
        blocker = BlockStoreDependencies()
        sys.meta_path.insert(0, blocker)
        cls.addClassCleanup(sys.meta_path.remove, blocker)
        cls.native = importlib.import_module('rtp_llm.cpp.models.test.libth_hidden_state_file_publisher_cpu_test')
        cls.files = importlib.import_module('rtp_llm.utils.hidden_state_file_store')

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(self.assertEqual, self.kernel_imports, [])
        self.root = Path(tmp.name)
        env = patch.dict(os.environ, RTP_HIDDEN_STATE_CAPTURE_3FS_DIR=tmp.name,
                         RTP_HIDDEN_STATE_CAPTURE_3FS_LOCAL_PROTOCOL_TEST_ONLY='1',
                         MOONCAKE_STORE_KEY_NAMESPACE='CpuTest')
        env.start()
        self.addCleanup(env.stop)
        self.reader = self.files.FileCaptureReader(tmp.name)
        self.packed = torch.tensor([[1, 2, 3], [-4, 5, -6]], dtype=torch.float16)
        self.ids = torch.tensor([29, 31], dtype=torch.int64)

    def publisher(self, fp8=False, samples=1, **storage):
        publisher = self.native.CapturePublisher(samples=samples, fp8=fp8, layer_ids=[7, 2],
                                                  poll_interval_ms=60000, **storage)
        self.addCleanup(publisher.close)
        self.assertTrue(publisher.drain(5000))
        return publisher

    def submit(self, publisher, request_id, packed=None):
        return publisher.submit(self.packed if packed is None else packed, self.ids, torch.tensor([2], dtype=torch.int32),
                                torch.tensor([request_id], dtype=torch.int64), 2, 'file publisher CPU test')

    def publish(self, publisher, request_id, packed=None):
        admission = publisher.reserve(request_id, 2)
        self.assertTrue(admission.ok, admission.message)
        self.assertTrue(self.submit(publisher, request_id, packed), publisher.diagnostic)

    def assert_released(self, publisher):
        self.assertTrue(publisher.drain(5000))
        self.assertFalse(publisher.fatal, publisher.diagnostic)
        self.assertEqual((publisher.inflight_samples, publisher.inflight_bytes), (0, 0))

    def test_native_file_roundtrip_timeline(self):
        shards = self.root / 'trace'
        # The trace module snapshots its directory at import. This target owns the process.
        with patch.dict(os.environ, RTP_HIDDEN_STATE_CAPTURE_TRACE_DIR=str(shards)):
            trace = importlib.reload(importlib.import_module('rtp_llm.utils.hidden_state_capture_trace'))
            try:
                for fp8, request_id in [(False, 501), (True, 502)]:
                    key = f'rtp_CpuTest_{request_id}'
                    codec = 'zstd' if fp8 else 'none'
                    with patch.dict(os.environ, RTP_HIDDEN_STATE_CAPTURE_3FS_DIR=str(self.root / 'missing')):
                        options = dict(compression=codec, io_depth=3, io_workers=1, compression_level=2)
                        with patch.object(self.files, 'FileCaptureStore', wraps=self.files.FileCaptureStore) as factory:
                            publisher = self.publisher(fp8, storage_dir=str(self.root), **options)
                        factory.assert_called_once_with(str(self.root), **options)
                    ticket = publisher.reserve(request_id, 2, strict=True).reservation
                    self.assertTrue(self.submit(publisher, request_id))
                    self.assert_released(publisher)  # Before even opening the file.
                    self.assertEqual(ticket.publication(), (0, ''))
                    publisher.close()
                    with trace.span('consumer.read', [key]):
                        sample = self.reader.read(key)
                    self.assertEqual(sample['codec'], codec)
                    tensors = sample['tensors']
                    self.assertEqual(sample['metadata']['capture_layer_ids'], [7, 2])
                    torch.testing.assert_close(tensors['input_ids'], self.ids)
                    torch.testing.assert_close(tensors['last_hidden_states'], self.packed[:, 2:].bfloat16())
                    hidden = self.packed[:, :2].contiguous()
                    if fp8:
                        # One scalar per auxiliary layer: E4M3FN saturates to +/-448.
                        codes = torch.tensor([[126, 126], [254, 126]], dtype=torch.uint8)
                        scale = torch.tensor([[1., 2.], [4., 5.]]) / 448
                        torch.testing.assert_close(tensors['hidden_states'].view(torch.uint8), codes)
                        torch.testing.assert_close(tensors['hidden_states_scale'], scale, rtol=0, atol=0)
                    else:
                        self.assertNotIn('hidden_states_scale', tensors)
                        torch.testing.assert_close(tensors['hidden_states'], hidden.bfloat16())
                    self.assertEqual(sample['logical_bytes'], 40 if fp8 else 28)
                    trace.mark('consumer.verified', [key], dtype=str(tensors['hidden_states'].dtype),
                               logical_bytes=sample['logical_bytes'], file_bytes=sample['file_bytes'],
                               sha256=hashlib.sha256((self.root / (key + '.capture')).read_bytes()).hexdigest())
            finally:
                trace.close()
        timeline = self.root / 'timeline.json'
        trace.convert(shards, timeline)
        events = json.loads(timeline.read_text())['traceEvents']
        for request_id, writes in [(501, 3), (502, 4)]:
            key = f'rtp_CpuTest_{request_id}'
            chain = sorted((e for e in events if e['args']['keys'] == [key]), key=lambda e: e['ts'])
            expected = ['publisher.submit', 'publisher.dequeue', 'publisher.producer_ready', *['file.payload_write'] * writes,
                        'file.fsync', 'file.committed', 'publisher.flush_return', 'consumer.read', 'consumer.verified']
            self.assertEqual([e['name'] for e in chain if e['name'] in expected], expected)
            read = next(e for e in chain if e['name'] == 'consumer.read')
            submit, dequeue, ready = chain[:3]
            self.assertEqual(submit['tid'], threading.get_native_id())
            self.assertNotEqual(dequeue['tid'], submit['tid'])
            self.assertEqual(ready['tid'], dequeue['tid'])
            self.assertEqual(next(e for e in chain if e['name'] == 'publisher.flush_return')['tid'], dequeue['tid'])
            self.assertNotIn('file.d2h', [e['name'] for e in chain])
            self.assertEqual(read['ph'], 'X')
            self.assertNotEqual(read['tid'], next(e for e in chain if e['name'] == 'file.committed')['tid'])
        if out := os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR'):
            (Path(out) / 'timeline.json').write_bytes(timeline.read_bytes())

    @patch.dict(os.environ, RTP_HIDDEN_STATE_CAPTURE_FUSED_FP8='1')
    def test_fp8_fallback_packs_one_batch_before_splitting_requests(self):
        from rtp_llm.utils import hidden_state_file_contract as contract

        with patch.object(contract, 'quantize_aux_hidden_states', wraps=contract.quantize_aux_hidden_states) as quantize:
            publisher = self.publisher(fp8=True, samples=2)
            quantize.assert_not_called()
            packed = torch.arange(18, dtype=torch.float16).reshape(3, 6)[:, ::2]
            tickets = [publisher.reserve(i, n, strict=True).reservation for i, n in [(511, 1), (512, 2)]]
            self.assertTrue(publisher.submit(packed, torch.arange(3), torch.tensor([1, 2], dtype=torch.int32),
                                             torch.tensor([511, 512]), 3, 'FP8 batch split'))
            self.assert_released(publisher)
            quantize.assert_called_once()
            aux, layers = quantize.call_args.args
            self.assertEqual(layers, 2)
            self.assertTrue(aux.is_contiguous())
            torch.testing.assert_close(aux, packed[:, :2])
        codes, scales = contract.quantize_aux_hidden_states(aux, layers)
        for ticket, key, rows in zip(tickets, [511, 512], [slice(0, 1), slice(1, 3)]):
            self.assertEqual(ticket.publication(), (0, ''))
            fields = self.reader.read(f'rtp_CpuTest_{key}')['tensors']
            torch.testing.assert_close(fields['hidden_states'].view(torch.uint8), codes[rows].view(torch.uint8))
            torch.testing.assert_close(fields['hidden_states_scale'], scales[rows], rtol=0, atol=0)
            torch.testing.assert_close(fields['last_hidden_states'], packed[rows, 2:].bfloat16())

    def test_no_reservation_or_backpressure_is_not_publication(self):
        publisher = self.publisher(fp8=True)
        held = publisher.reserve(601, 2)
        self.assertTrue(held.ok)
        self.assertFalse(publisher.reserve(602, 2).ok)
        self.assertFalse(self.submit(publisher, 602))
        held.reservation.cancel()
        self.assert_released(publisher)
        self.assertEqual(self.reader.keys(), [])
        with self.assertRaises(FileNotFoundError):
            self.reader.read('rtp_CpuTest_602')  # The low-level reader probes once, not a wait API.

    def test_accepted_write_failure_releases_quota_without_reader(self):
        publisher = self.publisher()
        entered = threading.Event()
        resume = threading.Event()

        def full_disk(*_):
            entered.set()
            if not resume.wait(5):
                raise AssertionError('test did not release the writer')
            raise OSError(errno.ENOSPC, 'full disk: /private/spool/rtp_CpuTest_702.capture')

        with patch.object(os, 'fsync', full_disk):
            try:
                receipt = publisher.reserve(701, 2, strict=True).reservation
                self.assertTrue(self.submit(publisher, 701))
                self.assertTrue(entered.wait(5))
                self.assertEqual(publisher.inflight_samples, 1)
                self.assertGreater(publisher.inflight_bytes, 0)
                denied = publisher.reserve(702, 2, strict=True)
                self.assertFalse(denied.ok)
                self.assertIsNone(denied.reservation)
                receipt.cancel()  # Stream/KV release must not cancel the submitted write.
                self.assertEqual(receipt.publication(), 'pending')
            finally:
                resume.set()
            self.assert_released(publisher)
        self.assertEqual(receipt.publication(), (606, 'file capture publication failed'))
        self.assertIn('/private/spool/rtp_CpuTest_702.capture', publisher.diagnostic)
        self.assertIn('full_disk', publisher.diagnostic)
        self.assertEqual(self.reader.keys(), [])
        self.publish(publisher, 702)
        self.assert_released(publisher)
        key = 'rtp_CpuTest_702'
        self.publish(publisher, 702, -self.packed)  # Same name, different bytes; never overwrite.
        self.assert_released(publisher)
        self.assertIn('duplicate', publisher.diagnostic)
        torch.testing.assert_close(self.reader.read(key)['tensors']['hidden_states'], self.packed[:, :2].bfloat16())

    def test_receipt_early_abort_shutdown_and_request_id_reuse(self):
        publisher = self.publisher()
        old = publisher.reserve(900, 2, strict=True).reservation
        publisher.discard(900)
        self.assertEqual(old.publication(), (606, 'file capture rejected before submit'))
        new = publisher.reserve(900, 2, strict=True).reservation
        old.cancel()
        self.assertEqual(new.publication(), 'pending')
        self.assertTrue(self.submit(publisher, 900))
        self.assert_released(publisher)
        self.assertEqual(new.publication(), (0, ''))
        aborted = publisher.reserve(901, 2, strict=True).reservation
        aborted.cancel()
        self.assertIn('cancelled before submit', aborted.publication()[1])
        ordinary = publisher.reserve(902, 2).reservation
        self.assertIsNone(ordinary.publication())
        ordinary.cancel()
        shutdown = publisher.reserve(903, 2, strict=True).reservation
        publisher.close()
        self.assertIn('stopped before submit', shutdown.publication()[1])
        self.assertFalse(publisher.reserve(904, 2, strict=True).ok)

    def test_close_drains_accepted_publications_and_dropped_queue_entry(self):
        publisher = self.publisher(samples=3, batches=1)
        entered, resume = threading.Event(), threading.Event()
        fsync = os.fsync

        def hold_fsync(fd):
            entered.set()
            if not resume.wait(10):
                raise AssertionError('close test did not release fsync')
            return fsync(fd)

        closer = threading.Thread(target=publisher.close)
        with patch.object(os, 'fsync', hold_fsync):
            first, queued, dropped = [publisher.reserve(i, 2, strict=True).reservation for i in (911, 912, 913)]
            try:
                self.assertTrue(self.submit(publisher, 911))
                self.assertTrue(entered.wait(5))
                self.assertTrue(self.submit(publisher, 912))
                self.assertFalse(self.submit(publisher, 913))
                closer.start()
                self.assertTrue(publisher.wait_stopping())
                self.assertFalse(publisher.reserve(914, 2, strict=True).ok)
                self.assertTrue(closer.is_alive())
                self.assertEqual(first.publication(), 'pending')
                self.assertEqual(queued.publication(), 'pending')
            finally:
                resume.set()
                if closer.ident is not None:
                    closer.join(10)
        self.assertFalse(closer.is_alive())
        self.assertEqual(first.publication(), (0, ''))
        self.assertEqual(queued.publication(), (0, ''))
        self.assertEqual(dropped.publication(), (8107, 'file capture queue full'))
        self.assertEqual(publisher.inflight_samples, 0)
        self.assertEqual(sorted(self.reader.keys()), ['rtp_CpuTest_911', 'rtp_CpuTest_912'])

    def test_token_and_publisher_destructors_retire_unsubmitted_generation(self):
        publisher = self.native.CapturePublisher(samples=1)
        self.assertTrue(publisher.drain(5000))
        abandoned = publisher.reserve(915, 2, strict=True).reservation
        self.assertIsNotNone(abandoned)
        del abandoned
        self.assertEqual(publisher.inflight_samples, 0)
        surviving = publisher.reserve(915, 2, strict=True).reservation
        del publisher
        self.assertEqual(surviving.publication(), (8100, 'file capture publisher stopped before submit'))
        surviving.cancel()

    def submit_pair(self, publisher, first, strict=False):
        reservations = [publisher.reserve(i, 2, strict) for i in (first, first + 1)]
        self.assertTrue(all(r.ok for r in reservations))
        self.assertTrue(publisher.submit(-self.packed.repeat(2, 1), self.ids.repeat(2),
                                         torch.tensor([2, 2], dtype=torch.int32),
                                         torch.tensor([first, first + 1]), 4, 'mixed batch'))
        return [r.reservation for r in reservations]

    def test_mixed_batch_attributes_duplicate_to_second_key(self):
        publisher = self.publisher(samples=2)
        self.publish(publisher, 712)
        self.assert_released(publisher)
        first, second = self.submit_pair(publisher, 711, strict=True)
        self.assert_released(publisher)
        self.assertEqual(first.publication(), (0, ''))
        self.assertEqual(second.publication(), (606, 'file capture publication failed'))
        self.assertIn("committed_keys=('rtp_CpuTest_711',)", publisher.diagnostic)
        self.assertIn('first uncommitted key=rtp_CpuTest_712', publisher.diagnostic)
        self.assertIn('duplicate', publisher.diagnostic)
        for request_id, sign in ((711, -1), (712, 1)):
            actual = self.reader.read(f'rtp_CpuTest_{request_id}')['tensors']['hidden_states']
            torch.testing.assert_close(actual, sign * self.packed[:, :2].bfloat16())

    def test_mixed_batch_preserves_typed_fatal_cause(self):
        publisher = self.publisher(samples=2)
        with patch.object(os, 'fsync', side_effect=torch.distributed.DistError('injected typed fault')):
            receipts = self.submit_pair(publisher, 721, strict=True)
            self.assertTrue(publisher.drain(5000))
        self.assertTrue(publisher.fatal, publisher.diagnostic)
        self.assertIn('first uncommitted key=rtp_CpuTest_721', publisher.diagnostic)
        self.assertFalse(publisher.reserve(723, 2).ok)
        for receipt in receipts:
            self.assertEqual(receipt.publication(), (606, 'file capture publisher failed before confirming publication'))

    def test_hardlinked_prefix_succeeds_even_when_later_key_is_fatal(self):
        publisher = self.publisher(samples=2)
        prepare = self.files.FileCaptureStore._prepare

        def fail_second(store, directory, key, *args):
            if key == 'rtp_CpuTest_742':
                raise torch.distributed.DistError('typed suffix fault')
            return prepare(store, directory, key, *args)

        with patch.object(self.files.FileCaptureStore, '_prepare', fail_second):
            first, second = self.submit_pair(publisher, 741, strict=True)
            self.assertTrue(publisher.drain(5000))
        self.assertTrue(publisher.fatal, publisher.diagnostic)
        self.assertEqual(first.publication(), (0, ''))
        self.assertEqual(second.publication(), (606, 'file capture publisher failed before confirming publication'))
        self.assertEqual(self.reader.keys(), ['rtp_CpuTest_741'])
        torch.testing.assert_close(self.reader.read('rtp_CpuTest_741')['tensors']['input_ids'], self.ids)
        self.assertEqual(publisher.inflight_samples, 2)  # fatal quota remains quarantined

    def test_additional_future_fatal_overrides_duplicate_and_preserves_prefix(self):
        publisher = self.publisher(samples=3, io_depth=3)
        self.publish(publisher, 752)  # The head failure is a hardlink collision, not a device fault.
        self.assert_released(publisher)
        prepare = self.files.FileCaptureStore._prepare

        def fail_last(store, directory, key, *args):
            if key == 'rtp_CpuTest_753':
                raise torch.distributed.DistError('joined fault')
            return prepare(store, directory, key, *args)

        with patch.object(self.files.FileCaptureStore, '_prepare', fail_last):
            receipts = [publisher.reserve(i, 2, strict=True).reservation for i in (751, 752, 753)]
            self.assertTrue(publisher.submit(-self.packed.repeat(3, 1), self.ids.repeat(3),
                                             torch.tensor([2, 2, 2], dtype=torch.int32),
                                             torch.tensor([751, 752, 753]), 6, 'joined exceptions'))
            self.assertTrue(publisher.drain(5000))
        self.assertTrue(publisher.fatal, publisher.diagnostic)
        self.assertIn('joined fault', publisher.diagnostic)
        self.assertEqual(receipts[0].publication(), (0, ''))
        for item in receipts[1:]:
            self.assertEqual(item.publication(), (606, 'file capture publisher failed before confirming publication'))
        self.assertFalse(publisher.reserve(754, 2, strict=True).ok)
        self.assertEqual(sorted(self.reader.keys()), ['rtp_CpuTest_751', 'rtp_CpuTest_752'])
        for request_id, sign in ((751, -1), (752, 1)):
            actual = self.reader.read(f'rtp_CpuTest_{request_id}')['tensors']['hidden_states']
            torch.testing.assert_close(actual, sign * self.packed[:, :2].bfloat16())

    def test_io_error_during_prefix_drain_cannot_mask_a_device_fatal(self):
        publisher = self.publisher(samples=2)
        self.publish(publisher, 731)
        self.assert_released(publisher)
        stage = self.files._stage

        def fail_second(fields, index, *args):
            if index == 1:
                raise torch.distributed.DistError('masked stage fault')
            return stage(fields, index, *args)

        with patch.object(self.files, '_stage', fail_second):
            self.submit_pair(publisher, 731)
            self.assertTrue(publisher.drain(5000))
        self.assertTrue(publisher.fatal, publisher.diagnostic)
        self.assertIn('masked stage fault', publisher.diagnostic)
        self.assertEqual(self.reader.keys(), ['rtp_CpuTest_731'])

    def test_device_fatal_is_not_a_recoverable_file_error(self):
        publisher = self.publisher()
        with patch.object(self.files.FileCaptureStore, 'put_batch', side_effect=RuntimeError('CUDA error: illegal address')):
            self.publish(publisher, 801)
            self.assertTrue(publisher.drain(5000))
        self.assertTrue(publisher.fatal)
        self.assertFalse(publisher.reserve(802, 2).ok)
        self.assertEqual(publisher.inflight_samples, 1)
        self.assertEqual(self.reader.keys(), [])


if __name__ == '__main__':
    unittest.main()
