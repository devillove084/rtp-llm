"""File FP8 routing and byte-level conformance; CUDA cases never skip."""

import os
import unittest

import torch

from rtp_llm.utils.hidden_state_file_contract import quantize_aux_hidden_states


def _fixture(tokens, layers, width, gap, step=1, dtype=torch.bfloat16):
    # A request starts part-way into a batch. Even with no padding the aux
    # narrow has a whole final vector between adjacent tokens' aux rows.
    row = (layers + 1) * width
    backing = torch.full((tokens + 2, row * step + gap), 32768, dtype=dtype)
    packed = backing[1:1 + tokens, gap // 2:gap // 2 + row * step:step]
    aux = packed[:, :layers * width].unflatten(1, (layers, width))
    rng = torch.Generator().manual_seed(4501)
    aux.copy_(torch.randn(aux.shape, generator=rng))
    aux[0].zero_()
    aux[1].mul_(2**-20 if dtype == torch.float16 else 2**-48)
    # With scale exactly 1 these midpoints round both down and up to even;
    # the sole maximum in the final lane must participate in the reduction.
    aux[2].zero_()
    aux[2, :, :6] = torch.tensor([1.0625, 1.1875, -1.0625, -1.1875, 2**-10, -0.0])
    aux[2, :, -1] = 448
    # Distinguish layer permutations and wrong token strides after the edge cases.
    aux[3:].mul_(torch.arange(1, layers + 1).view(1, layers, 1))
    # ATen CUDA divides by the scalar bound using a rounded reciprocal. With
    # amax=3 this changes the scale by one ULP and flips this FP8 midpoint.
    aux[3].zero_()
    aux[3, :, :2] = torch.tensor([3 * 2**-16, -3 * 2**-16])
    aux[3, :, -1] = 3
    # Separate vectors: NaN poisons amax, whereas infinities produce inf
    # scales and both finite/inf and inf/inf quotients must be encoded.
    aux[5, :, -1] = float("inf")
    aux[6, :, -1] = -float("inf")
    aux[5:7, :, 0] = -0.0
    final = packed[:, layers * width:]
    final.copy_(torch.randn(final.shape, generator=rng))
    final[:, :7] = torch.tensor([-0., float("inf"), -float("inf"), 1 + 2**-8,
                                  1 + 3 * 2**-8, -1 - 2**-8, -1 - 3 * 2**-8])
    # Set distinct signed NaN payloads without an intermediate host cast.
    bits, nan_bits = {
        torch.bfloat16: (torch.int16, (32705, -61)),
        torch.float16: (torch.int16, (0x7e01, -509)),
        torch.float32: (torch.int32, (0x7fc00001, -4194301)),
    }[dtype]
    payloads = torch.tensor(nan_bits, dtype=bits)
    final.view(bits)[:, 7:9] = payloads
    # Both payloads compete in the same amax reduction. Reverse lane order
    # across layers; the device file contract, not the fixture, selects the bytes.
    nan_rows = aux[4].view(bits)
    nan_rows[::2, [1, -1]] = payloads
    nan_rows[1::2, [1, -1]] = payloads.flip(0)
    return backing, packed


class CpuRouteTest(unittest.TestCase):
    def test_supported_layout_on_cpu_declines(self):
        from rtp_llm.utils.hidden_state_file_fp8_kernel import try_pack_file_fp8

        for gap in (0, 11):
            with self.subTest(gap=gap):
                _, packed = _fixture(7, 5, 256, gap)
                self.assertIsNone(try_pack_file_fp8(packed, 5))


class CudaContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("TRITON_INTERPRET") == "1":
            raise RuntimeError("CUDA contract target must not run through the Triton interpreter")
        if torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError("file FP8 CUDA contract test requires a real NVIDIA GPU")
        sm = torch.cuda.get_device_capability()
        expected = os.environ.get("FILE_FP8_TEST_SM")
        if expected and expected != f"{sm[0]}.{sm[1]}":
            raise RuntimeError(f"file FP8 target requires SM {expected}, got {sm} on {torch.cuda.get_device_name()}")
        if sm not in ((9, 0), (10, 0), (12, 0)):
            raise RuntimeError(f"unsupported file FP8 device: {torch.cuda.get_device_name()} {sm}; "
                               "expected SM90, SM100 or SM120")

    def test_ineligible_cuda_layout_declines(self):
        from rtp_llm.utils.hidden_state_file_fp8_kernel import try_pack_file_fp8

        packed = torch.empty((7, 384), dtype=torch.bfloat16, device="cuda")
        for value, layers in ((packed, 0), (packed[:, :-1], 2), (packed[:0], 2),
                              (packed.flatten(), 2), (packed.to(torch.int32), 2)):
            with self.subTest(shape=value.shape, layers=layers, dtype=value.dtype):
                self.assertIsNone(try_pack_file_fp8(value, layers))

    def test_packed_file_bytes(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for tokens, layers, width, gap, step in (
                (9, 2, 128, 0, 1), (17, 5, 257, 13, 1), (17, 5, 257, 13, 2)
            ):
                with self.subTest(dtype=dtype, layers=layers, width=width, step=step):
                    self._check_packed(tokens, layers, width, gap, step, dtype)

    def _check_packed(self, tokens, layers, width, gap, step, dtype):
        from rtp_llm.utils.hidden_state_file_fp8_kernel import try_pack_file_fp8

        backing, view = _fixture(tokens, layers, width, gap, step, dtype)
        # Transfer the backing, not the view: .cuda() on the narrow could
        # erase the request offset and pack away both row and column gaps.
        packed = backing.cuda().as_strided(view.size(), view.stride(), view.storage_offset())
        self.assertEqual(packed.is_contiguous(), gap == 0 and step == 1)
        self.assertEqual(packed.stride(1), step)
        aux = packed[:, :layers * width].contiguous()
        codes, scales = quantize_aux_hidden_states(aux, layers)
        self.assertTrue(codes.is_cuda and scales.is_cuda)
        self.assertTrue((scales[3].view(torch.int32) == 0x3bdb6db8).all().item(),
                        "CUDA oracle did not use the rounded reciprocal scale for amax=3")
        pair = codes[3].view(torch.uint8).reshape(layers, width)[:, :2]
        expected_pair = torch.tensor([0x03, 0x83], device=packed.device, dtype=torch.uint8)
        torch.testing.assert_close(pair, expected_pair.expand_as(pair), rtol=0, atol=0)
        final = packed[:, layers * width:].to(torch.bfloat16).contiguous()
        result = try_pack_file_fp8(packed, layers)
        self.assertIsInstance(result, tuple, f"eligible input declined on {torch.cuda.get_device_name()}")
        self.assertEqual(len(result), 3)
        for field, actual, ref in zip(("codes", "scales", "final"), result, (codes, scales, final)):
            with self.subTest(field=field, dtype=dtype, layers=layers, width=width):
                self.assertEqual(actual.device, packed.device)
                self.assertEqual(actual.dtype, ref.dtype)
                self.assertEqual(actual.shape, ref.shape)
                self.assertTrue(actual.is_contiguous())
                torch.testing.assert_close(actual.view(torch.uint8), ref.view(torch.uint8), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
