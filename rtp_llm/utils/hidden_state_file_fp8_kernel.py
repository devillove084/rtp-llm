"""Optional single-launch packing for the FP8 hidden-state file layout."""

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _file_fp8_kernel():
    try:
        import triton
    except ModuleNotFoundError as error:
        if error.name == "triton":
            return None
        raise
    import triton.language as tl

    @triton.jit
    def kernel(src, codes, scales, final, row_stride, col_stride,
               L: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr):
        token = tl.program_id(0).to(tl.int64)
        slot = tl.program_id(1).to(tl.int64)
        h = tl.arange(0, BLOCK).to(tl.int64)
        x = tl.load(src + token * row_stride + (slot * H + h) * col_stride,
                    mask=h < H, other=0)
        if slot == L:
            # A BF16 input already has the output pointer's dtype: no FP round trip.
            tl.store(final + token * H + h, x.to(tl.bfloat16), mask=h < H)
        else:
            values = x.to(tl.float32)
            # Positive FP32 bit patterns sort by magnitude, with NaNs above +inf.
            # Unlike tl.max on floats this matches torch.amax NaN propagation.
            magnitude_bits = tl.abs(values).to(tl.uint32, bitcast=True)
            amax = tl.max(magnitude_bits, 0).to(tl.float32, bitcast=True)
            # ATen CUDA div_(Python scalar) multiplies by float32(1.0 / bound);
            # values / scale below is tensor/tensor and must still divide.
            scale = tl.where(amax < 1e-12, 1e-12, amax) * (1.0 / 448.0)
            normalized = tl.div_rn(values, scale)
            bounded = tl.clamp(normalized, -448.0, 448.0)
            index = token * L + slot
            tl.store(scales + index, scale)
            code = bounded.to(tl.float8e4nv, fp_downcast_rounding="rtne")
            # ATen clamp returns its NaN input unchanged. Float8_e4m3fn(value)
            # encodes the sign of that quotient, not the sign of the source.
            bits = normalized.to(tl.uint32, bitcast=True)
            nan_code = ((bits >> 24) | 0x7f).to(tl.uint8)
            byte = tl.where((bits & 0x7fffffff) > 0x7f800000, nan_code, code.to(tl.uint8, bitcast=True))
            tl.store(codes + index * H + h, byte.to(tl.float8e4nv, bitcast=True), mask=h < H)

    return kernel


def try_pack_file_fp8(packed: torch.Tensor, num_layers: int):
    """Return compact (E4M3FN codes, FP32 scales, BF16 final) or None for fallback."""
    if (
        not isinstance(packed, torch.Tensor)
        or not packed.is_cuda
        or torch.version.cuda is None
        or torch.version.hip is not None
        or packed.layout != torch.strided
        or packed.is_nested
        or packed.is_neg()
        or packed.ndim != 2
        or packed.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or type(num_layers) is not int
        or not 1 <= num_layers < 65535
    ):
        return None
    tokens, width = packed.shape
    hidden, remainder = divmod(width, num_layers + 1)
    # One block covers one slot; bound both CUDA grid dimensions and the reduction tile.
    if remainder or not 0 < tokens < 2**31 or not 0 < hidden <= 65536:
        return None
    # Only targets checked for direct FP32 -> E4M3 RTNE (SM89 goes via FP16).
    if torch.cuda.get_device_capability(packed.device) not in ((9, 0), (10, 0), (12, 0)):
        return None
    kernel = _file_fp8_kernel()
    if kernel is None:
        return None

    codes = packed.new_empty((tokens, num_layers * hidden), dtype=torch.float8_e4m3fn)
    scales = packed.new_empty((tokens, num_layers), dtype=torch.float32)
    final = packed.new_empty((tokens, hidden), dtype=torch.bfloat16)
    block = 1 << (hidden - 1).bit_length()
    with torch.cuda.device(packed.device):
        kernel[(tokens, num_layers + 1)](
            packed, codes, scales, final, *packed.stride(), num_layers, hidden, block,
            num_warps=4 if block <= 2048 else 8,
        )
    return codes, scales, final
