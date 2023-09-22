import torch
import triton
import triton.language as tl

def convert_float_to_float32(fp: torch.tensor, dtype=None):
    if not dtype:
        dtype = getattr(tl, torch_dtype_name(fp.dtype))

    fp = fp.view(getattr(torch, f"int{dtype.primitive_bitwidth}"))
    exp_width = dtype.primitive_bitwidth - dtype.fp_mantissa_width - 1
    exp_bias = dtype.exponent_bias
    sign = ((fp >> (dtype.primitive_bitwidth - 1)) & 0x01).int()
    exp = ((fp >> dtype.fp_mantissa_width) & ((1 << exp_width) - 1)).int()
    frac = (fp & ((1 << dtype.fp_mantissa_width) - 1)).int()

    output = torch.where(exp == 0,
                         # subnormal
                         ((-1.0) ** sign) * (2.0 ** (1 - exp_bias)) * (frac / (2.0 ** dtype.fp_mantissa_width)),
                         # normal
                         ((-1.0) ** sign) * (2.0 ** (exp - exp_bias)) * (1.0 + frac / (2.0 ** dtype.fp_mantissa_width))).float()

    extended_exp = ((1 << (tl.float32.primitive_bitwidth - tl.float32.fp_mantissa_width - 1)) - 1) << tl.float32.fp_mantissa_width
    # special cases, exp is 0b11..1
    if dtype in [tl.float8e4, tl.float8e4b15]:
        # float8e4m3 does not have infinities
        output[fp == 0b01111111] = torch.nan
        output[fp == 0b11111111] = torch.nan
    else:
        output = torch.where(exp == (1 << exp_width) - 1,
                             ((sign << (tl.float32.primitive_bitwidth - 1)) | extended_exp | (frac << (tl.float32.fp_mantissa_width - dtype.fp_mantissa_width))).view(torch.float32),
                             output)
    return output


def test_abs_fp8(in_dtype, device):
    @triton.jit
    def abs_kernel(X, Z, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        z = tl.abs(x)
        tl.store(Z + off, z)

    f8_tensor = torch.tensor(range(-128, 128), dtype=torch.int8, device=device)
    # f32_to_f8 doesn't handle nan, so we make sure f8_tensor doesn't contain any nan
    all_exp_ones = (f8_tensor & 0b01111100) == 128 - 2**in_dtype.fp_mantissa_width
    f8_tensor[all_exp_ones] = 0
    f8 = triton.reinterpret(f8_tensor, in_dtype)
    n_elements = f8_tensor.numel()
    out_f8 = torch.empty_like(f8_tensor)
    abs_kernel[(1,)](f8, triton.reinterpret(out_f8, in_dtype), n_elements)

    f32_tensor = convert_float_to_float32(f8_tensor, in_dtype)
    expect = f32_tensor.abs()
    actual_f8 = convert_float_to_float32(out_f8, in_dtype)
    torch.testing.assert_close(actual_f8, expect)

def test_add_fp8(in_dtype, device):
    @triton.jit
    def abs_kernel(X, Z, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        z = tl.abs(x)
        tl.store(Z + off, z)

    f8_tensor = torch.tensor(range(-128, 128), dtype=torch.int8, device=device)
    # f32_to_f8 doesn't handle nan, so we make sure f8_tensor doesn't contain any nan
    all_exp_ones = (f8_tensor & 0b01111100) == 128 - 2**in_dtype.fp_mantissa_width
    f8_tensor[all_exp_ones] = 0
    f8 = triton.reinterpret(f8_tensor, in_dtype)
    n_elements = f8_tensor.numel()
    out_f8 = torch.empty_like(f8_tensor)
    abs_kernel[(1,)](f8, triton.reinterpret(out_f8, in_dtype), n_elements)

    f32_tensor = convert_float_to_float32(f8_tensor, in_dtype)
    expect = f32_tensor.abs()
    actual_f8 = convert_float_to_float32(out_f8, in_dtype)
    torch.testing.assert_close(actual_f8, expect)    

test_abs_fp8(tl.float8e4b15, "cuda")