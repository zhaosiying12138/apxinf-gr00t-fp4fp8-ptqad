"""NVFP4 quantization primitives (bit-exact, numpy-only, CPU).

Mirrors the semantics verified by spike/fp4_gemm_bench.cu on sm_120:
  - data: E2M1 (values +/- {0,.5,1,1.5,2,3,4,6}), round-nearest-even, satfinite
  - scale: E4M3 per 16-element block along the last (K) dim, secondary per-tensor FP32
  - physical scale layout (cuBLASLt "VEC16_UE4M3" swizzle):
      PS = 512*ceil(KB/4);  offset(r,b) = PS*(r//128) + 512*(b//4) + 16*(r%32) + 4*((r//32)%4) + (b%4)
All encoders are lookup/bit-exact so fake-quant (PyTorch) and engine (Rust/CUDA)
can be validated against this single reference.
"""
from __future__ import annotations
import numpy as np

# ---------------- bit-exact format encoders (LUT) ----------------
_E2M1_POS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
_E2M1_MIDS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])  # between codes i and i+1

def encode_e2m1(x: np.ndarray) -> np.ndarray:
    """Round-nearest E2M1 code; exact midpoints -> even code; satfinite at ±6. Shape-agnostic."""
    shape = x.shape
    xa = np.asarray(x, dtype=np.float64).ravel()
    sign = np.signbit(xa).astype(np.int8)
    a = np.abs(xa)
    code = np.searchsorted(_E2M1_MIDS, a, side="left").astype(np.int8)  # a<mids[0] -> 0
    i = np.clip(code, 0, 6)
    tied = a == _E2M1_MIDS[i]
    even = np.where(i % 2 == 0, i, i + 1)
    code = np.where(tied, even, code)
    return (code | (sign << 3)).reshape(shape).astype(np.int8)

def decode_e2m1(c: np.ndarray) -> np.ndarray:
    c = c.astype(np.int8) & 0xF
    mag = _E2M1_POS[c & 0x7]
    return np.where((c & 0x8) != 0, -mag, mag).astype(np.float32)

_E4M3_BIAS = 7
def encode_e4m3(x: np.ndarray) -> np.ndarray:
    """Round to nearest E4M3 (bit-exact float32<->fp8 via exponent ladder). Satfinite ±448."""
    a = np.asarray(x, dtype=np.float64)
    sign = np.uint8(0x80) * (a < 0)
    a = np.abs(a)
    out = np.zeros(a.shape, dtype=np.uint8)
    # exponent ladder for normal range: 2^(e-7), e in 1..15; mantissa 3 bits
    # subnormals: 2^-9 * m/8 (m=1..7); build full positive grid
    grid, codes = [], []
    for m in range(1, 8):
        grid.append(m * 2.0 ** -9); codes.append(m)            # subnormal codes 1..7
    for e in range(1, 16):
        for m in range(0, 8):
            grid.append((1 + m / 8.0) * 2.0 ** (e - _E4M3_BIAS))
            codes.append((e << 3) | m)
    grid = np.array(grid); codes = np.array(codes, dtype=np.uint8)
    assert np.all(np.diff(grid) > 0)
    idx = np.searchsorted(grid, a, side="left") - 1
    idx = np.clip(idx, 0, len(grid) - 1)
    # nearest of idx / idx+1 with even-tie on mantissa bit parity
    lo = grid[idx]; hi = grid[np.minimum(idx + 1, len(grid) - 1)]
    choose_hi = (a - lo) > (hi - a)
    exact_tie = (a - lo) == (hi - a)
    cl, ch = codes[idx], codes[np.minimum(idx + 1, len(grid) - 1)]
    # RN-even on the full code: prefer even code (mantissa bit0==0)
    tie_pick = np.where((cl & 1) == 0, cl, ch)
    res = np.where(choose_hi, ch, cl)
    res = np.where(exact_tie, tie_pick, res)
    res = np.where(a >= 464, np.uint8(0x7E), res)              # satfinite: >=464 -> 448
    res = np.where(res == np.uint8(0x7F), np.uint8(0x7E), res) # never emit NaN code
    res = np.where(a == 0, np.uint8(0), res)
    return (res | sign.astype(np.uint8)).astype(np.uint8)

def decode_e4m3(b: np.ndarray) -> np.ndarray:
    b = b.astype(np.uint8)
    sign = np.where((b & 0x80) != 0, -1.0, 1.0)
    e = (b >> 3) & 0xF
    m = b & 0x7
    mag = np.where(
        e == 0,
        m.astype(np.float64) * 2.0 ** -9,
        (1.0 + m / 8.0) * np.power(2.0, e.astype(np.float64) - _E4M3_BIAS),
    )
    out = sign * mag
    out = np.where((b & 0x7F) == 0x7F, np.nan, out)  # 0x7F/0xFF = NaN in E4M3 (OCP)
    return out.astype(np.float32)

# ---------------- NVFP4 tensor quantization ----------------
def nvfp4_quantize(W: np.ndarray, block: int = 16, tscale: float | None = None):
    """W: (rows, K) float. Returns packed (rows,K//2) uint8, scales (rows,KB) uint8, tscale float32.
    tscale: optional precomputed per-tensor scale (chunked conversion must share one)."""
    W = np.asarray(W, dtype=np.float64)
    rows, K = W.shape
    assert K % block == 0, f"K={K} must be divisible by {block}"
    KB = K // block
    wblk = W.reshape(rows, KB, block)
    amax = np.abs(wblk).max(axis=2)                          # (rows, KB)
    if tscale is None:
        tscale = float(np.float32(amax.max() / 448.0)) if amax.max() > 0 else 1.0
    scale_f = amax / 6.0 / tscale                            # leave E4M3 headroom for global scale
    scales = encode_e4m3(scale_f)
    sd = decode_e4m3(scales).astype(np.float64) * tscale     # exact dequantized scale
    sd[sd == 0] = 1.0
    q = encode_e2m1(wblk / sd[:, :, None]).astype(np.int8)
    packed = ((q[:, :, 0::2] & 0xF) | ((q[:, :, 1::2] & 0xF) << 4)).astype(np.uint8)
    packed = packed.reshape(rows, K // 2)
    return packed, scales, np.float32(tscale)

def nvfp4_dequantize(packed: np.ndarray, scales: np.ndarray, tscale: np.float32, block: int = 16) -> np.ndarray:
    rows, Khalf = packed.shape
    K = Khalf * 2
    KB = K // block
    lo = decode_e2m1(packed.astype(np.int8) & 0xF)
    hi = decode_e2m1((packed.astype(np.int16) >> 4) & 0xF)
    q = np.empty((rows, K), dtype=np.float32)
    q[:, 0::2] = lo; q[:, 1::2] = hi
    sd = decode_e4m3(scales).astype(np.float32) * np.float32(tscale)
    sd[sd == 0] = 1.0
    return (q.reshape(rows, KB, block) * sd[:, :, None]).reshape(rows, K)

# ---------------- cuBLASLt scale swizzle ----------------
def swizzle_scales(scales: np.ndarray) -> np.ndarray:
    """scales: (rows, KB) uint8 -> physical buffer per decoded cuBLASLt layout."""
    rows, KB = scales.shape
    PS = 512 * ((KB + 3) // 4)
    out = np.zeros(PS * ((rows + 127) // 128), dtype=np.uint8)
    r = np.arange(rows)[:, None]
    b = np.arange(KB)[None, :]
    off = PS * (r // 128) + 512 * (b // 4) + 16 * (r % 32) + 4 * ((r // 32) % 4) + (b % 4)
    out[off.ravel()] = scales.ravel()
    return out

# ---------------- self-test ----------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # e4m3 roundtrip: every code decodes, and encode(decode(c)) == c for all codes
    codes = np.arange(256, dtype=np.uint8)
    val = decode_e4m3(codes)
    finite = ~np.isnan(val)
    re = encode_e4m3(val[finite])
    same = (re == codes[finite])
    print(f"e4m3 re-encode exact: {same.all()} ({same.sum()}/{same.size})")
    # e2m1 roundtrip
    c = rng.integers(0, 16, 4096).astype(np.int8)
    v = decode_e2m1(c)
    print("e2m1 re-encode exact:", (encode_e2m1(v) == c).all())
    # nvfp4 quantize/dequant error on a linear-ish weight
    W = rng.normal(0, 0.05, size=(256, 512)).astype(np.float32)
    packed, scales, ts = nvfp4_quantize(W)
    Wq = nvfp4_dequantize(packed, scales, ts)
    rel = np.abs(Wq - W).mean() / np.abs(W).mean()
    print(f"nvfp4 roundtrip: mean-rel-err={rel:.4f}  (expect ~0.03-0.08)")
    # swizzle: logical scale at (r,b) must land at the decoded offset
    s = rng.integers(0, 255, size=(300, 20)).astype(np.uint8)
    phys = swizzle_scales(s)
    PS = 512 * ((20 + 3) // 4)
    ok = True
    for (r, b) in [(0, 0), (0, 15), (1, 4), (31, 19), (32, 0), (97, 3), (128, 0), (299, 19)]:
        off = PS * (r // 128) + 512 * (b // 4) + 16 * (r % 32) + 4 * ((r // 32) % 4) + (b % 4)
        ok &= phys[off] == s[r, b]
    print("swizzle offsets exact:", ok)
    print("packed bytes per 3B model ~= %.2f GB (weights+scales)" % (3e9 / 2 / 1e9 + 3e9 / 16 / 1e9))
