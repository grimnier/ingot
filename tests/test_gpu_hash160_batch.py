"""
GPU batch test for hash160_batch_u32.mtl.

Launches N threads; each thread processes its own per-thread slice of flat
buffer arrays.  All threads run the same pubkey (k=1 compressed) so every
output must equal 751e76e8199196d454941c45d1b3a323f1433bd6.

Test also verifies that the per-thread indexing is correct (no thread bleeds
into a neighbour's state) by confirming that all N results are individually
correct.
"""

import struct, ctypes, os, sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ── SHA256 constants ───────────────────────────────────────────────────────────

SHA256_K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]

SHA256_H0 = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
]

# ── RIPEMD-160 constants ───────────────────────────────────────────────────────

RMD_H0 = [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0]

RMD_RL = [
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
    1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
    4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13,
]

RMD_RR = [
    5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
    6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
    8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
    12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11,
]

RMD_SL = [
    11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
    7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
    11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
    9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6,
]

RMD_SR = [
    8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
    9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
    15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
    8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11,
]

RMD_KL = (
    [0x00000000]*16 + [0x5a827999]*16 + [0x6ed9eba1]*16 +
    [0x8f1bbcdc]*16 + [0xa953fd4e]*16
)

RMD_KR = (
    [0x50a28be6]*16 + [0x5c4dd124]*16 + [0x6d703ef3]*16 +
    [0x7a6d76e9]*16 + [0x00000000]*16
)

# ── helpers ────────────────────────────────────────────────────────────────────

def _rotr32(x, n):
    return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF

def _sha256_schedule(pubkey_33: bytes) -> list:
    assert len(pubkey_33) == 33
    padded = pubkey_33 + b'\x80' + b'\x00' * (64 - 33 - 1 - 8) + struct.pack('>Q', 33 * 8)
    W = list(struct.unpack('>16I', padded))
    for i in range(16, 64):
        s0 = _rotr32(W[i-15], 7) ^ _rotr32(W[i-15], 18) ^ (W[i-15] >> 3)
        s1 = _rotr32(W[i-2], 17) ^ _rotr32(W[i-2], 19) ^ (W[i-2] >> 10)
        W.append((W[i-16] + s0 + W[i-7] + s1) & 0xFFFFFFFF)
    return W

def _pack_u32s(words):
    return struct.pack(f'<{len(words)}I', *words)

# ── check CUDA available ───────────────────────────────────────────────────────

def _cuda_available():
    try:
        from runtime.cuda_driver import CUDA, CUdevice, byref
        CUDA.cuInit(0)
        dev = CUdevice()
        return CUDA.cuDeviceGet(byref(dev), 0) == 0
    except Exception:
        return False

CUDA_AVAILABLE = _cuda_available()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA GPU not available")
def test_hash160_batch_correctness():
    """
    Compile hash160_batch_u32.mtl, launch N=256 threads (1 block x 256),
    each with the same compressed pubkey for k=1.  All 256 HASH160 outputs
    must equal 751e76e8199196d454941c45d1b3a323f1433bd6.
    """
    from compiler import Compiler
    from runtime.cuda_driver import (
        CUDA, CUdevice, CUcontext, CUmodule, CUfunction,
        byref, c_void_p, CUdeviceptr, CHECK,
    )

    N = 256   # threads = 1 block x 256

    # ── Compile ────────────────────────────────────────────────────────────────
    examples_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')
    c = Compiler()
    ptx, prog = c.compile_file(os.path.join(examples_dir, 'hash160_batch_u32.mtl'))

    # ── Init GPU ───────────────────────────────────────────────────────────────
    CUDA.cuInit(0)
    dev = CUdevice(); CHECK(CUDA.cuDeviceGet(byref(dev), 0))
    ctx = CUcontext(); CHECK(CUDA.cuCtxCreate_v2(byref(ctx), 0, dev))

    mod = CUmodule()
    CHECK(CUDA.cuModuleLoadData(byref(mod), ptx.encode('utf-8') + b'\x00'))
    fn = CUfunction()
    CHECK(CUDA.cuModuleGetFunction(byref(fn), mod,
                                   b'hash160_batch_u32_hash160_batch_launch'))

    # ── Test vector ────────────────────────────────────────────────────────────
    pubkey = bytes.fromhex(
        '0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798'
    )
    expected = bytes.fromhex('751e76e8199196d454941c45d1b3a323f1433bd6')

    sched = _sha256_schedule(pubkey)   # 64 u32 words for one pubkey

    # ── Build flat per-thread host buffers ─────────────────────────────────────
    # sha_msg   [N * 64]  -- each thread's pre-expanded schedule (identical here)
    sha_msg_words = sched * N

    # sha_state [N *  8]  -- each thread's SHA256 IV
    sha_state_words = list(SHA256_H0) * N

    # sha_k     [    64]  -- shared constants
    sha_k_words = list(SHA256_K)

    # tmp       [N * 16]  -- bridge scratch, written by motor
    tmp_words = [0] * (N * 16)

    # rmd_state [N *  5]  -- each thread's RIPEMD-160 IV
    rmd_state_words = list(RMD_H0) * N

    # shared RIPEMD-160 tables (80 words each)
    rmd_rl_words = list(RMD_RL)
    rmd_rr_words = list(RMD_RR)
    rmd_sl_words = list(RMD_SL)
    rmd_sr_words = list(RMD_SR)
    rmd_kl_words = list(RMD_KL)
    rmd_kr_words = list(RMD_KR)

    buffers = [
        sha_msg_words, sha_state_words, sha_k_words, tmp_words, rmd_state_words,
        rmd_rl_words, rmd_rr_words, rmd_sl_words, rmd_sr_words, rmd_kl_words, rmd_kr_words,
    ]
    host_bytes = [_pack_u32s(b) for b in buffers]

    # ── Allocate and upload ────────────────────────────────────────────────────
    d_ptrs = []
    for hb in host_bytes:
        dp = CUdeviceptr()
        CHECK(CUDA.cuMemAlloc_v2(byref(dp), len(hb)))
        CHECK(CUDA.cuMemcpyHtoD_v2(dp, hb, len(hb)))
        d_ptrs.append(dp)

    params = [ctypes.cast(ctypes.byref(dp), c_void_p) for dp in d_ptrs]
    pa = (c_void_p * len(params))(*params)

    # ── Launch: 1 block x 256 threads ─────────────────────────────────────────
    CHECK(CUDA.cuLaunchKernel(fn, 1,1,1, N,1,1, 0, None, pa, None))
    CHECK(CUDA.cuCtxSynchronize())

    # ── Read back rmd_state (11 buffers; rmd_state is index 4) ────────────────
    rmd_idx = 4
    out_buf = (ctypes.c_byte * len(host_bytes[rmd_idx]))()
    CHECK(CUDA.cuMemcpyDtoH_v2(out_buf, d_ptrs[rmd_idx], len(host_bytes[rmd_idx])))

    # ── Verify every thread's output ───────────────────────────────────────────
    wrong = []
    for tid in range(N):
        offset = tid * 5 * 4   # 5 u32 words * 4 bytes each
        five_words = struct.unpack_from('<5I', bytes(out_buf), offset)
        got = struct.pack('<5I', *five_words)
        if got != expected:
            wrong.append((tid, got.hex()))

    # ── Cleanup ────────────────────────────────────────────────────────────────
    for dp in d_ptrs:
        CUDA.cuMemFree_v2(dp)
    CUDA.cuModuleUnload(mod)
    CUDA.cuCtxDestroy_v2(ctx)

    if wrong:
        sample = wrong[:5]
        pytest.fail(
            f"{len(wrong)}/{N} threads wrong.\n"
            f"First failures: {sample}\n"
            f"Expected: {expected.hex()}"
        )

    print(f"\nBATCH HASH160 VERIFIED: {N} threads, all output {expected.hex()}")
