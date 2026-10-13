"""
GPU correctness test for hash160_compact.mtl — 38-register, 0-spill motor.

Compact motor signature:
  sha_msg     [N*64 u32]  pre-expanded SHA256 message schedule
  hash160_out [N*5  u32]  output HASH160 words (little-endian per word)

No sha_state / tmp / rmd_state — IVs hardcoded as literals.

Test vector: k=1
  pubkey  : 0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
  HASH160 : 751e76e8199196d454941c45d1b3a323f1433bd6
"""

import struct, ctypes, os, sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

PUBKEY_K1 = bytes.fromhex(
    '0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798'
)
HASH160_K1 = bytes.fromhex('751e76e8199196d454941c45d1b3a323f1433bd6')


def _rotr32(x, n):
    return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF


def _sha256_schedule(pk33: bytes) -> list[int]:
    assert len(pk33) == 33
    padded = pk33 + b'\x80' + b'\x00' * (64 - 33 - 1 - 8) + struct.pack('>Q', 264)
    W = list(struct.unpack('>16I', padded))
    for i in range(16, 64):
        s0 = _rotr32(W[i-15], 7) ^ _rotr32(W[i-15], 18) ^ (W[i-15] >> 3)
        s1 = _rotr32(W[i-2], 17) ^ _rotr32(W[i-2], 19) ^ (W[i-2] >> 10)
        W.append((W[i-16] + s0 + W[i-7] + s1) & 0xFFFFFFFF)
    return W


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
def test_hash160_compact_k1():
    """Single-thread correctness: compact motor on k=1 pubkey."""
    from compiler import Compiler
    from runtime.cuda_driver import (
        CUDA, CUdevice, CUcontext, CUmodule, CUfunction,
        byref, c_void_p, CUdeviceptr, CHECK,
    )

    examples_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')
    c = Compiler()
    ptx, _ = c.compile_file(os.path.join(examples_dir, 'hash160_compact.mtl'))

    CUDA.cuInit(0)
    dev = CUdevice(); CHECK(CUDA.cuDeviceGet(byref(dev), 0))
    ctx = CUcontext(); CHECK(CUDA.cuCtxCreate_v2(byref(ctx), 0, dev))

    mod = CUmodule()
    CHECK(CUDA.cuModuleLoadData(byref(mod), ptx.encode('utf-8') + b'\x00'))
    fn = CUfunction()
    CHECK(CUDA.cuModuleGetFunction(byref(fn), mod,
                                   b'hash160_compact_hash160_compact_launch'))

    N = 256  # 1 block × 256 threads
    sched = _sha256_schedule(PUBKEY_K1)
    sha_msg_words = sched * N
    hash160_out_words = [0] * (N * 5)

    def _pack(words):
        return struct.pack(f'<{len(words)}I', *words)

    h_msg = _pack(sha_msg_words)
    h_out = _pack(hash160_out_words)

    d_msg = CUdeviceptr(); CHECK(CUDA.cuMemAlloc_v2(byref(d_msg), len(h_msg)))
    d_out = CUdeviceptr(); CHECK(CUDA.cuMemAlloc_v2(byref(d_out), len(h_out)))
    CHECK(CUDA.cuMemcpyHtoD_v2(d_msg, h_msg, len(h_msg)))
    CHECK(CUDA.cuMemcpyHtoD_v2(d_out, h_out, len(h_out)))

    params = [ctypes.cast(ctypes.byref(d_msg), c_void_p),
              ctypes.cast(ctypes.byref(d_out), c_void_p)]
    pa = (c_void_p * 2)(*params)

    CHECK(CUDA.cuLaunchKernel(fn, 1,1,1, N,1,1, 0, None, pa, None))
    CHECK(CUDA.cuCtxSynchronize())

    out_buf = (ctypes.c_byte * len(h_out))()
    CHECK(CUDA.cuMemcpyDtoH_v2(out_buf, d_out, len(h_out)))

    CUDA.cuMemFree_v2(d_msg)
    CUDA.cuMemFree_v2(d_out)
    CUDA.cuModuleUnload(mod)
    CUDA.cuCtxDestroy_v2(ctx)

    wrong = []
    for tid in range(N):
        offset = tid * 5 * 4
        words = struct.unpack_from('<5I', bytes(out_buf), offset)
        got = struct.pack('<5I', *words)
        if got != HASH160_K1:
            wrong.append((tid, got.hex()))

    if wrong:
        sample = wrong[:5]
        pytest.fail(
            f"{len(wrong)}/{N} threads wrong.\n"
            f"First failures: {sample}\n"
            f"Expected: {HASH160_K1.hex()}"
        )

    print(f"\nCOMPACT HASH160 VERIFIED: {N} threads, all output {HASH160_K1.hex()}")


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA GPU not available")
def test_hash160_compact_spill_gate():
    """Compile compact motor and verify ≤38 registers, 0 spills via ptxas."""
    import shutil
    if not shutil.which('ptxas'):
        pytest.skip("ptxas not found — spill gate skipped")

    from compiler import Compiler
    from native_verify import single_kernel_spill_gate

    examples_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')
    c = Compiler()
    ptx, _ = c.compile_file(os.path.join(examples_dir, 'hash160_compact.mtl'))

    result = single_kernel_spill_gate(
        ptx,
        kernel_name="hash160_compact",
        sm="sm_89",
        max_registers=38,
        spill_allowed=False,
        local_memory_allowed=False,
    )

    assert result.spills_store == 0, f"spill_store={result.spills_store}"
    assert result.spills_load == 0, f"spill_load={result.spills_load}"
    assert result.registers <= 38, f"registers={result.registers} exceeds 38"
    print(f"\nSPILL GATE: {result.registers} regs, 0 spills — PASS")
