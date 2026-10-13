"""
GPU correctness test for hash160_rawpk.mtl — 37-register, on-device schedule expansion.

Motor signature:
  pubkeys     [N*9 u32]  W[0..8] big-endian packed SHA256 block words
  hash160_out [N*5 u32]  HASH160 output words (little-endian)

CPU packing: pack_pubkey_to_9words(pk33) → 9 u32s where
  words[0..7] = pk33 bytes 0-31 as big-endian u32s
  words[8]    = (pk33[32] << 24) | 0x00800000
  W[9..14]=0, W[15]=0x108 are motor-internal constants.

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


def pack_pubkey_to_9words(pk33: bytes) -> list[int]:
    assert len(pk33) == 33
    words = list(struct.unpack('>8I', pk33[:32]))
    words.append((pk33[32] << 24) | 0x00800000)
    return words  # 9 u32s; W[9..14]=0, W[15]=0x108 are motor constants


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
def test_hash160_rawpk_k1():
    """256-thread batch on k=1 pubkey; all outputs must equal HASH160_K1."""
    from compiler import Compiler
    from runtime.cuda_driver import (
        CUDA, CUdevice, CUcontext, CUmodule, CUfunction,
        byref, c_void_p, CUdeviceptr, CHECK,
    )

    N = 256
    examples_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')
    c = Compiler()
    ptx, _ = c.compile_file(os.path.join(examples_dir, 'hash160_rawpk.mtl'))

    CUDA.cuInit(0)
    dev = CUdevice(); CHECK(CUDA.cuDeviceGet(byref(dev), 0))
    ctx = CUcontext(); CHECK(CUDA.cuCtxCreate_v2(byref(ctx), 0, dev))

    mod = CUmodule()
    CHECK(CUDA.cuModuleLoadData(byref(mod), ptx.encode('utf-8') + b'\x00'))
    fn = CUfunction()
    CHECK(CUDA.cuModuleGetFunction(byref(fn), mod,
                                   b'hash160_rawpk_hash160_rawpk_launch'))

    words9 = pack_pubkey_to_9words(PUBKEY_K1)
    pubkeys_words  = words9 * N
    hash160_words  = [0] * (N * 5)

    def _pack(words):
        return struct.pack(f'<{len(words)}I', *words)

    h_pk  = _pack(pubkeys_words)
    h_out = _pack(hash160_words)

    d_pk  = CUdeviceptr(); CHECK(CUDA.cuMemAlloc_v2(byref(d_pk),  len(h_pk)))
    d_out = CUdeviceptr(); CHECK(CUDA.cuMemAlloc_v2(byref(d_out), len(h_out)))
    CHECK(CUDA.cuMemcpyHtoD_v2(d_pk,  h_pk,  len(h_pk)))
    CHECK(CUDA.cuMemcpyHtoD_v2(d_out, h_out, len(h_out)))

    params = [ctypes.cast(ctypes.byref(d_pk),  c_void_p),
              ctypes.cast(ctypes.byref(d_out), c_void_p)]
    pa = (c_void_p * 2)(*params)

    CHECK(CUDA.cuLaunchKernel(fn, 1,1,1, N,1,1, 0, None, pa, None))
    CHECK(CUDA.cuCtxSynchronize())

    out_buf = (ctypes.c_byte * len(h_out))()
    CHECK(CUDA.cuMemcpyDtoH_v2(out_buf, d_out, len(h_out)))

    CUDA.cuMemFree_v2(d_pk)
    CUDA.cuMemFree_v2(d_out)
    CUDA.cuModuleUnload(mod)
    CUDA.cuCtxDestroy_v2(ctx)

    wrong = []
    for tid in range(N):
        offset = tid * 5 * 4
        five_words = struct.unpack_from('<5I', bytes(out_buf), offset)
        got = struct.pack('<5I', *five_words)
        if got != HASH160_K1:
            wrong.append((tid, got.hex()))

    if wrong:
        sample = wrong[:5]
        pytest.fail(
            f"{len(wrong)}/{N} threads wrong.\n"
            f"First failures: {sample}\n"
            f"Expected: {HASH160_K1.hex()}"
        )

    print(f"\nRAWPK HASH160 VERIFIED: {N} threads, all output {HASH160_K1.hex()}")


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA GPU not available")
def test_hash160_rawpk_spill_gate():
    """Compile rawpk motor and verify ≤38 registers, 0 spills."""
    import shutil
    if not shutil.which('ptxas'):
        pytest.skip("ptxas not found")

    from compiler import Compiler
    from native_verify import single_kernel_spill_gate

    examples_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')
    c = Compiler()
    ptx, _ = c.compile_file(os.path.join(examples_dir, 'hash160_rawpk.mtl'))

    result = single_kernel_spill_gate(
        ptx,
        kernel_name="hash160_rawpk",
        sm="sm_89",
        max_registers=42,   # occupancy boundary for sm_89 @ 256 threads/block
        spill_allowed=False,
        local_memory_allowed=False,
    )

    assert result.spills_store == 0, f"spill_store={result.spills_store}"
    assert result.spills_load == 0, f"spill_load={result.spills_load}"
    assert result.registers <= 42, f"registers={result.registers} exceeds occupancy limit 42"
    print(f"\nSPILL GATE: {result.registers} regs, 0 spills — PASS (100% occ)")
