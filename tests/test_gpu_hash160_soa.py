"""
GPU correctness + spill-gate tests for hash160_rawpk_soa.mtl — SoA transposed I/O.

Motor signature:
  pubkeys     [N*9 u32]  SoA: word j of hash i at pubkeys[j*N + i]
  hash160_out [N*5 u32]  SoA: word w of hash i at hash160_out[w*N + i]

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
    return words


def pack_soa(all_9word_lists: list[list[int]]) -> bytes:
    """Pack N pubkeys into SoA layout: [word0 for all N, word1 for all N, ..., word8 for all N]."""
    n = len(all_9word_lists)
    flat = []
    for j in range(9):
        for i in range(n):
            flat.append(all_9word_lists[i][j])
    return struct.pack(f'<{n*9}I', *flat)


def unpack_soa_output(raw: bytes, n: int) -> list[bytes]:
    """Unpack SoA output [word0 for all N, ..., word4 for all N] → list of 20-byte hashes."""
    words = struct.unpack(f'<{n*5}I', raw)
    results = []
    for i in range(n):
        five = [words[j * n + i] for j in range(5)]
        results.append(struct.pack('<5I', *five))
    return results


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
def test_hash160_soa_k1():
    """256-thread batch on k=1 pubkey; all SoA outputs must equal HASH160_K1."""
    from compiler import Compiler
    from runtime.cuda_driver import (
        CUDA, CUdevice, CUcontext, CUmodule, CUfunction,
        byref, c_void_p, CUdeviceptr, CHECK,
    )

    N = 256
    examples_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')
    c = Compiler()
    ptx, _ = c.compile_file(os.path.join(examples_dir, 'hash160_rawpk_soa.mtl'))

    CUDA.cuInit(0)
    dev = CUdevice(); CHECK(CUDA.cuDeviceGet(byref(dev), 0))
    ctx = CUcontext(); CHECK(CUDA.cuCtxCreate_v2(byref(ctx), 0, dev))

    mod = CUmodule()
    CHECK(CUDA.cuModuleLoadData(byref(mod), ptx.encode('utf-8') + b'\x00'))
    fn = CUfunction()
    CHECK(CUDA.cuModuleGetFunction(byref(fn), mod,
                                   b'hash160_rawpk_hash160_rawpk_launch'))

    words9 = pack_pubkey_to_9words(PUBKEY_K1)
    h_pk  = pack_soa([words9] * N)       # SoA layout
    h_out = bytes(N * 5 * 4)             # zeroed output

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

    hashes = unpack_soa_output(bytes(out_buf), N)
    wrong = [(i, h.hex()) for i, h in enumerate(hashes) if h != HASH160_K1]

    if wrong:
        pytest.fail(
            f"{len(wrong)}/{N} threads wrong.\n"
            f"First failures: {wrong[:5]}\n"
            f"Expected: {HASH160_K1.hex()}"
        )

    print(f"\nSoA HASH160 VERIFIED: {N} threads, all output {HASH160_K1.hex()}")


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA GPU not available")
def test_hash160_soa_spill_gate():
    """Compile SoA motor and verify ≤42 registers, 0 spills (100% occupancy)."""
    import shutil
    if not shutil.which('ptxas'):
        pytest.skip("ptxas not found")

    from compiler import Compiler
    from native_verify import single_kernel_spill_gate

    examples_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')
    c = Compiler()
    ptx, _ = c.compile_file(os.path.join(examples_dir, 'hash160_rawpk_soa.mtl'))

    result = single_kernel_spill_gate(
        ptx,
        kernel_name="hash160_rawpk",
        sm="sm_89",
        max_registers=42,
        spill_allowed=False,
        local_memory_allowed=False,
    )

    assert result.spills_store == 0, f"spill_store={result.spills_store}"
    assert result.spills_load == 0, f"spill_load={result.spills_load}"
    assert result.registers <= 42, f"registers={result.registers} exceeds 42 (100% occ boundary)"
    print(f"\nSoA SPILL GATE: {result.registers} regs, 0 spills — PASS (100% occ)")
