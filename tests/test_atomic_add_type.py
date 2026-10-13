"""atomicAdd prints atom.{space}.add.{element type}.

The value is converted to the element type and passed in a register of that
bank. A shared array is atom.shared at the .u32 shared-space offset. The old
printer emitted atom.global.add.u32 for every call.
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from compiler import CompileError, Compiler

_PTXAS_CANDIDATES = [
    shutil.which('ptxas'),
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\ptxas.exe',
]
PTXAS = next((p for p in _PTXAS_CANDIDATES if p and os.path.isfile(p)), None)

_EXAMPLE_HASHES = {
    'simple_add': 'c25ed4e5062c594a',
    'minimal': 'd49478924947143f',
    'fe_add': 'e71841a6e0e874bc',
    'vector_add': 'a8e7bbe67829ec72',
}
_ROOT = os.path.join(os.path.dirname(__file__), '..')

_ATOM = re.compile(
    r"atom\.(global|shared)\.add\.(u32|u64|f32)\s+(%[\w.]+),\s+\[([^\]]+)\],\s+(%[\w.]+);"
)


def _compile(src: str) -> str:
    ptx, _ = Compiler().compile_source(src, '<atomic>')
    return ptx


def _assemble(ptx: str) -> subprocess.CompletedProcess:
    assert PTXAS, 'ptxas is not installed'
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, 'k.ptx')
        out = os.path.join(td, 'k.cubin')
        with open(src, 'w', encoding='utf-8', newline='\n') as f:
            f.write(ptx)
        return subprocess.run(
            [PTXAS, '-arch=sm_89', '-o', out, src],
            capture_output=True, text=True,
        )


def _atoms(ptx: str):
    return _ATOM.findall(ptx)


def _tx(body: str, params: str) -> str:
    return f"""
motor atom_m {{
{body}
}}
transmission atom_tx {{
    grid (1, 1, 1);
    block (32, 1, 1);
    motor atom_m {{ {params} }}
}}
"""


def test_float_global_atomic_is_f32():
    ptx = _compile(_tx(
        """
    @register(R0) float* slot;
    float v;
    v = 0.25;
    atomicAdd(&slot, v);
        """,
        "inout float* slot;",
    ))
    atoms = _atoms(ptx)
    assert atoms, ptx
    space, ty, dest, _addr, val = atoms[0]
    assert space == 'global' and ty == 'f32'
    assert dest.startswith('%f') and val.startswith('%f')
    assert 'atom.global.add.u32' not in ptx
    proc = _assemble(ptx)
    assert proc.returncode == 0, proc.stderr


def test_int_literal_into_float_slot_converts():
    ptx = _compile(_tx(
        """
    @register(R0) float* slot;
    atomicAdd(slot, 1);
        """,
        "inout float* slot;",
    ))
    atoms = _atoms(ptx)
    assert len(atoms) == 1
    space, ty, dest, _addr, val = atoms[0]
    assert space == 'global' and ty == 'f32'
    assert dest.startswith('%f') and val.startswith('%f')
    assert 'atom.global.add.u32' not in ptx
    proc = _assemble(ptx)
    assert proc.returncode == 0, proc.stderr


def test_u32_index_atomic_scales_by_four():
    ptx = _compile(_tx(
        """
    @register(R0) u32* counter;
    int i;
    atomicAdd(counter[i], 1);
        """,
        "inout u32* counter;",
    ))
    atoms = _atoms(ptx)
    assert len(atoms) == 1
    space, ty, dest, _addr, val = atoms[0]
    assert space == 'global' and ty == 'u32'
    assert dest.startswith('%r') and not dest.startswith('%rd')
    assert val.startswith('%r') and not val.startswith('%rd')
    assert 'mad.wide.u32' in ptx
    assert re.search(r'mad\.wide\.u32\s+%[\w.]+,\s+%[\w.]+,\s+4,', ptx)
    proc = _assemble(ptx)
    assert proc.returncode == 0, proc.stderr


def test_shared_u32_base_is_atom_shared():
    ptx = _compile(_tx(
        """
    __shared__ u32 bins[32];
    atomicAdd(bins, 1);
        """,
        "",
    ))
    assert '.shared .align 16 .b8 bins[128];' in ptx
    assert 'cvta.shared' not in ptx
    assert re.search(r'mov\.u32\s+%r\d+,\s+bins;', ptx)
    atoms = _atoms(ptx)
    assert len(atoms) == 1
    space, ty, dest, addr, val = atoms[0]
    assert space == 'shared' and ty == 'u32'
    assert dest.startswith('%r') and not dest.startswith('%rd')
    assert val.startswith('%r') and not val.startswith('%rd')
    assert addr.startswith('%r') and not addr.startswith('%rd')
    assert 'atom.global' not in ptx
    proc = _assemble(ptx)
    assert proc.returncode == 0, proc.stderr


def test_shared_u64_widens_the_added_one():
    ptx = _compile(_tx(
        """
    __shared__ u64 scan[8];
    atomicAdd(scan, 1);
        """,
        "",
    ))
    assert 'cvta.shared' not in ptx
    assert re.search(r'mov\.u32\s+%r\d+,\s+scan;', ptx)
    atoms = _atoms(ptx)
    assert len(atoms) == 1
    space, ty, dest, addr, val = atoms[0]
    assert space == 'shared' and ty == 'u64'
    assert addr.startswith('%r') and not addr.startswith('%rd')
    assert dest.startswith('%rd') or dest.startswith('%v')
    assert val.startswith('%rd') or val.startswith('%v')
    assert 'atom.global.add.u32' not in ptx
    proc = _assemble(ptx)
    assert proc.returncode == 0, proc.stderr


def test_shared_float_index_is_atom_shared_f32():
    ptx = _compile(_tx(
        """
    __shared__ float tile[32];
    int i;
    float v;
    v = 1.5;
    atomicAdd(tile[i], v);
        """,
        "",
    ))
    assert 'cvta.shared' not in ptx
    assert re.search(r'mov\.u32\s+%r\d+,\s+tile;', ptx)
    assert 'mad.wide' not in ptx
    assert re.search(r'mad\.lo\.u32\s+\S+,\s+\S+,\s+4,', ptx)
    atoms = _atoms(ptx)
    assert len(atoms) == 1
    space, ty, dest, addr, val = atoms[0]
    assert space == 'shared' and ty == 'f32'
    assert addr.startswith('%r') and not addr.startswith('%rd')
    assert dest.startswith('%f') and val.startswith('%f')
    assert 'atom.global' not in ptx
    proc = _assemble(ptx)
    assert proc.returncode == 0, proc.stderr


def test_scalar_address_is_a_compile_error():
    with pytest.raises(CompileError):
        _compile(_tx(
            """
    float v;
    v = 1.0;
    atomicAdd(v, 1);
            """,
            "",
        ))


def test_f64_atomic_is_a_compile_error():
    with pytest.raises(CompileError):
        _compile(_tx(
            """
    __shared__ double acc[4];
    atomicAdd(acc, 1.0);
            """,
            "",
        ))


def _launch(ptx: str, threads: int) -> float:
    import ctypes
    from ctypes import c_int, c_uint, c_void_p, c_ulonglong, c_float, POINTER, byref
    cuda = ctypes.windll.LoadLibrary("nvcuda.dll")
    for name, args in (
        ("cuInit", [c_uint]),
        ("cuDeviceGet", [POINTER(c_int), c_int]),
        ("cuCtxCreate_v2", [POINTER(c_void_p), c_uint, c_int]),
        ("cuCtxDestroy_v2", [c_void_p]),
        ("cuModuleLoadData", [POINTER(c_void_p), ctypes.c_char_p]),
        ("cuModuleGetFunction", [POINTER(c_void_p), c_void_p, ctypes.c_char_p]),
        ("cuModuleUnload", [c_void_p]),
        ("cuMemAlloc_v2", [POINTER(c_ulonglong), c_ulonglong]),
        ("cuMemFree_v2", [c_ulonglong]),
        ("cuMemcpyHtoD_v2", [c_ulonglong, c_void_p, c_ulonglong]),
        ("cuMemcpyDtoH_v2", [c_void_p, c_ulonglong, c_ulonglong]),
        ("cuLaunchKernel", [
            c_void_p, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint,
            c_uint, c_void_p, POINTER(c_void_p), POINTER(c_void_p),
        ]),
        ("cuCtxSynchronize", []),
    ):
        fn = getattr(cuda, name)
        fn.argtypes = args
        fn.restype = c_int
    entry = re.search(r"\.entry\s+(\w+)", ptx).group(1)
    assert cuda.cuInit(0) == 0
    dev = c_int()
    assert cuda.cuDeviceGet(byref(dev), 0) == 0
    ctx = c_void_p()
    assert cuda.cuCtxCreate_v2(byref(ctx), 0, dev) == 0
    try:
        mod = c_void_p()
        load = cuda.cuModuleLoadData(byref(mod), ptx.encode())
        assert load == 0, load
        fn = c_void_p()
        assert cuda.cuModuleGetFunction(byref(fn), mod, entry.encode()) == 0
        d = c_ulonglong()
        assert cuda.cuMemAlloc_v2(byref(d), 4) == 0
        zero = c_float(0.0)
        assert cuda.cuMemcpyHtoD_v2(d, byref(zero), 4) == 0
        params = (c_void_p * 1)(ctypes.cast(ctypes.pointer(d), c_void_p))
        launch_rc = cuda.cuLaunchKernel(fn, 1, 1, 1, threads, 1, 1, 0, None, params, None)
        sync = cuda.cuCtxSynchronize()
        out = c_float()
        if sync == 0:
            assert cuda.cuMemcpyDtoH_v2(byref(out), d, 4) == 0
        cuda.cuMemFree_v2(d)
        cuda.cuModuleUnload(mod)
        assert launch_rc == 0, launch_rc
        assert sync == 0, sync
        return float(out.value)
    finally:
        cuda.cuCtxDestroy_v2(ctx)


def test_shared_lane_reduction_launches():
    ptx = _compile(_tx(
        """
    @register(R0) float* out;
    __shared__ float reduction[256];
    int lane;
    float one;
    lane = threadIdx.x;
    one = 1.0;
    reduction[lane] = 0.0;
    __syncthreads();
    atomicAdd(&reduction[0], one);
    __syncthreads();
    if (lane == 0) {
        out[0] = reduction[0];
    }
        """,
        "out float* out;",
    ))
    assert ".shared .align 16 .b8 reduction[1024];" in ptx
    assert "cvta.shared" not in ptx
    assert "atom.shared.add.f32" in ptx
    assert "atom.global" not in ptx
    got = _launch(ptx, 256)
    assert got == pytest.approx(256.0)


def test_global_atomic_still_launches():
    ptx = _compile(_tx(
        """
    @register(R0) float* out;
    float one;
    one = 1.0;
    atomicAdd(&out[0], one);
        """,
        "inout float* out;",
    ))
    assert "atom.global.add.f32" in ptx
    assert "atom.shared" not in ptx
    assert "cvta.shared" not in ptx
    got = _launch(ptx, 256)
    assert got == pytest.approx(256.0)


def test_example_hashes_unchanged():
    for name, digest in _EXAMPLE_HASHES.items():
        path = os.path.join(_ROOT, 'examples', name + '.mtl')
        ptx, _ = Compiler().compile_file(path)
        got = hashlib.sha256(ptx.encode()).hexdigest()[:16]
        assert got == digest, name
