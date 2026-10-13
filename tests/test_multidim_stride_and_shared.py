"""Multi-dimensional indexes lower to stride products. Shared memory barriers assemble.

`float* manifold[][12][4]; manifold[expert][row][col]` is row-major:
expert * (12*4) + row * 4 + col, then mad.wide by the 4-byte element.
The source does not contain a hand-written `* 12`.

`__shared__` / `shared` and `__syncthreads` / `syncthreads` are keywords.
The emitter prints `.shared .b8`, `mov.u32` of the symbol, `ld.shared` / `st.shared`,
and `bar.sync 0`. The shared address is a .u32 offset, not a generic pointer.
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

from ast_nodes import SharedDeclStmt, SyncThreadsStmt, VarDecl
from compiler import CompileError, Compiler
from lexer import Lexer
from parser import Parser

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


def _compile(src: str) -> str:
    ptx, _ = Compiler().compile_source(src, '<stride>')
    return ptx


def _instr(ptx: str) -> str:
    kept = []
    for line in ptx.splitlines():
        s = line.strip()
        if not s or s.startswith('//') or s.startswith('.'):
            continue
        kept.append(s)
    return '\n'.join(kept)


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


STRIDE = """
motor stride_math {
    @register(R0) float* manifold[][12][4];
    int expert_id;
    int row;
    int col;
    float v;
    v = manifold[expert_id][row][col];
    manifold[expert_id][row][col] = v;
}
transmission stride_tx {
    grid (1, 1, 1);
    block (1, 1, 1);
    motor stride_math { inout float* manifold; }
}
"""

CONST_STRIDE = """
motor stride_const {
    @register(R0) float* manifold[][12][4];
    float v;
    v = manifold[1][2][3];
    manifold[0][0][0] = v;
}
transmission stride_const_tx {
    grid (1, 1, 1);
    block (1, 1, 1);
    motor stride_const { inout float* manifold; }
}
"""

SHARED = """
motor tile_reduce {
    @register(R0) float* out;
    int r;
    int c;
    __shared__ float tile[32][8];
    tile[r][c] = out[0];
    __syncthreads();
    out[0] = tile[c][r];
}
transmission tile_tx {
    grid (1, 1, 1);
    block (32, 1, 1);
    motor tile_reduce { out float* out; }
}
"""

SHARED_ALIAS = """
motor tile_alias {
    @register(R0) float* out;
    int r;
    int c;
    shared float tile[8][4];
    tile[r][c] = out[0];
    syncthreads();
    out[0] = tile[c][r];
}
transmission tile_alias_tx {
    grid (1, 1, 1);
    block (32, 1, 1);
    motor tile_alias { out float* out; }
}
"""

WINDOW_AND_SHAPE = """
motor both {
    @window(R2..R3) u64 w[2];
    @register(R0) float* manifold[][12][4];
    int expert_id;
    int row;
    int col;
    w[0] = w[1];
    manifold[expert_id][row][col] = 1.0;
}
transmission both_tx {
    grid (1, 1, 1);
    block (1, 1, 1);
    motor both { inout float* manifold; }
}
"""


def test_parser_records_dims_and_keywords():
    src = """
motor m {
    @register(R0) float* manifold[][12][4];
    __shared__ float tile[32][8];
    shared float tile2[4];
    syncthreads();
}
transmission t { grid(1,1,1); block(1,1,1); motor m { inout float* manifold; } }
"""
    prog = Parser(Lexer(src).tokenize()).parse()
    body = prog.motors[0].body.stmts
    man = next(s for s in body if isinstance(s, VarDecl) and s.name == 'manifold')
    assert man.array_size is None
    assert man.dims[0] is None
    assert [d.value for d in man.dims[1:]] == [12, 4]
    tile = next(s for s in body if isinstance(s, SharedDeclStmt) and s.name == 'tile')
    assert [d.value for d in tile.dims] == [32, 8]
    assert tile.array_size is None
    tile2 = next(s for s in body if isinstance(s, SharedDeclStmt) and s.name == 'tile2')
    assert tile2.array_size.value == 4
    assert any(isinstance(s, SyncThreadsStmt) for s in body)


def test_dynamic_strides_are_the_extent_product():
    ptx = _compile(STRIDE)
    instr = _instr(ptx)
    assert re.search(r'\b(?:mul\.lo\.u32|mad\.lo\.u32)\b.*\b48\b', instr)
    assert re.search(r'\b(?:mul\.lo\.u32|mad\.lo\.u32)\b.*\b4\b', instr)
    wides = re.findall(r'mad\.wide\.u32\s+\S+,\s+\S+,\s+4,', instr)
    assert len(wides) >= 1
    assert 'ld.global.f32' in instr or 'ld.global.f32' in ptx
    assert 'st.global.f32' in ptx
    # 12 was consumed into the stride 48. It is not a runtime scale.
    assert not re.search(r'\b12\b', instr)
    assert 'cvta.shared' not in ptx
    proc = _assemble(ptx)
    assert proc.returncode == 0, proc.stderr


def test_constant_index_folds_the_same_product():
    # 1*(12*4) + 2*4 + 3 = 59 elements, times 4 bytes = 236.
    ptx = _compile(CONST_STRIDE)
    assert '[%R0+236]' in ptx
    instr = _instr(ptx)
    assert 'mad.wide' not in instr
    assert not re.search(r'\b48\b', instr)
    proc = _assemble(ptx)
    assert proc.returncode == 0, proc.stderr


def test_shared_tile_barrier_assembles():
    ptx = _compile(SHARED)
    assert '.shared .align 16 .b8 tile[1024];' in ptx
    assert '.b64' not in ptx
    assert 'cvta.shared' not in ptx
    assert re.search(r'mov\.u32\s+%r\d+,\s+tile;', ptx)
    assert 'st.shared.f32' in ptx
    assert 'ld.shared.f32' in ptx
    assert 'bar.sync 0;' in ptx
    assert ptx.find('.shared') < ptx.find('mov.u32')
    assert ptx.find('.shared') < ptx.find('ld.param')
    st = ptx.find('st.shared')
    bar = ptx.find('bar.sync 0;')
    ld = ptx.find('ld.shared')
    assert 0 <= st < bar < ld
    # Row stride of [32][8] is 8. The shared byte scale is a .u32 mad, not mad.wide.
    instr = _instr(ptx)
    assert re.search(r'\b(?:mul\.lo\.u32|mad\.lo\.u32)\b.*\b8\b', instr)
    assert re.search(r'mad\.lo\.u32\s+\S+,\s+\S+,\s+4,', instr)
    assert 'mad.wide' not in instr
    proc = _assemble(ptx)
    assert proc.returncode == 0, proc.stderr


def test_bare_shared_and_syncthreads_keywords():
    ptx = _compile(SHARED_ALIAS)
    assert '.shared .align 16 .b8 tile[128];' in ptx
    assert 'bar.sync 0;' in ptx
    assert 'cvta.shared' not in ptx
    assert re.search(r'mov\.u32\s+%r\d+,\s+tile;', ptx)
    proc = _assemble(ptx)
    assert proc.returncode == 0, proc.stderr


def test_window_rewrite_keeps_pointer_dims():
    ptx = _compile(WINDOW_AND_SHAPE)
    instr = _instr(ptx)
    assert re.search(r'\b(?:mul\.lo\.u32|mad\.lo\.u32)\b.*\b48\b', instr)


def test_rank_mismatch_is_a_compile_error():
    src = STRIDE.replace(
        'v = manifold[expert_id][row][col];',
        'v = manifold[expert_id][row];',
    )
    with pytest.raises(CompileError, match='rank 3'):
        _compile(src)


def test_unsized_inner_dimension_is_a_compile_error():
    src = """
motor bad {
    @register(R0) float* manifold[4][][2];
    manifold[0][0][0] = 1.0;
}
transmission t {
    grid (1, 1, 1);
    block (1, 1, 1);
    motor bad { inout float* manifold; }
}
"""
    with pytest.raises(CompileError, match="dimension 1"):
        _compile(src)


def test_non_pointer_multidim_is_a_compile_error():
    src = """
motor bad {
    float local[4][4];
    local[0][1] = 1.0;
}
transmission t {
    grid (1, 1, 1);
    block (1, 1, 1);
    motor bad { }
}
"""
    with pytest.raises(CompileError, match='__shared__'):
        _compile(src)


def test_unshaped_chain_is_a_compile_error():
    src = """
motor bad {
    @register(R0) float* manifold;
    int e;
    int row;
    int col;
    manifold[e][row][col] = 1.0;
}
transmission t {
    grid (1, 1, 1);
    block (1, 1, 1);
    motor bad { inout float* manifold; }
}
"""
    with pytest.raises(CompileError, match='no declared shape'):
        _compile(src)


def test_example_hashes_unchanged():
    for name, expect in _EXAMPLE_HASHES.items():
        path = os.path.join(_ROOT, 'examples', name + '.mtl')
        ptx, _ = Compiler().compile_file(path)
        got = hashlib.sha256(ptx.encode()).hexdigest()[:16]
        assert got == expect, name


def test_integer_kernels_gain_no_float_cvt():
    for name in ('sha256_compress', 'ripemd160_compress', 'point_add_mixed'):
        path = os.path.join(_ROOT, 'examples', name + '.mtl')
        ptx, _ = Compiler().compile_file(path)
        assert 'cvt.rn.f32' not in ptx, name
