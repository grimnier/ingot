"""
WP6: Warp shuffle intrinsics and multi-kernel PTX emission.

Tests:
  1. syncthreads() lowers to BarrierOp → bar.sync in PTX
  2. shfl_sync_u32(mask, src, lane) → shfl.sync.idx.b32
  3. shfl_up_sync_u32(mask, src, delta) → shfl.sync.up.b32
  4. shfl_down_sync_u32(mask, src, delta) → shfl.sync.down.b32
  5. shfl_bfly_sync_u32(mask, src, xmask) → shfl.sync.bfly.b32
  6. lane_id() → mov %laneid into a u32 register
  7. compile_all_transmissions: two motors → two .entry blocks in one PTX
  8. multi-kernel PTX shares the .version/.target/.address_size header (no dup)
  9. spill gate still fires per sealed transmission in multi-kernel mode
 10. hash160_split_engines.mtl example compiles clean
"""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lexer import Lexer
from parser import Parser
from ast_nodes import Program
from compiler import Compiler, CompileError
from ir import IRBuilder, BarrierOp, ShflOp


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_src(src: str) -> Program:
    return Parser(Lexer(src).tokenize()).parse()

def compile_src(src: str):
    return Compiler().compile_source(src)

def compile_all(src: str):
    return Compiler().compile_all_transmissions(src)

MINIMAL_TX = "transmission t { grid(1,1,1); block(256,1,1); motor m {} }"


# ── 1. syncthreads → bar.sync ─────────────────────────────────────────────────

def test_syncthreads_emits_bar_sync():
    src = f"""
motor m {{
    @register(R0) u64 a;
    @register(R1) u64 b;
    a = b + b;
    syncthreads();
    b = a + a;
}}
{MINIMAL_TX}
"""
    ptx, _ = compile_src(src)
    assert "bar.sync 0" in ptx


# ── 2. shfl_sync_u32 → shfl.sync.idx.b32 ─────────────────────────────────────

def test_shfl_sync_u32_emits_idx():
    src = f"""
motor m {{
    @register(R0) u64 result;
    u32 mask;
    u32 src;
    u32 lane;
    result = shfl_sync_u32(mask, src, lane);
}}
{MINIMAL_TX}
"""
    ptx, _ = compile_src(src)
    assert "shfl.sync.idx.b32" in ptx


# ── 3. shfl_up_sync_u32 → shfl.sync.up.b32 ───────────────────────────────────

def test_shfl_up_sync_u32_emits_up():
    src = f"""
motor m {{
    @register(R0) u64 result;
    u32 mask;
    u32 src;
    u32 delta;
    result = shfl_up_sync_u32(mask, src, delta);
}}
{MINIMAL_TX}
"""
    ptx, _ = compile_src(src)
    assert "shfl.sync.up.b32" in ptx


# ── 4. shfl_down_sync_u32 → shfl.sync.down.b32 ───────────────────────────────

def test_shfl_down_sync_u32_emits_down():
    src = f"""
motor m {{
    @register(R0) u64 result;
    u32 mask;
    u32 src;
    u32 delta;
    result = shfl_down_sync_u32(mask, src, delta);
}}
{MINIMAL_TX}
"""
    ptx, _ = compile_src(src)
    assert "shfl.sync.down.b32" in ptx


# ── 5. shfl_bfly_sync_u32 → shfl.sync.bfly.b32 ───────────────────────────────

def test_shfl_bfly_sync_u32_emits_bfly():
    src = f"""
motor m {{
    @register(R0) u64 result;
    u32 mask;
    u32 src;
    u32 xmask;
    result = shfl_bfly_sync_u32(mask, src, xmask);
}}
{MINIMAL_TX}
"""
    ptx, _ = compile_src(src)
    assert "shfl.sync.bfly.b32" in ptx


# ── 6. lane_id() → %laneid ────────────────────────────────────────────────────

def test_lane_id_emits_laneid():
    src = f"""
motor m {{
    @register(R0) u64 my_lane;
    my_lane = lane_id();
}}
{MINIMAL_TX}
"""
    ptx, _ = compile_src(src)
    assert "%laneid" in ptx


# ── 7. compile_all_transmissions: two motors → two .entry blocks ──────────────

TWO_MOTOR_SRC = """
motor walk {
    @window(R0..R3) u64 x[4];
    @register(R4) u64 out;
    out = x[0] + x[1];
}
motor hash {
    @window(R0..R3) u64 inp[4];
    @register(R4) u64 h;
    h = inp[0] + inp[2];
}
transmission walk_tx { grid(1,1,1); block(256,1,1); motor walk {
    inout registers(R0..R3) u64 x;
    inout R4 u64 out;
}}
transmission hash_tx { grid(1,1,1); block(256,1,1); motor hash {
    inout registers(R0..R3) u64 inp;
    inout R4 u64 h;
}}
"""

def test_compile_all_two_entry_blocks():
    ptx, prog = compile_all(TWO_MOTOR_SRC)
    entry_count = ptx.count(".entry ")
    assert entry_count == 2, f"Expected 2 .entry blocks, got {entry_count}"


# ── 8. shared header not duplicated ───────────────────────────────────────────

def test_compile_all_single_header():
    ptx, _ = compile_all(TWO_MOTOR_SRC)
    assert ptx.count(".version") == 1
    assert ptx.count(".address_size") == 1


# ── 9. sealed gate still fires in multi-kernel mode ──────────────────────────

def test_compile_all_sealed_gate_fires():
    src = """
profile tight { max_registers = 2; spill_allowed = false; }
motor a {
    @window(R0..R3) u64 x[4];
    @register(R4)  u64 y;
    y = x[0] + x[1];
}
motor b {
    @window(R0..R3) u64 p[4];
    @register(R4)  u64 q;
    q = p[0] + p[1];
}
transmission a_tx {
    target nvidia.sm_89;
    profile tight;
    require no_spills;
    require max_registers(2);
    motor a { inout registers(R0..R3) u64 x; inout R4 u64 y; }
}
transmission b_tx {
    grid(1,1,1); block(256,1,1);
    motor b { inout registers(R0..R3) u64 p; inout R4 u64 q; }
}
"""
    # max_registers(2) with 5 locked registers should trip the gate
    with pytest.raises(CompileError, match=r"register|spill|seal|FAIL"):
        compile_all(src)


# ── 10. hash160_split_engines.mtl compiles clean ─────────────────────────────

def test_split_engines_example_compiles():
    example = Path(__file__).parent.parent / "examples" / "hash160_split_engines.mtl"
    assert example.exists(), "hash160_split_engines.mtl not found"
    ptx, prog = Compiler().compile_all_file(example)
    assert ".entry " in ptx
    motor_names = [m.name for m in prog.motors]
    assert "point_walk_normalize" in motor_names
    assert "hash160_compare" in motor_names
    entry_count = ptx.count(".entry ")
    assert entry_count == 2, f"Expected 2 .entry blocks, got {entry_count}"
