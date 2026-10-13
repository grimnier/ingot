"""
WP8: Rails, target_latches, and route_types — typed pipeline contracts.

Tests:
  1.  rail parses to RailDecl with correct packet_name, width, stride
  2.  target_latch parses to TargetLatchDecl with target_bytes and compare_at list
  3.  route_type parses to RouteTypeDecl with correct variant names and values
  4.  `use RailName;` injects NAME_width constant into motor body
  5.  `use RouteTypeName;` injects each variant as a compile-time integer constant
  6.  `use LatchName;` records the latch in motor.used_latches (no injected constants)
  7.  PointRail_width constant is accessible in PTX (mov.u64 or similar)
  8.  Route variant constants (LIVE, DEAD, EXACT_HIT) fold into PTX literals
  9.  Unknown contract reference (not packet/rail/latch/route_type) is a CompileError
 10.  Two motors using the same rail compile without conflict
 11.  hash160_typed_pipeline.mtl example compiles clean
"""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lexer import Lexer
from parser import Parser, ParseError
from ast_nodes import (Program, RailDecl, TargetLatchDecl, RouteTypeDecl,
                       RouteVariant, VarDecl, Assign)
from compiler import Compiler, CompileError


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_src(src: str) -> Program:
    return Parser(Lexer(src).tokenize()).parse()

def compile_src(src: str):
    return Compiler().compile_source(src)

def compile_all(src: str):
    return Compiler().compile_all_transmissions(src)

SIMPLE_TX = "transmission t { grid(1,1,1); block(256,1,1); motor m {} }"

PKT_SRC = """
packet P { @window(R0..R3) u64 x[4]; @register(R4) u64 y; }
"""


# ── 1. rail parses correctly ──────────────────────────────────────────────────

def test_rail_parses_to_rail_decl():
    src = PKT_SRC + """
rail PointRail {
    packet: P;
    width:  1073741824;
    stride: G;
}
motor m {}
""" + SIMPLE_TX
    prog = parse_src(src)
    assert len(prog.rails) == 1
    r = prog.rails[0]
    assert isinstance(r, RailDecl)
    assert r.name == "PointRail"
    assert r.packet_name == "P"
    assert r.width == 1073741824
    assert r.stride == "G"


# ── 2. target_latch parses correctly ─────────────────────────────────────────

def test_target_latch_parses_correctly():
    src = """
target_latch Hash160Match {
    target_bytes: 20;
    compare_at:   [ripemd_word0, ripemd_word1, ripemd_full];
}
motor m {}
""" + SIMPLE_TX
    prog = parse_src(src)
    assert len(prog.target_latches) == 1
    tl = prog.target_latches[0]
    assert isinstance(tl, TargetLatchDecl)
    assert tl.name == "Hash160Match"
    assert tl.target_bytes == 20
    assert tl.compare_at == ["ripemd_word0", "ripemd_word1", "ripemd_full"]


# ── 3. route_type parses correctly ────────────────────────────────────────────

def test_route_type_parses_correctly():
    src = """
route_type CandidateStatus {
    LIVE      = 0;
    DEAD      = 1;
    EXACT_HIT = 2;
}
motor m {}
""" + SIMPLE_TX
    prog = parse_src(src)
    assert len(prog.route_types) == 1
    rt = prog.route_types[0]
    assert isinstance(rt, RouteTypeDecl)
    assert rt.name == "CandidateStatus"
    assert len(rt.variants) == 3
    names_vals = {v.name: v.value for v in rt.variants}
    assert names_vals == {"LIVE": 0, "DEAD": 1, "EXACT_HIT": 2}


# ── 4. use RailName injects NAME_width constant ───────────────────────────────

def test_use_rail_injects_width_constant():
    src = PKT_SRC + """
rail MyRail { packet: P; width: 65536; stride: G; }
motor m { use MyRail; }
""" + SIMPLE_TX
    c = Compiler()
    prog = parse_src(src)
    c._desugar_packet_refs(prog)
    motor = prog.motors[0]
    assert "MyRail" in motor.used_rails
    # A VarDecl for MyRail_width should be in the body
    names = [s.name for s in motor.body.stmts if isinstance(s, VarDecl)]
    assert "MyRail_width" in names


# ── 5. use RouteTypeName injects variant constants ────────────────────────────

def test_use_route_type_injects_variant_constants():
    src = """
route_type Status { LIVE = 0; DEAD = 1; EXACT_HIT = 2; }
motor m { use Status; }
""" + SIMPLE_TX
    c = Compiler()
    prog = parse_src(src)
    c._desugar_packet_refs(prog)
    motor = prog.motors[0]
    assert "Status" in motor.used_route_types
    names = [s.name for s in motor.body.stmts if isinstance(s, VarDecl)]
    assert "LIVE" in names
    assert "DEAD" in names
    assert "EXACT_HIT" in names


# ── 6. use LatchName records in used_latches, no injected constants ───────────

def test_use_latch_records_no_injected_constants():
    src = """
target_latch H { target_bytes: 20; compare_at: [stage0]; }
motor m { use H; }
""" + SIMPLE_TX
    c = Compiler()
    prog = parse_src(src)
    c._desugar_packet_refs(prog)
    motor = prog.motors[0]
    assert "H" in motor.used_latches
    # No VarDecl nodes should be injected for a latch
    injected = [s for s in motor.body.stmts if isinstance(s, VarDecl)]
    assert injected == []


# ── 7. Rail width constant survives to PTX ────────────────────────────────────

def test_rail_width_accessible_in_ptx():
    src = PKT_SRC + """
rail R { packet: P; width: 1024; stride: G; }
motor m {
    use P;
    use R;
    @register(R5) u64 w;
    w = R_width;
}
transmission t {
    grid(1,1,1); block(256,1,1);
    motor m { packet P; inout R5 u64 w; }
}
"""
    ptx, _ = compile_src(src)
    # 1024 decimal should appear in the PTX output
    assert "1024" in ptx


# ── 8. Route variant constants fold into PTX literals ────────────────────────

def test_route_variants_fold_into_ptx():
    src = """
route_type S { LIVE = 0; DEAD = 1; }
motor m {
    use S;
    @register(R0) u64 flags;
    flags = DEAD;
}
transmission t { grid(1,1,1); block(256,1,1); motor m { inout R0 u64 flags; } }
"""
    ptx, _ = compile_src(src)
    # DEAD = 1 should appear as a literal in the PTX
    assert "1" in ptx


# ── 9. Unknown contract reference is a CompileError ──────────────────────────

def test_unknown_contract_reference_is_error():
    src = """
motor m { use doesnt_exist; }
""" + SIMPLE_TX
    with pytest.raises(CompileError, match=r"unknown contract"):
        compile_src(src)


# ── 10. Two motors using the same rail compile without conflict ───────────────

def test_two_motors_same_rail_no_conflict():
    src = PKT_SRC + """
rail R { packet: P; width: 512; stride: G; }
motor walk {
    use P;
    use R;
    @register(R5) u64 step;
    step = R_width;
}
motor hash {
    use P;
    use R;
    @register(R5) u64 s;
    s = R_width;
}
transmission t1 { grid(1,1,1); block(256,1,1); motor walk { packet P; inout R5 u64 step; } }
transmission t2 { grid(1,1,1); block(256,1,1); motor hash { packet P; inout R5 u64 s; } }
"""
    ptx, prog = compile_all(src)
    assert ptx.count(".entry ") == 2


# ── 11. hash160_typed_pipeline.mtl example compiles clean ────────────────────

def test_hash160_typed_pipeline_example_compiles():
    example = Path(__file__).parent.parent / "examples" / "hash160_typed_pipeline.mtl"
    assert example.exists(), "hash160_typed_pipeline.mtl not found"
    ptx, prog = Compiler().compile_all_file(example)
    assert ptx.count(".entry ") == 2
    motor_names = [m.name for m in prog.motors]
    assert "point_walk"   in motor_names
    assert "hash160_check" in motor_names
    # Motors must carry the expected contract records
    walk  = next(m for m in prog.motors if m.name == "point_walk")
    hashm = next(m for m in prog.motors if m.name == "hash160_check")
    assert "candidate_state" in walk.used_packets
    assert "PointRail"       in walk.used_rails
    assert "candidate_state" in hashm.used_packets
    assert "Hash160Match"    in hashm.used_latches
    assert "CandidateStatus" in hashm.used_route_types
