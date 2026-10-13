"""
WP5: Register windows and sealed profile bodies.

Tests:
  1. @window annotation parses and desugars to locked VarDecls
  2. Window Index rewrites to correct register variable
  3. Out-of-range window index is a compile error
  4. Variable (non-constant) window index is a compile error
  5. Profile body parses (max_registers, spill_allowed, local_memory_allowed)
  6. Profile body referenced from transmission is resolved
  7. require no_spills / require no_local_memory parse correctly
  8. require max_registers(N) parses correctly
  9. Window register conflict is detected
 10. Window count mismatch (declared vs range) is a parse error
"""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lexer import Lexer
from parser import Parser, ParseError
from ast_nodes import WindowDecl, ProfileDecl, VarDecl, Var, Program
from compiler import Compiler, CompileError


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_src(src: str) -> Program:
    toks = Lexer(src).tokenize()
    return Parser(toks).parse()

def compile_src(src: str):
    return Compiler().compile_source(src)


# ── 1. window parses ──────────────────────────────────────────────────────────

def test_window_parses_to_window_decl():
    src = """
motor point_step {
    @window(R0..R3) u64 X[4];
    X[0] = X[1];
}
transmission t { grid(1,1,1); block(256,1,1); motor point_step {} }
"""
    prog = parse_src(src)
    # The WindowDecl should appear in the motor body before desugaring
    wdecl = next(s for s in prog.motors[0].body.stmts if isinstance(s, WindowDecl))
    assert wdecl.name == "X"
    assert wdecl.elem_type == "u64"
    assert wdecl.count == 4
    assert wdecl.start_reg == "R0"
    assert wdecl.end_reg == "R3"


# ── 2. desugar expands to locked VarDecls ─────────────────────────────────────

def test_window_desugars_to_locked_vars():
    src = """
motor walk {
    @window(R0..R3) u64 X[4];
}
transmission t { grid(1,1,1); block(256,1,1); motor walk {} }
"""
    c = Compiler()
    prog = parse_src(src)
    c._desugar_windows(prog)
    motor = prog.motors[0]
    # Original WindowDecl is gone; four VarDecl nodes exist
    assert not any(isinstance(s, WindowDecl) for s in motor.body.stmts)
    var_names = [s.name for s in motor.body.stmts if isinstance(s, VarDecl)]
    assert "X_0" in var_names
    assert "X_3" in var_names
    assert motor.locked_regs.get("X_0") == "R0"
    assert motor.locked_regs.get("X_3") == "R3"
    assert "X" in motor.windows


# ── 3. out-of-range index is compile error ─────────────────────────────────────

def test_window_out_of_range_index_is_error():
    src = """
motor bad {
    @window(R0..R3) u64 X[4];
    X[5] = X[0];
}
transmission t { grid(1,1,1); block(256,1,1); motor bad {} }
"""
    with pytest.raises(CompileError, match=r"out of range"):
        compile_src(src)


# ── 4. variable index is compile error ────────────────────────────────────────

def test_window_variable_index_is_error():
    src = """
motor bad {
    @window(R0..R3) u64 X[4];
    int i;
    X[i] = X[0];
}
transmission t { grid(1,1,1); block(256,1,1); motor bad {} }
"""
    with pytest.raises(CompileError, match=r"constant"):
        compile_src(src)


# ── 5. profile body parses ────────────────────────────────────────────────────

def test_profile_body_parses():
    src = """
profile hash160_no_spill {
    max_registers = 96;
    spill_allowed = false;
    local_memory_allowed = false;
}
motor m {}
transmission t { grid(1,1,1); block(256,1,1); motor m {} }
"""
    prog = parse_src(src)
    assert len(prog.profiles) == 1
    p = prog.profiles[0]
    assert p.name == "hash160_no_spill"
    assert p.properties["max_registers"] == 96
    assert p.properties["spill_allowed"] is False
    assert p.properties["local_memory_allowed"] is False


# ── 6. profile referenced from transmission is resolved ───────────────────────

def test_profile_referenced_from_transmission():
    src = """
profile tight {
    max_registers = 32;
}
motor m {
    @register(R0) u64 a;
}
transmission t {
    target nvidia.sm_89;
    profile tight;
    motor m {
        inout u64 a;
    }
}
"""
    prog = parse_src(src)
    assert prog.transmissions[0].profile == "tight"
    assert prog.profiles[0].name == "tight"


# ── 7. require no_spills parses ───────────────────────────────────────────────

def test_require_no_spills_parses():
    src = """
motor m {}
transmission t {
    require no_spills;
    grid(1,1,1); block(256,1,1);
    motor m {}
}
"""
    prog = parse_src(src)
    kinds = [r.kind for r in prog.transmissions[0].require_clauses]
    assert "no_spills" in kinds


# ── 8. require max_registers parses ───────────────────────────────────────────

def test_require_max_registers_parses():
    src = """
motor m {}
transmission t {
    require max_registers(96);
    grid(1,1,1); block(256,1,1);
    motor m {}
}
"""
    prog = parse_src(src)
    reqs = prog.transmissions[0].require_clauses
    mr = next(r for r in reqs if r.kind == "max_registers")
    assert mr.arg == 96


# ── 9. window register conflict is detected ───────────────────────────────────

def test_window_register_conflict_is_error():
    src = """
motor bad {
    @register(R0) u64 already_taken;
    @window(R0..R3) u64 X[4];
}
transmission t { grid(1,1,1); block(256,1,1); motor bad {} }
"""
    with pytest.raises(CompileError, match=r"conflict"):
        compile_src(src)


# ── 10. window count mismatch is parse error ──────────────────────────────────

def test_window_count_mismatch_is_parse_error():
    src = """
motor bad {
    @window(R0..R3) u64 X[8];
}
transmission t { grid(1,1,1); block(256,1,1); motor bad {} }
"""
    with pytest.raises(ParseError, match=r"does not match"):
        parse_src(src)


# ── 11. window without explicit size is fine ─────────────────────────────────

def test_window_no_explicit_size():
    src = """
motor m {
    @window(R4..R7) u64 Y;
}
transmission t { grid(1,1,1); block(256,1,1); motor m {} }
"""
    prog = parse_src(src)
    wdecl = next(s for s in prog.motors[0].body.stmts if isinstance(s, WindowDecl))
    assert wdecl.count == 4  # inferred from R4..R7
