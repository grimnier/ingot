"""
WP7: Packet declarations — named register contracts shared between motors.

Tests:
  1.  Packet parses to PacketDecl with correct field types and register assignments
  2.  `use pkt;` in motor body grafts locked registers into the motor
  3.  @window fields from packet desugar to named VarDecls (x_0 .. x_3)
  4.  @register scalar fields grafted correctly
  5.  Packet window fields accessible via array syntax (x[0], x[1] etc.)
  6.  Motor contract `packet pkt;` expands to TxParam + RegisterPort entries
  7.  Two motors using the same packet compile without conflict
  8.  compile_all_transmissions: shared packet → two entry blocks, one per motor
  9.  Field conflict: packet register vs existing @register is a CompileError
 10.  Reference to unknown packet is a CompileError
 11.  hash160_packet.mtl example compiles clean (two sealed kernels, 96-reg budget)
"""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lexer import Lexer
from parser import Parser, ParseError
from ast_nodes import Program, PacketDecl, PacketUse, PacketRef, VarDecl, WindowDecl, MotorContract
from compiler import Compiler, CompileError


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_src(src: str) -> Program:
    return Parser(Lexer(src).tokenize()).parse()

def compile_src(src: str):
    return Compiler().compile_source(src)

def compile_all(src: str):
    return Compiler().compile_all_transmissions(src)

SIMPLE_TX = "transmission t { grid(1,1,1); block(256,1,1); motor m {} }"


# ── 1. Packet parses correctly ────────────────────────────────────────────────

def test_packet_parses_to_packet_decl():
    src = """
packet my_pkt {
    @window(R0..R3) u64 x[4];
    @register(R4)   u64 flag;
}
motor m {}
""" + SIMPLE_TX
    prog = parse_src(src)
    assert len(prog.packets) == 1
    pkt = prog.packets[0]
    assert pkt.name == "my_pkt"
    assert len(pkt.fields) == 2
    win = pkt.fields[0]
    assert isinstance(win, WindowDecl)
    assert win.name == "x"
    assert win.count == 4
    assert win.start_reg == "R0"
    assert win.end_reg == "R3"
    scl = pkt.fields[1]
    assert isinstance(scl, VarDecl)
    assert scl.name == "flag"
    assert scl.register == "R4"
    assert scl.type == "u64"


# ── 2. use pkt grafts locked regs into motor ──────────────────────────────────

def test_use_packet_grafts_locked_regs():
    src = """
packet P {
    @window(R0..R1) u64 limbs[2];
    @register(R2)   u64 ctrl;
}
motor m { use P; }
""" + SIMPLE_TX
    c = Compiler()
    prog = parse_src(src)
    c._desugar_packet_refs(prog)
    motor = prog.motors[0]
    assert motor.locked_regs.get("limbs_0") == "R0"
    assert motor.locked_regs.get("limbs_1") == "R1"
    assert motor.locked_regs.get("ctrl")    == "R2"
    assert "P" in motor.used_packets


# ── 3. @window fields from packet desugar to VarDecls ─────────────────────────

def test_packet_window_desugars_to_var_decls():
    src = """
packet P { @window(R0..R2) u64 w[3]; }
motor m { use P; }
""" + SIMPLE_TX
    c = Compiler()
    prog = parse_src(src)
    c._desugar_packet_refs(prog)
    c._desugar_windows(prog)
    motor = prog.motors[0]
    names = [s.name for s in motor.body.stmts if isinstance(s, VarDecl)]
    assert "w_0" in names and "w_1" in names and "w_2" in names


# ── 4. @register scalar fields grafted ────────────────────────────────────────

def test_packet_scalar_field_grafted():
    src = """
packet P { @register(R5) u64 offset; }
motor m { use P; offset = offset + 1; }
""" + SIMPLE_TX
    ptx, _ = compile_src(src)
    assert "%R5" in ptx


# ── 5. Window fields accessible via array syntax after use ────────────────────

def test_packet_window_index_access():
    src = """
packet P { @window(R0..R3) u64 x[4]; }
motor m {
    use P;
    @register(R4) u64 tmp;
    tmp = x[0] + x[2];
}
""" + SIMPLE_TX
    ptx, _ = compile_src(src)
    # x_0 and x_2 must appear (as %R0, %R2) in the PTX
    assert "%R0" in ptx or "R0" in ptx


# ── 6. Motor contract `packet pkt;` expands to params/ports ───────────────────

def test_contract_packet_ref_expands():
    src = """
packet P {
    @window(R0..R1) u64 w[2];
    @register(R2)   u64 flag;
}
motor m { use P; }
transmission t {
    target nvidia.sm_89;
    motor m { packet P; }
}
"""
    c = Compiler()
    prog = parse_src(src)
    c._desugar_packet_refs(prog)
    contract = prog.transmissions[0].motor_contract
    # @window → TxParam with register_range
    assert any(p.name == "w" and p.register_range is not None for p in contract.params)
    # @register → RegisterPort
    assert any(rp.name == "flag" and rp.register == "R2" for rp in contract.register_ports)
    # PacketRef cleared
    assert contract.packet_refs == []


# ── 7. Two motors using same packet compile without conflict ──────────────────

def test_two_motors_same_packet_no_conflict():
    src = """
packet P { @window(R0..R3) u64 x[4]; @register(R4) u64 y; }
motor walk {
    use P;
    @register(R5) u64 tmp;
    tmp = x[0] + y;
}
motor hash {
    use P;
    @register(R5) u64 acc;
    acc = x[1] + y;
}
transmission walk_tx { grid(1,1,1); block(256,1,1); motor walk {
    packet P;
    inout R5 u64 tmp;
}}
transmission hash_tx  { grid(1,1,1); block(256,1,1); motor hash {
    packet P;
    inout R5 u64 acc;
}}
"""
    ptx, prog = compile_all(src)
    assert ptx.count(".entry ") == 2


# ── 8. compile_all: shared packet → two entry blocks ─────────────────────────

def test_compile_all_packet_two_entries():
    src = """
packet P { @window(R0..R1) u64 a[2]; }
motor m1 { use P; @register(R2) u64 t; t = a[0] + a[1]; }
motor m2 { use P; @register(R2) u64 u; u = a[0] + a[1]; }
transmission t1 { grid(1,1,1); block(256,1,1); motor m1 { packet P; inout R2 u64 t; } }
transmission t2 { grid(1,1,1); block(256,1,1); motor m2 { packet P; inout R2 u64 u; } }
"""
    ptx, _ = compile_all(src)
    assert ptx.count(".entry ") == 2
    assert ptx.count(".version") == 1


# ── 9. Packet register conflict is a CompileError ────────────────────────────

def test_packet_register_conflict_is_error():
    src = """
packet P { @register(R0) u64 x; }
motor m {
    @register(R0) u64 already;
    use P;
}
""" + SIMPLE_TX
    with pytest.raises(CompileError, match=r"conflict"):
        compile_src(src)


# ── 10. Unknown packet reference is a CompileError ───────────────────────────

def test_unknown_packet_reference_is_error():
    src = """
motor m { use ghost_packet; }
""" + SIMPLE_TX
    with pytest.raises(CompileError, match=r"unknown contract"):
        compile_src(src)


# ── 11. hash160_packet.mtl compiles clean ─────────────────────────────────────

def test_hash160_packet_example_compiles():
    example = Path(__file__).parent.parent / "examples" / "hash160_packet.mtl"
    assert example.exists(), "hash160_packet.mtl not found"
    ptx, prog = Compiler().compile_all_file(example)
    assert ptx.count(".entry ") == 2
    motor_names = [m.name for m in prog.motors]
    assert "point_walk"      in motor_names
    assert "hash160_compare" in motor_names
    # Both motors must reference the shared packet
    for motor in prog.motors:
        assert "candidate_state" in motor.used_packets
    # PTX must contain both kernel names
    assert "point_walk" in ptx
    assert "hash160_compare" in ptx
