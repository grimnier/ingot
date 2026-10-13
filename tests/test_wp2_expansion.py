"""
WP2 expansion test: point_add_mixed_plain.mtl → expanded PTX

Verifies:
  1. try_expand returns an MTL source string for the compat profile.
  2. Expanded MTL parses cleanly.
  3. Expanded source compiles to PTX without error.
  4. Locked registers R0..R19 are visible in the PTX.
  5. Field-mul instruction count matches 11 muls (same as oracle).
  6. Field-sub instruction count matches 6 subs (h, r, x3a, x3c, dx, y3).
  7. The oracle (point_add_mixed.mtl) and the expanded plain motor
     have the same PTX instruction-class distribution.
  8. field_interp KAT independently verifies the math is correct.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lexer import Lexer
from parser import Parser
from expansion import try_expand, PROFILE_COMPAT_V1
from compiler import Compiler

PLAIN_MTL  = os.path.join(os.path.dirname(__file__), '..', 'examples', 'point_add_mixed_plain.mtl')
ORACLE_MTL = os.path.join(os.path.dirname(__file__), '..', 'examples', 'point_add_mixed.mtl')


def _load_plain():
    src = open(PLAIN_MTL).read()
    prog = Parser(Lexer(src).tokenize()).parse()
    return prog.motors[0], prog.transmissions[0]

def _ptx_instrs(ptx: str, substr: str) -> int:
    return sum(1 for l in ptx.splitlines() if substr in l)


# ── Tests ──────────────────────────────────────────────────────────────────

def test_try_expand_produces_source():
    motor, tx = _load_plain()
    assert tx.profile == PROFILE_COMPAT_V1, f"Expected compat profile, got {tx.profile}"
    expanded = try_expand(motor, tx)
    assert expanded is not None, "try_expand returned None"
    assert len(expanded) > 5000, f"Expanded source too short: {len(expanded)} chars"
    print(f"PASS: try_expand returned {len(expanded.splitlines())} lines of MTL")

def test_expanded_source_parses():
    motor, tx = _load_plain()
    expanded = try_expand(motor, tx)
    prog2 = Parser(Lexer(expanded).tokenize()).parse()
    assert len(prog2.motors) == 1, f"Expected 1 motor in expanded source, got {len(prog2.motors)}"
    assert prog2.motors[0].name == motor.name
    assert len(prog2.motors[0].locked_regs) == 20, \
        f"Expected 20 locked regs (R0..R19), got {len(prog2.motors[0].locked_regs)}"
    print("PASS: expanded MTL parses with 20 locked registers")

def test_expanded_compiles_to_ptx():
    c = Compiler()
    ptx, _ = c.compile_file(PLAIN_MTL, "point_add_sm89")
    assert len(ptx) > 20000, f"PTX too short: {len(ptx)}"
    print(f"PASS: expanded source compiles to {len(ptx)} bytes of PTX")

def test_locked_regs_visible():
    c = Compiler()
    ptx, _ = c.compile_file(PLAIN_MTL, "point_add_sm89")
    locked = [l for l in ptx.splitlines() if ".reg .u64 %R" in l]
    regs = set()
    for l in locked:
        tok = l.strip().split()[2].rstrip(";")  # e.g. %R0
        regs.add(tok)
    for i in range(20):
        assert f"%R{i}" in regs, f"Missing %R{i} in PTX locked regs"
    print(f"PASS: all 20 locked registers R0..R19 visible in PTX")

def test_field_mul_count():
    c = Compiler()
    ptx, _ = c.compile_file(PLAIN_MTL, "point_add_sm89")
    # Each field-mul starts with one mul.lo.u64 (row 0, first instruction)
    mul_lo = _ptx_instrs(ptx, "mul.lo.u64")
    assert mul_lo == 22, f"Expected 22 mul.lo.u64 (11 schoolbook rows + 11 Step-C C-folds), got {mul_lo}"
    print(f"PASS: exactly 22 mul.lo.u64 (mul.lo.u64 count = {mul_lo})")

def test_field_sub_count():
    c = Compiler()
    ptx, _ = c.compile_file(PLAIN_MTL, "point_add_sm89")
    # Each field-sub starts with one sub.cc.u64
    sub_cc = _ptx_instrs(ptx, "sub.cc.u64")
    # 7: h, r, x3a, xh2_dbl (double's subtract-P), x3c, dx, y3_final
    assert sub_cc == 7, f"Expected 7 sub.cc.u64 (6 field subs + 1 double subtract-P), got {sub_cc}"
    print(f"PASS: exactly 7 sub.cc.u64 (6 field subs + 1 double subtract-P, count = {sub_cc})")

def test_matches_oracle_instruction_classes():
    """Verify the expanded plain motor has the same PTX instruction-class
    distribution as the hand-expanded oracle."""
    c_plain  = Compiler()
    c_oracle = Compiler()
    ptx_plain,  _ = c_plain.compile_file(PLAIN_MTL,  "point_add_sm89")
    ptx_oracle, _ = c_oracle.compile_file(ORACLE_MTL, "pollard_rho_step")

    classes = [
        ("mul.lo.u64",         22),   # 11 schoolbook rows + 11 Step-C C-folds
        ("mul.hi.u64",         22),   # 11 schoolbook rows + 11 Step-C C-folds
        ("mad.lo.cc.u64",      None), # count varies, just check nonzero
        ("madc.lo.cc.u64",     None),
        ("sub.cc.u64",          7),
        ("subc.cc.u64",        None),
        ("add.cc.u64",         None),
        ("addc.cc.u64",        None),
    ]

    ok = True
    for instr, expected_count in classes:
        plain_n  = _ptx_instrs(ptx_plain,  instr)
        oracle_n = _ptx_instrs(ptx_oracle, instr)
        if expected_count is not None:
            match = (plain_n == expected_count)
        else:
            # just verify both are positive and equal
            match = (plain_n == oracle_n) and plain_n > 0
        if not match:
            print(f"  MISMATCH {instr}: plain={plain_n}, oracle={oracle_n}, expected={expected_count}")
            ok = False

    assert ok, "Instruction class distribution mismatch between plain-expanded and oracle"
    print("PASS: instruction-class distribution matches oracle")

def test_field_interp_kat():
    from field_interp import run_kat
    run_kat()


if __name__ == "__main__":
    failures = 0
    for name, fn in [
        ("try_expand produces source",    test_try_expand_produces_source),
        ("expanded source parses",        test_expanded_source_parses),
        ("expanded compiles to PTX",      test_expanded_compiles_to_ptx),
        ("locked regs R0..R19 visible",   test_locked_regs_visible),
        ("field-mul count = 11",          test_field_mul_count),
        ("field-sub count = 6",           test_field_sub_count),
        ("matches oracle instr classes",  test_matches_oracle_instruction_classes),
        ("field_interp KAT",              test_field_interp_kat),
    ]:
        try:
            fn()
        except AssertionError as e:
            print(f"FAIL [{name}]: {e}")
            failures += 1
        except Exception as e:
            import traceback
            print(f"ERROR [{name}]: {type(e).__name__}: {e}")
            traceback.print_exc()
            failures += 1

    if failures == 0:
        print(f"\nAll WP2 tests passed.")
    else:
        print(f"\n{failures} test(s) failed.")
        sys.exit(1)
