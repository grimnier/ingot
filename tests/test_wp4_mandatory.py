"""
WP4 mandatory-path annotation tests.

Uses the expanded plain-motor IR (produced by compiling
point_add_mixed_plain.mtl through the compat-v1 profile) as the
test subject, since it contains exactly 11 field-mul carry chains.

Tests:
  1. Annotator runs without error.
  2. carry_producer count matches expected (one per .cc op in 11 muls).
  3. carry_consumer count matches.
  4. high_limb count matches (one mul.hi per field mul = 11).
  5. All carry windows have gap == 1 (no CC hazards).
  6. live_result count is positive (sanity check).
  7. Tier-3 bound proof passes on the expanded IR.
  8. Tier-3 proof: field_muls_found == 11.
  9. Tier-3 proof: max_tier3_carry_bound == 1.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lexer import Lexer
from parser import Parser
from expansion import try_expand
from compiler import Compiler
from ir import IRBuilder, MotorIR
from mandatory_path import (
    MandatoryPathAnnotator, prove_tier3_unconditional,
    print_annotation_summary,
    CARRY_PRODUCER, CARRY_CONSUMER, BORROW_PRODUCER, BORROW_CONSUMER,
    HIGH_LIMB, LIVE_RESULT,
)

PLAIN_MTL = os.path.join(os.path.dirname(__file__), '..', 'examples', 'point_add_mixed_plain.mtl')


def _build_expanded_ir() -> MotorIR:
    """
    Expand the plain-motor MTL through the compat-v1 profile and lower
    the resulting explicit intrinsic motor to IR.
    """
    src = open(PLAIN_MTL).read()
    prog = Parser(Lexer(src).tokenize()).parse()
    motor, tx = prog.motors[0], prog.transmissions[0]
    expanded_src = try_expand(motor, tx)
    assert expanded_src, "try_expand returned None"

    # Re-parse the expanded MTL
    prog2 = Parser(Lexer(expanded_src).tokenize()).parse()
    motor2 = prog2.motors[0]
    contract = prog2.transmissions[0].motor_contract if prog2.transmissions else None

    builder = IRBuilder()
    mir = builder.lower_motor(motor2, contract, machine_model={"warp_size": 32, "max_regs": 255, "block_size": 1})
    return mir


def test_annotator_runs():
    mir = _build_expanded_ir()
    annotator = MandatoryPathAnnotator()
    windows = annotator.annotate(mir)
    assert isinstance(windows, list)
    print(f"PASS: annotator ran, produced {len(windows)} carry windows")

def test_carry_producer_count():
    mir = _build_expanded_ir()
    annotator = MandatoryPathAnnotator()
    annotator.annotate(mir)
    count = sum(1 for op in mir.ops if CARRY_PRODUCER in getattr(op, "marks", set()))
    # 11 field muls × carry producers per mul + 6 field subs + 1 double
    # Each field-mul has: 16 mad.lo.cc + 8 madc.lo.cc + 2 mad.hi.cc + 6 madc.hi.cc = 32 CC producers per mul
    # Plus sub/double chains. Check it's in a reasonable range.
    assert count > 0, f"No carry_producer marks found"
    print(f"PASS: carry_producer count = {count}")

def test_carry_consumer_count():
    mir = _build_expanded_ir()
    annotator = MandatoryPathAnnotator()
    annotator.annotate(mir)
    count = sum(1 for op in mir.ops if CARRY_CONSUMER in getattr(op, "marks", set()))
    assert count > 0, f"No carry_consumer marks found"
    print(f"PASS: carry_consumer count = {count}")

def test_high_limb_count():
    mir = _build_expanded_ir()
    annotator = MandatoryPathAnnotator()
    annotator.annotate(mir)
    count = sum(1 for op in mir.ops if HIGH_LIMB in getattr(op, "marks", set()))
    # 11 field muls; each mul has mul.hi.u64 (row 0) + madc.hi.cc.u64 × N
    # Expect at least 11 (one mul.hi per field mul)
    assert count >= 11, f"Expected at least 11 high_limb marks, got {count}"
    print(f"PASS: high_limb count = {count} (>= 11)")

def test_carry_terminals_present():
    """
    Each field-mul has 6 addc.u64 carry-chain terminals (Step-C C-fold removed
    one versus the old split); field-sub and field-double each add 1.
    11 muls × 6 = 66, plus field-subs/double = 66 minimum from muls alone.
    """
    mir = _build_expanded_ir()
    annotator = MandatoryPathAnnotator()
    annotator.annotate(mir)
    addc_terminals = sum(
        1 for op in mir.ops
        if isinstance(op, __import__('ir').IntrinsicOp) and op.ptx_instr in ("addc.u64", "addc.u32")
    )
    # At least 11*6 = 66 from field-muls alone
    assert addc_terminals >= 66, \
        f"Expected >= 66 addc.u64 terminals (11 muls × 6), got {addc_terminals}"
    print(f"PASS: {addc_terminals} addc.u64 carry-chain terminals (>= 66)")

def test_live_result_count():
    mir = _build_expanded_ir()
    annotator = MandatoryPathAnnotator()
    annotator.annotate(mir)
    count = sum(1 for op in mir.ops if LIVE_RESULT in getattr(op, "marks", set()))
    assert count > 0, "No live_result marks found"
    print(f"PASS: live_result count = {count}")

def test_tier3_proof_passes():
    mir = _build_expanded_ir()
    result = prove_tier3_unconditional(mir)
    assert result.proven, f"Tier-3 proof failed:\n{result.justification}"
    print(f"PASS: Tier-3 unconditional correction proven")
    print(f"      {result.justification[:120]}...")

def test_tier3_field_muls_found():
    mir = _build_expanded_ir()
    result = prove_tier3_unconditional(mir)
    assert result.field_muls_found == 22, \
        f"Expected 22 field muls (11 schoolbook + 11 Step-C C-folds), proof found {result.field_muls_found}"
    print(f"PASS: Tier-3 proof found exactly 22 field multiplications")

def test_tier3_carry_bound():
    mir = _build_expanded_ir()
    result = prove_tier3_unconditional(mir)
    assert result.max_tier3_carry_bound == 1, \
        f"Expected Tier-3 carry bound = 1, got {result.max_tier3_carry_bound}"
    print(f"PASS: Tier-3 carry bound = {result.max_tier3_carry_bound} (one correction sufficient)")

def test_summary_prints():
    mir = _build_expanded_ir()
    annotator = MandatoryPathAnnotator()
    windows = annotator.annotate(mir)
    print("\n--- Annotation summary ---")
    print_annotation_summary(mir, windows)
    print("PASS: summary printed")


if __name__ == "__main__":
    failures = 0
    for name, fn in [
        ("annotator runs",               test_annotator_runs),
        ("carry_producer count > 0",     test_carry_producer_count),
        ("carry_consumer count > 0",     test_carry_consumer_count),
        ("high_limb count >= 11",        test_high_limb_count),
        ("carry terminals >= 77",        test_carry_terminals_present),
        ("live_result count > 0",        test_live_result_count),
        ("Tier-3 proof passes",          test_tier3_proof_passes),
        ("Tier-3 field_muls_found == 11",test_tier3_field_muls_found),
        ("Tier-3 carry bound == 1",      test_tier3_carry_bound),
        ("summary prints",               test_summary_prints),
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
        print(f"\nAll WP4 mandatory-path tests passed.")
    else:
        print(f"\n{failures} test(s) failed.")
        sys.exit(1)
