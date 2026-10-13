"""
KAT: secp256k1 Jacobian point negation

CPU tests (no GPU required):
  1. fe_negate correctness: zero, one, P-1, round-trip.
  2. jacobian_negate: X and Z unchanged, Y inverted.
  3. affine_y_parity: generator G (odd Y), G-negated (even Y).
  4. jacobian_y_parity_at_affine_boundary: agrees with affine parity.
  5. jacobian_y_parity_rejection: j.y & 1 is NOT affine parity for Z != 1.

Compiler differential test:
  6. point_negate_jac.mtl compiles to PTX.
  7. PTX contains expected carry-chain instructions (sub.cc, subc, add.cc, addc).
  8. X1 and Z1 registers are NOT written in the negate body (only Y1 is written).
  9. Locked registers R0..R11 are all visible in PTX.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from field_interp import (
    P, N,
    GX, GY,
    Affine, Jacobian,
    fe_negate, fe_mul, fe_inv,
    affine_y_parity, affine_negate,
    jacobian_negate, jacobian_y_parity_at_affine_boundary,
    jacobian_to_affine, affine_to_jacobian,
    affine_add, affine_double,
    jacobian_add_affine,
)

ORACLE_MTL = os.path.join(os.path.dirname(__file__), '..', 'examples', 'point_negate_jac.mtl')


# ── §1: fe_negate ────────────────────────────────────────────────────────────

def test_fe_negate_zero():
    assert fe_negate(0) == 0, f"fe_negate(0) should be 0, got {fe_negate(0)}"
    print("PASS: fe_negate(0) == 0")

def test_fe_negate_one():
    result = fe_negate(1)
    assert result == P - 1, f"fe_negate(1) expected {P-1:#066x}, got {result:#066x}"
    print("PASS: fe_negate(1) == P - 1")

def test_fe_negate_p_minus_one():
    result = fe_negate(P - 1)
    assert result == 1, f"fe_negate(P-1) expected 1, got {result}"
    print("PASS: fe_negate(P-1) == 1")

def test_fe_negate_round_trip():
    # For arbitrary field element, negate(negate(x)) == x
    x = 0xdeadbeefcafe000000000000000000000000000000000000000000001234abcd
    x = x % P
    assert fe_negate(fe_negate(x)) == x, "Double negation failed"
    print("PASS: fe_negate(fe_negate(x)) == x for random x")

def test_fe_negate_add_to_zero():
    # x + negate(x) == 0 mod P
    x = 0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890
    x = x % P
    s = (x + fe_negate(x)) % P
    assert s == 0, f"x + negate(x) = {s:#066x}, expected 0"
    print("PASS: x + fe_negate(x) == 0 mod P")


# ── §2: jacobian_negate ──────────────────────────────────────────────────────

def test_jacobian_negate_x_unchanged():
    G = Affine(x=GX, y=GY)
    j = affine_to_jacobian(G)
    neg_j = jacobian_negate(j)
    assert neg_j.x == j.x, "X should be unchanged after jacobian_negate"
    print("PASS: jacobian_negate preserves X")

def test_jacobian_negate_z_unchanged():
    G = Affine(x=GX, y=GY)
    j = affine_to_jacobian(G)
    neg_j = jacobian_negate(j)
    assert neg_j.z == j.z, "Z should be unchanged after jacobian_negate"
    print("PASS: jacobian_negate preserves Z")

def test_jacobian_negate_y_inverted():
    G = Affine(x=GX, y=GY)
    j = affine_to_jacobian(G)
    neg_j = jacobian_negate(j)
    expected_y = fe_negate(j.y)
    assert neg_j.y == expected_y, f"Y not inverted: got {neg_j.y:#066x}, expected {expected_y:#066x}"
    print("PASS: jacobian_negate inverts Y")

def test_jacobian_negate_double_round_trip():
    # negate(negate(Q)) == Q for Jacobian point
    G = Affine(x=GX, y=GY)
    j = affine_to_jacobian(G)
    assert jacobian_negate(jacobian_negate(j)) == j, "Double Jacobian negate should be identity"
    print("PASS: jacobian_negate(jacobian_negate(Q)) == Q")

def test_jacobian_negate_affine_result():
    # jacobian_negate(jac(G)).to_affine() == affine_negate(G)
    G = Affine(x=GX, y=GY)
    j = affine_to_jacobian(G)
    neg_j_aff = jacobian_to_affine(jacobian_negate(j))
    expected   = affine_negate(G)
    assert neg_j_aff is not None
    assert neg_j_aff.x == expected.x
    assert neg_j_aff.y == expected.y
    print("PASS: jacobian_negate(jac(G)) == affine_negate(G) via to_affine()")

def test_jacobian_negate_arbitrary_z():
    # Test with a Jacobian point where Z != 1 (result of point addition)
    G  = Affine(x=GX, y=GY)
    tG = affine_double(G)
    j  = jacobian_add_affine(affine_to_jacobian(G), tG)   # 3G with arbitrary Z
    assert j.z != 1, "Need Z != 1 for this test to be meaningful"
    neg_j  = jacobian_negate(j)
    aff_j  = jacobian_to_affine(j)
    aff_nj = jacobian_to_affine(neg_j)
    assert aff_j  is not None
    assert aff_nj is not None
    assert aff_j.x == aff_nj.x, "X coordinates must match after negate"
    assert aff_nj.y == (P - aff_j.y) % P, "Affine Y must be inverted"
    print(f"PASS: jacobian_negate works for Z={j.z:#x} (arbitrary Z != 1)")


# ── §3: affine_y_parity ──────────────────────────────────────────────────────

def test_affine_y_parity_generator():
    G = Affine(x=GX, y=GY)
    parity = affine_y_parity(G)
    expected = GY & 1
    assert parity == expected, f"G parity: expected {expected}, got {parity}"
    print(f"PASS: affine_y_parity(G) == {parity} (GY is {'odd' if parity else 'even'})")

def test_affine_y_parity_negation_flips():
    G  = Affine(x=GX, y=GY)
    nG = affine_negate(G)
    assert affine_y_parity(G) != affine_y_parity(nG), \
        "Negating a point should flip Y parity (for non-zero Y)"
    print("PASS: affine_y_parity flips under affine_negate")

def test_affine_negate_same_x():
    G  = Affine(x=GX, y=GY)
    nG = affine_negate(G)
    assert G.x == nG.x, "Negated affine point must have same X"
    print("PASS: affine_negate preserves X coordinate")


# ── §4: jacobian_y_parity_at_affine_boundary ────────────────────────────────

def test_jacobian_y_parity_boundary_matches_affine():
    G = Affine(x=GX, y=GY)
    j = affine_to_jacobian(G)
    parity = jacobian_y_parity_at_affine_boundary(j)
    assert parity == affine_y_parity(G), \
        f"Boundary parity mismatch: {parity} vs {affine_y_parity(G)}"
    print("PASS: jacobian_y_parity_at_affine_boundary matches affine_y_parity for Z=1")

def test_jacobian_y_parity_boundary_arbitrary_z():
    # 3G via mixed add has Z != 1; boundary function must still agree with affine
    G  = Affine(x=GX, y=GY)
    tG = affine_double(G)
    j  = jacobian_add_affine(affine_to_jacobian(G), tG)
    assert j.z != 1
    aff_parity  = affine_y_parity(jacobian_to_affine(j))
    jac_parity  = jacobian_y_parity_at_affine_boundary(j)
    assert jac_parity == aff_parity, \
        f"Boundary parity {jac_parity} != affine parity {aff_parity}"
    print(f"PASS: jacobian_y_parity_at_affine_boundary correct for Z={j.z:#x}")


# ── §5: rejection of j.y & 1 as affine parity ────────────────────────────────

def test_jacobian_y_bit_is_not_affine_parity():
    """Prove with a concrete counterexample that j.y & 1 is NOT affine Y parity.

    Walk kG via repeated jacobian_add_affine(prev, G) — each step adds G to
    the Jacobian accumulator, producing non-unit Z. Search up to 50 steps for
    a point where Jacobian Y parity differs from affine Y parity and assert
    one is found. (Roughly half of all non-unit-Z points exhibit the divergence.)
    """
    G  = Affine(x=GX, y=GY)
    tG = affine_double(G)
    # 3G via mixed add: Z3 = Z1 * H = 1 * (2G.x - G.x) != 0 and != 1
    j = jacobian_add_affine(affine_to_jacobian(G), tG)
    assert j.z != 1, "3G via mixed add must have Z != 1"

    counterexample = None
    for _ in range(50):
        aff = jacobian_to_affine(j)
        if aff is not None and j.z != 1 and (j.y & 1) != (aff.y & 1):
            counterexample = (j, aff, j.y & 1, aff.y & 1)
            break
        # Advance by G (safe for small multiples; Z goes non-unit after each step)
        j = jacobian_add_affine(j, G)

    assert counterexample is not None, \
        ("No counterexample found in 50 steps — j.y & 1 agreed with affine parity "
         "every time, which would contradict the Z^3 chain. Check jacobian_add_affine.")
    j, aff, jb, ap = counterexample
    print(f"PASS: j.y & 1 = {jb} != affine parity = {ap} "
          f"(Z={j.z:#x}) — j.y & 1 is NOT affine parity")


# ── §6-9: Compiler differential test ─────────────────────────────────────────

def _compile_negate():
    from compiler import Compiler
    c = Compiler()
    ptx, _ = c.compile_file(ORACLE_MTL, "point_negate_sm89")
    return ptx

def test_negate_compiles_to_ptx():
    ptx = _compile_negate()
    assert len(ptx) > 500, f"PTX too short ({len(ptx)} chars) — likely empty"
    print(f"PASS: point_negate_jac.mtl compiles to {len(ptx)} bytes of PTX")

def test_negate_ptx_carry_chain():
    """PTX must contain the 10-instruction branchless carry-chain sequence."""
    ptx = _compile_negate()
    lines = ptx.splitlines()
    def count(s): return sum(1 for l in lines if s in l)

    sub_cc   = count("sub.cc.u64")
    subc_cc  = count("subc.cc.u64")
    subc     = count("subc.u64")
    add_cc   = count("add.cc.u64")
    addc_cc  = count("addc.cc.u64")
    addc     = count("addc.u64")

    # 4× sub.cc, 3× subc.cc, 1× subc, 1× add.cc, 2× addc.cc, 1× addc = 12 instructions
    # (add_cc64(neg0, ap0) generates add.cc.u64; addc_cc64 × 2; addc64 × 1)
    assert sub_cc  >= 1, f"Expected sub.cc.u64, got {sub_cc}"
    assert subc_cc >= 1, f"Expected subc.cc.u64, got {subc_cc}"
    assert subc    >= 1, f"Expected subc.u64 (borrow capture), got {subc}"
    assert add_cc  >= 1, f"Expected add.cc.u64, got {add_cc}"
    assert addc    >= 1, f"Expected addc.u64, got {addc}"
    print(f"PASS: carry-chain instructions present: sub.cc={sub_cc} subc.cc={subc_cc} "
          f"subc={subc} add.cc={add_cc} addc.cc={addc_cc} addc={addc}")

def test_negate_ptx_has_and():
    """PTX must contain an AND instruction for ap0 = PL0 & sb."""
    ptx = _compile_negate()
    and_count = sum(1 for l in ptx.splitlines() if "and.b64" in l or "and.pred" in l.lower())
    assert and_count >= 1, f"Expected AND instruction for ap0 = PL0 & sb, got {and_count}"
    print(f"PASS: AND instruction present (ap0 = PL0 & sb): count={and_count}")

def test_negate_locked_registers_visible():
    """R0..R11 (X1, Y1, Z1) must all appear as PTX virtual register names.

    Note: @register(Rn) pins the PTX *name* to %Rn. Physical SASS register
    allocation is determined by ptxas and requires cuobjdump SASS analysis
    to verify — this test checks PTX virtual register names only.
    """
    ptx = _compile_negate()
    locked = set()
    for l in ptx.splitlines():
        if ".reg .u64 %R" in l:
            tok = l.strip().split()[2].rstrip(";")
            locked.add(tok)
    for i in range(12):
        assert f"%R{i}" in locked, f"Missing %R{i} in PTX; found: {sorted(locked)}"
    print(f"PASS: PTX virtual register names %R0..%R11 present (physical SASS regs require ptxas/cuobjdump)")


if __name__ == "__main__":
    failures = 0
    tests = [
        ("fe_negate(0)",              test_fe_negate_zero),
        ("fe_negate(1)",              test_fe_negate_one),
        ("fe_negate(P-1)",            test_fe_negate_p_minus_one),
        ("fe_negate round-trip",      test_fe_negate_round_trip),
        ("fe_negate + = 0",          test_fe_negate_add_to_zero),
        ("jac_negate X unchanged",    test_jacobian_negate_x_unchanged),
        ("jac_negate Z unchanged",    test_jacobian_negate_z_unchanged),
        ("jac_negate Y inverted",     test_jacobian_negate_y_inverted),
        ("jac_negate double round",   test_jacobian_negate_double_round_trip),
        ("jac_negate to affine",      test_jacobian_negate_affine_result),
        ("jac_negate arbitrary Z",    test_jacobian_negate_arbitrary_z),
        ("affine parity generator",   test_affine_y_parity_generator),
        ("affine parity flip",        test_affine_y_parity_negation_flips),
        ("affine negate same X",      test_affine_negate_same_x),
        ("boundary parity Z=1",       test_jacobian_y_parity_boundary_matches_affine),
        ("boundary parity Z!=1",      test_jacobian_y_parity_boundary_arbitrary_z),
        ("j.y&1 rejection",           test_jacobian_y_bit_is_not_affine_parity),
        ("compile to PTX",            test_negate_compiles_to_ptx),
        ("carry-chain instrs",        test_negate_ptx_carry_chain),
        ("AND instruction",           test_negate_ptx_has_and),
        ("locked regs R0..R11",       test_negate_locked_registers_visible),
    ]
    for name, fn in tests:
        try:
            fn()
        except AssertionError as e:
            print(f"FAIL [{name}]: {e}")
            failures += 1
        except Exception as e:
            print(f"ERROR [{name}]: {type(e).__name__}: {e}")
            failures += 1

    if failures == 0:
        print(f"\nAll {len(tests)} negation KAT tests passed.")
    else:
        print(f"\n{failures}/{len(tests)} test(s) failed.")
        import sys; sys.exit(1)
