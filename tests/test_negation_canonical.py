"""
Canonicalization proofs: Livethine Negation-Map Black-Hole Contract §3-§6

Tests:
  1. target_sheet_from_compressed_prefix: 0x02 → even, 0x03 → odd, bad byte raises.
  2. canonicalize(Q) and canonicalize(-Q) produce the same canonical point.
  3. Every canonical output has the correct sheet parity.
  4. Canonical X equality on the same sheet implies same point (uniqueness on sheet).
  5. recover_scalar round-trips: recover(n-k, True) == k; recover(k, False) == k.
  6. recover_scalar(0, True) == 0 (edge case: negation of zero scalar).
  7. negation_orbit_canon: lower-Y representative is stable (idempotent).
  8. negation_orbit_canon: negated_flag is True iff Q was on the higher-Y half.
  9. Interval scan: black_hole_scan finds the correct scalar and sets negated_flag.
 10. Interval scan: scan of an empty interval returns None.
 11. proof_canonicalize_q_equals_neg_q passes for G, 2G, 3G, random multiples.
 12. proof_canonical_x_implies_same_point passes for equal and unequal points.
 13. proof_recover_scalar_roundtrip passes for k = 1, 2, n-1, random.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from field_interp import (
    P, N,
    GX, GY,
    Affine, Jacobian,
    fe_negate,
    jacobian_negate,
    affine_y_parity, affine_negate,
    jacobian_to_affine, affine_to_jacobian,
    affine_add, affine_double,
)
from point_negate import (
    TargetSheet, TargetSheetSource, CanonicalPoint, NegationMapHit,
    canonicalize_negation_pair,
    recover_scalar,
    negation_orbit_canon,
    target_sheet_from_compressed_prefix,
    black_hole_scan_with_negation_map,
    proof_canonicalize_q_equals_neg_q,
    proof_canonical_x_implies_same_point,
    proof_recover_scalar_roundtrip,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

G = Affine(x=GX, y=GY)

def scalar_mul_affine(k: int) -> Affine:
    """Compute k*G by repeated doubling."""
    k = k % N
    if k == 0:
        raise ValueError("scalar 0 gives point at infinity")
    result = None
    addend = G
    while k:
        if k & 1:
            result = addend if result is None else affine_add(result, addend)
        addend = affine_double(addend)
        k >>= 1
    return result

def scalar_mul_jacobian(k: int) -> Jacobian:
    return affine_to_jacobian(scalar_mul_affine(k))

def even_sheet() -> TargetSheet:
    return TargetSheet(y_parity=0, source=TargetSheetSource.KnownCompressedPublicKeySheet)

def odd_sheet() -> TargetSheet:
    return TargetSheet(y_parity=1, source=TargetSheetSource.KnownCompressedPublicKeySheet)


# ── §1: target_sheet_from_compressed_prefix ──────────────────────────────────

def test_prefix_02_even():
    s = target_sheet_from_compressed_prefix(0x02)
    assert s.y_parity == 0
    assert s.source == TargetSheetSource.KnownCompressedPublicKeySheet
    print("PASS: prefix 0x02 → even sheet")

def test_prefix_03_odd():
    s = target_sheet_from_compressed_prefix(0x03)
    assert s.y_parity == 1
    print("PASS: prefix 0x03 → odd sheet")

def test_prefix_invalid_raises():
    try:
        target_sheet_from_compressed_prefix(0x04)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("PASS: invalid prefix 0x04 raises ValueError")


# ── §2: canonicalize(Q) == canonicalize(-Q) ──────────────────────────────────

def _check_canon_pair(k: int, sheet: TargetSheet, label: str):
    q   = scalar_mul_jacobian(k)
    neg_q = jacobian_negate(q)
    cp_q,    _  = canonicalize_negation_pair(q,     sheet, source_position=k)
    cp_negq, _  = canonicalize_negation_pair(neg_q, sheet, source_position=N - k)
    aff_q    = jacobian_to_affine(cp_q.point)
    aff_negq = jacobian_to_affine(cp_negq.point)
    assert aff_q    is not None
    assert aff_negq is not None
    assert aff_q.x == aff_negq.x, f"{label}: canonical X mismatch"
    assert aff_q.y == aff_negq.y, f"{label}: canonical Y mismatch"
    return True

def test_canon_pair_generator():
    _check_canon_pair(1, even_sheet(), "G even-sheet")
    _check_canon_pair(1, odd_sheet(),  "G odd-sheet")
    print("PASS: canonicalize(G) == canonicalize(-G) on both sheets")

def test_canon_pair_2g():
    _check_canon_pair(2, even_sheet(), "2G even-sheet")
    _check_canon_pair(2, odd_sheet(),  "2G odd-sheet")
    print("PASS: canonicalize(2G) == canonicalize(-2G)")

def test_canon_pair_3g():
    _check_canon_pair(3, even_sheet(), "3G even-sheet")
    print("PASS: canonicalize(3G) == canonicalize(-3G)")

def test_canon_pair_multiple_scalars():
    sheet = even_sheet()
    for k in [5, 7, 11, 100, 12345]:
        _check_canon_pair(k, sheet, f"{k}G")
    print("PASS: canonicalize(Q) == canonicalize(-Q) for k in [5,7,11,100,12345]")


# ── §3: Every canonical output has correct sheet parity ──────────────────────

def _check_sheet_parity(k: int, sheet: TargetSheet, label: str):
    q  = scalar_mul_jacobian(k)
    cp, _ = canonicalize_negation_pair(q, sheet, source_position=k)
    aff = jacobian_to_affine(cp.point)
    assert aff is not None
    parity = affine_y_parity(aff)
    assert parity == sheet.y_parity, \
        f"{label}: canonical parity {parity} != sheet parity {sheet.y_parity}"

def test_canonical_output_correct_parity_even():
    sheet = even_sheet()
    for k in [1, 2, 3, 5, 7, 100]:
        _check_sheet_parity(k, sheet, f"{k}G even-sheet")
    print("PASS: all canonical outputs on even-sheet have even Y")

def test_canonical_output_correct_parity_odd():
    sheet = odd_sheet()
    for k in [1, 2, 3, 5, 7, 100]:
        _check_sheet_parity(k, sheet, f"{k}G odd-sheet")
    print("PASS: all canonical outputs on odd-sheet have odd Y")


# ── §4: Canonical X equality on same sheet → same point ──────────────────────

def test_canonical_x_same_sheet_unique():
    """Two canonical points with the same X on the same sheet must be equal."""
    sheet = even_sheet()
    # Canonicalize k and n-k onto the same sheet — they should produce the same point
    for k in [1, 2, 5, 17]:
        q_k    = scalar_mul_jacobian(k)
        q_negk = scalar_mul_jacobian(N - k)   # == -kG
        cp_k,    _ = canonicalize_negation_pair(q_k,    sheet, source_position=k)
        cp_negk, _ = canonicalize_negation_pair(q_negk, sheet, source_position=N - k)
        # Both are canonical on the same sheet; should be equal
        proof_canonical_x_implies_same_point(cp_k, cp_negk)
    print("PASS: canonical X equality on same sheet → same point for k in [1,2,5,17]")


# ── §5: recover_scalar round-trips ───────────────────────────────────────────

def test_recover_scalar_no_negate():
    for k in [1, 100, N - 1]:
        assert recover_scalar(k, negated_flag=False, n=N) == k, f"Failed for k={k}"
    print("PASS: recover_scalar(k, False) == k for k in [1, 100, N-1]")

def test_recover_scalar_negate():
    for k in [1, 2, 5, 100]:
        neg_k = (N - k) % N
        recovered = recover_scalar(neg_k, negated_flag=True, n=N)
        assert recovered == k, f"recover_scalar({neg_k}, True) = {recovered}, expected {k}"
    print("PASS: recover_scalar(n-k, True) == k for k in [1,2,5,100]")

def test_recover_scalar_n_minus_1():
    k = N - 1
    neg_k = (N - k) % N   # == 1
    recovered = recover_scalar(neg_k, negated_flag=True, n=N)
    assert recovered == k, f"recover_scalar(1, True) = {recovered}, expected {N-1}"
    print("PASS: recover_scalar(1, True) == N-1")


# ── §6: recover_scalar(0, True) edge case ────────────────────────────────────

def test_recover_scalar_zero():
    # k=0 is not a valid ECDLP scalar, but recover_scalar must not crash
    assert recover_scalar(0, negated_flag=False) == 0
    # (N - 0) % N == 0
    assert recover_scalar(0, negated_flag=True)  == 0
    print("PASS: recover_scalar(0, True/False) == 0 (arithmetic edge case)")


# ── §7: negation_orbit_canon is idempotent ────────────────────────────────────

def test_orbit_canon_idempotent():
    for k in [1, 2, 5, 100, 12345]:
        q = scalar_mul_jacobian(k)
        canon, _ = negation_orbit_canon(q)
        # Second application should not change the result
        canon2, neg2 = negation_orbit_canon(canon)
        aff_c  = jacobian_to_affine(canon)
        aff_c2 = jacobian_to_affine(canon2)
        assert aff_c is not None and aff_c2 is not None
        assert aff_c.x == aff_c2.x
        assert aff_c.y == aff_c2.y, f"orbit_canon not idempotent for k={k}"
        assert neg2 == False, f"Second application should not negate again for k={k}"
    print("PASS: negation_orbit_canon is idempotent for k in [1,2,5,100,12345]")


# ── §8: negation_orbit_canon selects lower-Y half ────────────────────────────

def test_orbit_canon_lower_y():
    for k in [1, 2, 5, 100]:
        q  = scalar_mul_jacobian(k)
        aff_q = jacobian_to_affine(q)
        neg_y = fe_negate(aff_q.y)
        canon, neg_flag = negation_orbit_canon(q)
        aff_canon = jacobian_to_affine(canon)
        assert aff_canon is not None
        if aff_q.y <= neg_y:
            assert not neg_flag, f"k={k}: Q has lower Y, but negated_flag is True"
            assert aff_canon.y == aff_q.y
        else:
            assert neg_flag, f"k={k}: Q has higher Y, but negated_flag is False"
            assert aff_canon.y == neg_y
    print("PASS: negation_orbit_canon selects lower-Y representative")


# ── §9: Interval scan finds correct scalar ───────────────────────────────────

def test_black_hole_scan_finds_target():
    # Target: 3G on even sheet
    target_k = 3
    target_aff = scalar_mul_affine(target_k)
    sheet = TargetSheet(y_parity=target_aff.y & 1,
                        source=TargetSheetSource.KnownCompressedPublicKeySheet)
    target_canon, _ = canonicalize_negation_pair(
        affine_to_jacobian(target_aff), sheet, source_position=target_k
    )
    target_aff_canon = jacobian_to_affine(target_canon.point)

    hit = black_hole_scan_with_negation_map(
        interval_low=1,
        interval_high=10,
        target_canonical=target_aff_canon,
        target_sheet=sheet,
        scalar_to_jacobian=scalar_mul_jacobian,
    )
    assert hit is not None, "Scan should find target k=3 in [1,10]"
    assert hit.source_scalar == target_k, \
        f"Wrong source scalar: {hit.source_scalar}, expected {target_k}"
    assert hit.exact_target_latch == True, "Should be an exact latch"
    print(f"PASS: black_hole_scan found k={hit.source_scalar}, negated={hit.negated_flag}")

def test_black_hole_scan_finds_negated_target():
    """Verify the negated path: scan [1,10] for a target whose canonical form
    requires negation to reach.

    3G lives on one Y sheet. We declare the OPPOSITE sheet as the target sheet.
    The canonical form of 3G on the opposite sheet is -3G.
    The scanner must find s=3, negate 3G to get -3G, and report negated_flag=True.
    """
    target_k = 3
    three_g = scalar_mul_affine(target_k)
    three_g_parity = three_g.y & 1

    # Use the sheet that is OPPOSITE to 3G's actual parity.
    opposite_parity = 1 - three_g_parity
    opposite_sheet = TargetSheet(
        y_parity=opposite_parity,
        source=TargetSheetSource.KnownCompressedPublicKeySheet,
    )

    # Canonical form of 3G on opposite sheet = -3G
    cp, neg_flag_setup = canonicalize_negation_pair(
        affine_to_jacobian(three_g), opposite_sheet, target_k
    )
    assert neg_flag_setup, "Setup: 3G on opposite sheet must require negation"
    target_aff_canon = jacobian_to_affine(cp.point)
    assert target_aff_canon.y & 1 == opposite_parity, \
        "Canonical target must be on the declared opposite sheet"

    hit = black_hole_scan_with_negation_map(
        interval_low=1,
        interval_high=10,
        target_canonical=target_aff_canon,
        target_sheet=opposite_sheet,
        scalar_to_jacobian=scalar_mul_jacobian,
    )
    assert hit is not None, \
        "Scan should find k=3 in [1,10] via negation (3G negated → -3G = target)"
    assert hit.source_scalar == target_k, \
        f"Expected source_scalar=3, got {hit.source_scalar}"
    assert hit.negated_flag == True, \
        f"Expected negated_flag=True (3G must be negated to reach opposite sheet), got False"
    assert hit.exact_target_latch == True, "Must be an exact latch"
    print(f"PASS: black_hole_scan negated path: source={hit.source_scalar}, "
          f"negated={hit.negated_flag} (3G parity={three_g_parity}, sheet={opposite_parity})")

def test_black_hole_scan_empty_interval():
    # 100G is not in interval [1,10]
    target_aff = scalar_mul_affine(100)
    sheet = even_sheet()
    cp, _ = canonicalize_negation_pair(affine_to_jacobian(target_aff), sheet, 100)
    target_aff_canon = jacobian_to_affine(cp.point)

    hit = black_hole_scan_with_negation_map(
        interval_low=1,
        interval_high=10,
        target_canonical=target_aff_canon,
        target_sheet=sheet,
        scalar_to_jacobian=scalar_mul_jacobian,
    )
    assert hit is None, f"Scan should return None for 100G not in [1,10], got {hit}"
    print("PASS: black_hole_scan returns None for target outside interval")


# ── §10-13: Proof helpers ─────────────────────────────────────────────────────

def test_proof_canon_eq_neg_q():
    sheet = even_sheet()
    for k in [1, 2, 3, 7, 100]:
        q = scalar_mul_jacobian(k)
        assert proof_canonicalize_q_equals_neg_q(q, sheet), f"proof failed for k={k}"
    print("PASS: proof_canonicalize_q_equals_neg_q for k in [1,2,3,7,100]")

def test_proof_canon_x_same_point():
    sheet = even_sheet()
    for k in [1, 2, 5]:
        q   = scalar_mul_jacobian(k)
        neg_q = jacobian_negate(q)
        cp_q,    _ = canonicalize_negation_pair(q,     sheet, k)
        cp_negq, _ = canonicalize_negation_pair(neg_q, sheet, N - k)
        proof_canonical_x_implies_same_point(cp_q, cp_negq)
    print("PASS: proof_canonical_x_implies_same_point for k in [1,2,5]")

def test_proof_recover_roundtrip():
    G_aff = G
    for k in [1, 2, N - 1, 12345]:
        proof_recover_scalar_roundtrip(k, G_aff, N)
    print("PASS: proof_recover_scalar_roundtrip for k in [1, 2, N-1, 12345]")


if __name__ == "__main__":
    failures = 0
    tests = [
        ("prefix 0x02 even",                test_prefix_02_even),
        ("prefix 0x03 odd",                 test_prefix_03_odd),
        ("prefix invalid raises",           test_prefix_invalid_raises),
        ("canon pair G",                    test_canon_pair_generator),
        ("canon pair 2G",                   test_canon_pair_2g),
        ("canon pair 3G",                   test_canon_pair_3g),
        ("canon pair multi-scalar",         test_canon_pair_multiple_scalars),
        ("canonical parity even",           test_canonical_output_correct_parity_even),
        ("canonical parity odd",            test_canonical_output_correct_parity_odd),
        ("canonical X same sheet unique",   test_canonical_x_same_sheet_unique),
        ("recover no-negate",               test_recover_scalar_no_negate),
        ("recover negate",                  test_recover_scalar_negate),
        ("recover N-1",                     test_recover_scalar_n_minus_1),
        ("recover zero edge",               test_recover_scalar_zero),
        ("orbit idempotent",                test_orbit_canon_idempotent),
        ("orbit lower-Y",                   test_orbit_canon_lower_y),
        ("scan finds target",               test_black_hole_scan_finds_target),
        ("scan finds negated target",       test_black_hole_scan_finds_negated_target),
        ("scan empty interval",             test_black_hole_scan_empty_interval),
        ("proof canon == neg",              test_proof_canon_eq_neg_q),
        ("proof X implies same point",      test_proof_canon_x_same_point),
        ("proof recover roundtrip",         test_proof_recover_roundtrip),
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
        print(f"\nAll {len(tests)} canonicalization tests passed.")
    else:
        print(f"\n{failures}/{len(tests)} test(s) failed.")
        import sys; sys.exit(1)
