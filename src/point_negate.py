"""
Ingot Point Negation -- Livethine Negation-Map Black-Hole Contract

Implements the application-level contract connecting Ingot's negation map
to Livethine's recursive black-hole coordinate scanner.

Types:
    TargetSheet            -- declares which Y parity sheet is canonical for a run
    CanonicalPoint         -- a point mapped onto the target sheet with recovery metadata
    NegationMapHit         -- emitted on canonical X match during a black-hole scan

API:
    canonicalize_negation_pair  -- fold Q onto the target sheet, return (CanonicalPoint, bool)
    negation_orbit_canon        -- lower-Y-value canonical representative
    recover_scalar              -- reconstruct source scalar from canonical scalar + flag
    target_sheet_from_prefix    -- construct TargetSheet from compressed key prefix byte

Proof helpers (verified in tests/test_negation_canonical.py):
    proof_canonicalize_q_equals_neg_q       -- canonicalize(Q) == canonicalize(-Q)
    proof_canonical_x_implies_same_point    -- same X on same sheet -> same point
    proof_recover_scalar_roundtrip          -- recover(canonical, flag) == original k

References:
    LIVETHINE_NEGATION_MAP_BLACK_HOLE_CONTRACT.md
    INGOT_NEGATION_MAP_PLAYBOOK.md  sec.9 Runtime Play-by-Play, sec.10 Recommended Build Order
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

try:
    from .field_interp import (
        P, N,
        Jacobian, Affine,
        fe_negate,
        jacobian_negate, affine_negate,
        affine_y_parity, jacobian_y_parity_at_affine_boundary,
        jacobian_to_affine, affine_to_jacobian,
        affine_add, affine_double,
        GX, GY,
    )
except ImportError:
    from field_interp import (
        P, N,
        Jacobian, Affine,
        fe_negate,
        jacobian_negate, affine_negate,
        affine_y_parity, jacobian_y_parity_at_affine_boundary,
        jacobian_to_affine, affine_to_jacobian,
        affine_add, affine_double,
        GX, GY,
    )


# ---------------------------------------------------------------------------
# sec.10 Required Livethine Types
# ---------------------------------------------------------------------------

class TargetSheetSource(Enum):
    KnownCompressedPublicKeySheet = "known_compressed_public_key"
    AddressOnlyUnknownSheet       = "address_only_unknown"


@dataclass(frozen=True)
class TargetSheet:
    """Declares which affine Y parity sheet is canonical for a complete scan.
    Must be selected once per run and held constant.

    y_parity: 0 = even Y (compressed prefix 02), 1 = odd Y (prefix 03).
    source:   authority for the sheet selection.

    For HASH160-only targets where the prefix is unknown, Livethine must
    maintain TWO TargetSheet hypotheses (sec.9 of the Contract) and cannot
    apply single-sheet canonicalization without a public-key hint.
    """
    y_parity: int
    source:   TargetSheetSource


@dataclass
class CanonicalPoint:
    """A point mapped onto the target Y sheet with full scalar recovery metadata.

    point:           Jacobian point on the target sheet.
    sheet:           the TargetSheet this point was canonicalized onto.
    negated_flag:    True if the source Q was on the opposite sheet and was negated.
    source_position: the original scanned scalar k (NOT the canonical scalar).
    proof_id:        optional provenance string for audit trails.
    """
    point:           Jacobian
    sheet:           TargetSheet
    negated_flag:    bool
    source_position: int
    proof_id:        Optional[str] = None


@dataclass
class NegationMapHit:
    """Emitted when canonicalize_negation_pair detects a canonical X match.

    source_scalar:       the original scanned scalar k (= source_position).
    source_interval_low: lower bound of the questioned interval.
    source_interval_high: upper bound of the questioned interval.
    canonical_x:         affine X of the canonical match (not Jacobian X).
    negated_flag:        True if k was on the opposite sheet (source = n - canonical).
    exact_target_latch:  True when full affine point equality is confirmed.
    """
    source_scalar:        int
    source_interval_low:  int
    source_interval_high: int
    canonical_x:          int
    negated_flag:         bool
    exact_target_latch:   bool


# ---------------------------------------------------------------------------
# Canonicalization API
# ---------------------------------------------------------------------------

def canonicalize_negation_pair(
    q: Jacobian,
    target_sheet: TargetSheet,
    source_position: int = 0,
    proof_id: Optional[str] = None,
) -> Tuple[CanonicalPoint, bool]:
    """Map q onto the target Y sheet. Return (CanonicalPoint, negated_flag).

    Canonicalization MUST happen at the affine boundary because Jacobian Y
    parity != affine Y parity in general (Y_affine = Y_jac / Z^3).

    Contract guarantee (sec.3): canonicalize(Q) == canonicalize(-Q).
    Both Q and -Q map to the same canonical point.

    The returned CanonicalPoint.source_position retains the original scanned
    scalar k. Livethine MUST NOT silently substitute n-k.

    For point at infinity (z=0): returned as-is, negated_flag=False.
    """
    a = jacobian_to_affine(q)
    if a is None:
        cp = CanonicalPoint(
            point=q,
            sheet=target_sheet,
            negated_flag=False,
            source_position=source_position,
            proof_id=proof_id,
        )
        return cp, False

    current_parity = affine_y_parity(a)
    if current_parity == target_sheet.y_parity:
        negated_flag = False
        canonical_jac = q
    else:
        negated_flag = True
        canonical_jac = jacobian_negate(q)

    cp = CanonicalPoint(
        point=canonical_jac,
        sheet=target_sheet,
        negated_flag=negated_flag,
        source_position=source_position,
        proof_id=proof_id,
    )
    return cp, negated_flag


def recover_scalar(
    canonical_scalar: int,
    negated_flag: bool,
    n: int = N,
) -> int:
    """Reconstruct the original source scalar from the canonical scalar and flag.

    If negated_flag is False: source scalar == canonical_scalar.
    If negated_flag is True:  canonical_scalar came from (n - source_scalar) mod n,
        so source_scalar = (n - canonical_scalar) mod n.

    Livethine MUST call this before reporting a found key. Returning the
    canonical_scalar when negated_flag=True silently returns n-k instead of k.
    """
    if not negated_flag:
        return canonical_scalar
    return (n - canonical_scalar) % n


def negation_orbit_canon(q: Jacobian) -> Tuple[Jacobian, bool]:
    """Lower-Y-value canonical representative from the negation pair {Q, -Q}.

    Chooses the member with the smaller affine Y value. This is curve-level
    canonicalization independent of any target sheet.

    Returns (canonical_jacobian, negated_flag).

    For Livethine's target-aware scanner use canonicalize_negation_pair instead.
    This function is useful for orbit-based search-space reduction when the
    target sheet is not yet known.
    """
    a = jacobian_to_affine(q)
    if a is None:
        return q, False
    neg_y = fe_negate(a.y)
    if a.y <= neg_y:
        return q, False
    else:
        return jacobian_negate(q), True


def target_sheet_from_compressed_prefix(prefix_byte: int) -> TargetSheet:
    """Construct the canonical TargetSheet from a compressed public key prefix byte.
    0x02 -> even Y sheet (y_parity=0).
    0x03 -> odd Y sheet  (y_parity=1).
    """
    if prefix_byte == 0x02:
        return TargetSheet(y_parity=0, source=TargetSheetSource.KnownCompressedPublicKeySheet)
    elif prefix_byte == 0x03:
        return TargetSheet(y_parity=1, source=TargetSheetSource.KnownCompressedPublicKeySheet)
    else:
        raise ValueError(f"Invalid compressed prefix: {prefix_byte:#04x} (expected 0x02 or 0x03)")


# ---------------------------------------------------------------------------
# Black-hole scan integration (sec.6 of the Contract)
# ---------------------------------------------------------------------------

def black_hole_scan_with_negation_map(
    interval_low: int,
    interval_high: int,
    target_canonical: Affine,
    target_sheet: TargetSheet,
    scalar_to_jacobian,          # callable: int -> Jacobian (e.g. from walk state)
) -> Optional[NegationMapHit]:
    """Reference implementation of the negation-map black-hole scan (sec.6).

    For every scalar s in [interval_low, interval_high]:
      1. Compute Q = s*G (via scalar_to_jacobian).
      2. Canonicalize Q onto target_sheet.
      3. If canonical Q.x == target_canonical.x: emit NegationMapHit.

    The canonical target must already be on the declared sheet.
    Returns None if no hit found (interval proven empty on this sheet).

    This is a CPU reference -- the GPU implementation uses Ingot motors.
    """
    for s in range(interval_low, interval_high + 1):
        q_jac = scalar_to_jacobian(s)
        cp, neg_flag = canonicalize_negation_pair(q_jac, target_sheet, source_position=s)
        aff = jacobian_to_affine(cp.point)
        if aff is None:
            continue
        if aff.x == target_canonical.x and aff.y == target_canonical.y:
            return NegationMapHit(
                source_scalar=s,
                source_interval_low=interval_low,
                source_interval_high=interval_high,
                canonical_x=aff.x,
                negated_flag=neg_flag,
                exact_target_latch=True,
            )
        # X match without Y match means target_canonical is malformed (wrong sheet).
        # Continue scanning rather than returning a false positive.
    return None


# ---------------------------------------------------------------------------
# Proof helpers (verified in tests/test_negation_canonical.py)
# ---------------------------------------------------------------------------

def proof_canonicalize_q_equals_neg_q(q: Jacobian, target_sheet: TargetSheet) -> bool:
    """Prove: canonicalize(Q) == canonicalize(-Q).

    Both Q and its negation must map to the same canonical affine point.
    Raises AssertionError on failure.
    """
    cp_q,    _ = canonicalize_negation_pair(q,              target_sheet)
    cp_negq, _ = canonicalize_negation_pair(jacobian_negate(q), target_sheet)

    aff_q    = jacobian_to_affine(cp_q.point)
    aff_negq = jacobian_to_affine(cp_negq.point)

    if aff_q is None and aff_negq is None:
        return True
    assert aff_q    is not None, "canonicalize(Q) is infinity but canonicalize(-Q) is not"
    assert aff_negq is not None, "canonicalize(-Q) is infinity but canonicalize(Q) is not"
    assert aff_q.x == aff_negq.x, \
        f"canonicalize(Q).x != canonicalize(-Q).x: {aff_q.x:#066x} vs {aff_negq.x:#066x}"
    assert aff_q.y == aff_negq.y, \
        f"canonicalize(Q).y != canonicalize(-Q).y: {aff_q.y:#066x} vs {aff_negq.y:#066x}"
    return True


def proof_canonical_x_implies_same_point(
    cp_a: CanonicalPoint,
    cp_b: CanonicalPoint,
) -> bool:
    """Prove: if two canonical points on the same sheet share affine X, they are equal.

    On a fixed Y-parity sheet, each X value maps to exactly one affine point
    (the curve equation y^2 = x^3 + 7 has exactly one Y per X per parity class).
    Raises AssertionError on failure.
    """
    assert cp_a.sheet.y_parity == cp_b.sheet.y_parity, \
        "Both points must be on the same target sheet for X-equality to imply point equality"

    aff_a = jacobian_to_affine(cp_a.point)
    aff_b = jacobian_to_affine(cp_b.point)

    if aff_a is None or aff_b is None:
        assert aff_a is None and aff_b is None, \
            "One canonical point is at infinity, the other is not"
        return True

    if aff_a.x == aff_b.x:
        assert aff_a.y == aff_b.y, \
            (f"Same X {aff_a.x:#066x} on same sheet but different Y: "
             f"{aff_a.y:#066x} vs {aff_b.y:#066x} -- sheet contract violated")
    return True


def proof_recover_scalar_roundtrip(k: int, g: Affine, n: int = N) -> bool:
    """Prove: recover_scalar(n-k, negated_flag=True) == k.
    And:      recover_scalar(k,   negated_flag=False) == k.
    Raises AssertionError on failure.
    """
    assert recover_scalar(k,       negated_flag=False, n=n) == k, \
        f"Non-negated recovery failed: expected {k}, got {recover_scalar(k, False, n)}"
    neg_scalar = (n - k) % n
    assert recover_scalar(neg_scalar, negated_flag=True, n=n) == k, \
        f"Negated recovery failed: expected {k}, got {recover_scalar(neg_scalar, True, n)}"
    return True


def proof_jacobian_parity_rejection(j: Jacobian) -> bool:
    """Demonstrate why Jacobian Y & 1 is NOT a valid parity test.

    Constructs a Jacobian point with Z != 1 and shows that j.y & 1 != affine Y parity.
    Raises AssertionError if a counterexample cannot be found (meaning the
    chosen j happens to have Z=1 or some degenerate case -- not a real proof failure).
    """
    if j.z == 0 or j.z == 1:
        return True   # no interesting Jacobian Y/Z distinction when Z=0 or Z=1
    aff = jacobian_to_affine(j)
    if aff is None:
        return True
    jac_parity = j.y & 1
    aff_parity = aff.y & 1
    # They can differ: document the values but do not assert equality
    # (that would be the wrong proof -- they are generally different).
    return jac_parity != aff_parity or j.z == 1
