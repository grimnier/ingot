"""
Ingot WP1 - secp256k1 Field and Point CPU Reference Interpreter

Implements the same formulas as the plain-operator Motor in
examples/point_add_mixed_plain.mtl.  Used to verify semantic
equivalence of the expanded oracle and any optimized profile.

All arithmetic is exact Python integers mod P.  No floating point.
"""

from dataclasses import dataclass
from typing import Optional

# secp256k1 prime: p = 2^256 - 2^32 - 977
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F

# secp256k1 curve order
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

# Generator point G (affine)
GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8


# ---------- Field element operations (GF(P)) ----------

def fe_add(a: int, b: int) -> int:
    return (a + b) % P

def fe_sub(a: int, b: int) -> int:
    return (a - b) % P

def fe_mul(a: int, b: int) -> int:
    return (a * b) % P

def fe_sqr(a: int) -> int:
    return (a * a) % P

def fe_inv(a: int) -> int:
    """Modular inverse via Fermat's little theorem: a^(P-2) mod P."""
    return pow(a, P - 2, P)

def fe_negate(a: int) -> int:
    """Additive inverse in GF(P): (-a) mod P.
    fe_negate(0) == 0. fe_negate(fe_negate(x)) == x."""
    return (-a) % P


# ---------- Point representations ----------

@dataclass
class Jacobian:
    x: int
    y: int
    z: int

@dataclass
class Affine:
    x: int
    y: int


# ---------- Negation, parity, and canonicalization ----------

def affine_y_parity(a: Affine) -> int:
    """Return 0 if Y is even (compressed prefix 02), 1 if Y is odd (prefix 03)."""
    return a.y & 1

def affine_negate(a: Affine) -> Affine:
    """-Q in affine form: (x, p-y). Same X, opposite Y sheet."""
    return Affine(x=a.x, y=fe_negate(a.y))

def jacobian_negate(j: Jacobian) -> Jacobian:
    """Negate a Jacobian point: (X:Y:Z) -> (X:-Y:Z).
    Only Y changes. X and Z are preserved exactly. Cost: one field_negate."""
    return Jacobian(x=j.x, y=fe_negate(j.y), z=j.z)

def jacobian_y_parity_at_affine_boundary(j: Jacobian) -> Optional[int]:
    """Derive the affine Y parity of a Jacobian point.
    Requires a field inversion -- use only at the affine boundary, not in hot loops.
    Returns None for the point at infinity (z == 0).

    REJECT: j.y & 1 is NOT the affine Y parity in general because the
    Jacobian Y satisfies Y_affine = Y_jac / Z^3, not Y_jac.
    This function is the ONLY correct path from Jacobian to affine parity."""
    a = jacobian_to_affine(j)
    if a is None:
        return None
    return affine_y_parity(a)


# ---------- Point-add formula (mirrors plain Motor exactly) ----------

def jacobian_add_affine(walker: Jacobian, table_point: Affine) -> Jacobian:
    """
    Mixed Jacobian + Affine addition: walker + table_point -> Jacobian result.

    Variable names match the plain Motor in point_add_mixed_plain.mtl exactly.
    Each line corresponds one-to-one with a Motor statement.
    """
    z_squared = fe_mul(walker.z, walker.z)
    u2        = fe_mul(table_point.x, z_squared)
    z_cubed   = fe_mul(walker.z, z_squared)
    s2        = fe_mul(table_point.y, z_cubed)

    h         = fe_sub(u2, walker.x)
    r         = fe_sub(s2, walker.y)

    h_squared = fe_mul(h, h)
    z3        = fe_mul(walker.z, h)
    h_cubed   = fe_mul(h, h_squared)
    xh2       = fe_mul(walker.x, h_squared)
    r_squared = fe_mul(r, r)

    x3 = fe_sub(fe_sub(r_squared, h_cubed), fe_add(xh2, xh2))
    y3 = fe_sub(fe_mul(r, fe_sub(xh2, x3)), fe_mul(walker.y, h_cubed))

    return Jacobian(x=x3, y=y3, z=z3)


# ---------- Conversion ----------

def jacobian_to_affine(j: Jacobian) -> Optional[Affine]:
    """Convert Jacobian (X:Y:Z) to affine (X/Z^2, Y/Z^3). Returns None for point at infinity."""
    if j.z == 0:
        return None
    z_inv  = fe_inv(j.z)
    z_inv2 = fe_mul(z_inv, z_inv)
    z_inv3 = fe_mul(z_inv2, z_inv)
    return Affine(x=fe_mul(j.x, z_inv2), y=fe_mul(j.y, z_inv3))

def affine_to_jacobian(a: Affine) -> Jacobian:
    """Lift affine point to Jacobian with Z=1."""
    return Jacobian(x=a.x, y=a.y, z=1)


# ---------- Reference: affine point doubling (for KAT baseline) ----------

def affine_double(p: Affine) -> Affine:
    """Standard affine doubling formula for secp256k1 (a=0)."""
    lam = fe_mul(3 * fe_mul(p.x, p.x) % P, fe_inv(fe_add(p.y, p.y)))
    x3  = fe_sub(fe_mul(lam, lam), fe_add(p.x, p.x))
    y3  = fe_sub(fe_mul(lam, fe_sub(p.x, x3)), p.y)
    return Affine(x=x3, y=y3)

def affine_add(p: Affine, q: Affine) -> Affine:
    """Standard affine addition formula for secp256k1."""
    if p.x == q.x:
        return affine_double(p)
    lam = fe_mul(fe_sub(q.y, p.y), fe_inv(fe_sub(q.x, p.x)))
    x3  = fe_sub(fe_mul(lam, lam), fe_add(p.x, q.x))
    y3  = fe_sub(fe_mul(lam, fe_sub(p.x, x3)), p.y)
    return Affine(x=x3, y=y3)


# ---------- Known-Answer Tests ----------

def run_kat() -> None:
    G = Affine(x=GX, y=GY)
    two_g   = affine_double(G)
    three_g = affine_add(two_g, G)
    four_g  = affine_double(two_g)

    # KAT 1: 2G + G = 3G
    # Mixed-add requires distinct x-coordinates (h != 0).
    # walker = 2G (normalized, z=1), table = G.
    walker1  = affine_to_jacobian(two_g)
    result1j = jacobian_add_affine(walker1, G)
    result1a = jacobian_to_affine(result1j)
    assert result1a is not None, "KAT1: result is point at infinity"
    assert result1a.x == three_g.x, f"KAT1 x: {hex(result1a.x)} != {hex(three_g.x)}"
    assert result1a.y == three_g.y, "KAT1 y mismatch"
    print("KAT1 PASS: 2G + G = 3G")

    # KAT 2: 3G + G = 4G
    walker2  = affine_to_jacobian(three_g)
    result2j = jacobian_add_affine(walker2, G)
    result2a = jacobian_to_affine(result2j)
    assert result2a is not None, "KAT2: result is point at infinity"
    assert result2a.x == four_g.x, f"KAT2 x mismatch"
    assert result2a.y == four_g.y, "KAT2 y mismatch"
    print("KAT2 PASS: 3G + G = 4G")

    # KAT 3: Z1=1 specialization produces the same result as the general formula.
    # When walker.z = 1: z_squared = 1, u2 = table.x, z_cubed = 1, s2 = table.y.
    # Inline the five removed steps and verify against jacobian_add_affine.
    table = two_g
    walker_z1 = Jacobian(x=GX, y=GY, z=1)
    general_j = jacobian_add_affine(walker_z1, table)
    general_a = jacobian_to_affine(general_j)

    h   = fe_sub(table.x, GX)    # u2 = table.x (z_squared=1), so h = table.x - walker.x
    r   = fe_sub(table.y, GY)    # s2 = table.y (z_cubed=1),  so r = table.y - walker.y
    hsq = fe_mul(h, h)
    z3  = h                       # walker.z * h = 1 * h = h
    hcu = fe_mul(h, hsq)
    xh2 = fe_mul(GX, hsq)
    rsq = fe_mul(r, r)
    x3  = fe_sub(fe_sub(rsq, hcu), fe_add(xh2, xh2))
    y3  = fe_sub(fe_mul(r, fe_sub(xh2, x3)), fe_mul(GY, hcu))
    shortcut_a = jacobian_to_affine(Jacobian(x=x3, y=y3, z=z3))
    assert general_a.x == shortcut_a.x, "KAT3 x mismatch: Z1=1 shortcut vs general"
    assert general_a.y == shortcut_a.y, "KAT3 y mismatch"
    print("KAT3 PASS: Z1=1 shortcut matches general formula for G + 2G")

    print("All field_interp KATs passed.")


# ---------- Affine hop (Phase F of leviathan_blockbatch.cu) ----------

def affine_hop(walker: Affine, jump: Affine, my_inv: int) -> Affine:
    """
    Affine hop: walker + jump -> new walker.
    my_inv must equal inverse(jump.x - walker.x).
    Variable names match affine_hop.mtl exactly.
    """
    num    = fe_sub(jump.y,    walker.y)
    slope  = fe_mul(num,       my_inv)
    slope2 = fe_mul(slope,     slope)
    new_ax = fe_sub(fe_sub(slope2, walker.x), jump.x)
    diff   = fe_sub(walker.x,  new_ax)
    new_ay = fe_sub(fe_mul(slope, diff), walker.y)
    return Affine(x=new_ax, y=new_ay)


def run_affine_hop_kat() -> None:
    G = Affine(x=GX, y=GY)
    two_g   = affine_double(G)
    three_g = affine_add(two_g, G)
    five_g  = affine_add(three_g, two_g)

    # KAT 1: G + 2G = 3G
    inv1    = fe_inv(fe_sub(two_g.x, G.x))
    result1 = affine_hop(G, two_g, inv1)
    assert result1.x == three_g.x, f"KAT1 x: {hex(result1.x)} != {hex(three_g.x)}"
    assert result1.y == three_g.y, "KAT1 y mismatch"
    print("AFFINE_HOP KAT1 PASS: G + 2G = 3G")

    # KAT 2: 2G + 3G = 5G
    inv2    = fe_inv(fe_sub(three_g.x, two_g.x))
    result2 = affine_hop(two_g, three_g, inv2)
    assert result2.x == five_g.x, f"KAT2 x: {hex(result2.x)} != {hex(five_g.x)}"
    assert result2.y == five_g.y, "KAT2 y mismatch"
    print("AFFINE_HOP KAT2 PASS: 2G + 3G = 5G")

    print("All affine_hop KATs passed.")


if __name__ == "__main__":
    run_kat()
    run_affine_hop_kat()
