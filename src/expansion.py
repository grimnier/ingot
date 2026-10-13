"""
Ingot WP2 -- Deterministic Profile Expansion

Maps plain-operator field expressions to versioned intrinsic templates.

Profile: secp256k1_mixed_add_compat_v1

Each field * field in the plain Motor expands to the schoolbook 4x4
multiply + unconditional pseudo-Mersenne reduction sequence used in
examples/point_add_mixed.mtl.  Field subtraction expands to the
4-limb sub-with-borrow + conditional-add-P pattern from the same oracle.
The doubling sub-step for x3 uses the branchless add-carry + conditional-
subtract-P pattern.

Output: a complete MTL source string (expanded motor + compatible
transmission) that can be parsed and compiled by the existing pipeline
without any changes to the IR builder or PTX emitter.
"""

from typing import List, Optional, Dict, Tuple
try:
    from .ast_nodes import (
        MotorDecl, TransmissionDecl, MotorParam,
        Expr, Var, BinOp, MemberAccess, Call, Block, VarDecl, Assign, IntLit,
    )
except ImportError:
    from ast_nodes import (
        MotorDecl, TransmissionDecl, MotorParam,
        Expr, Var, BinOp, MemberAccess, Call, Block, VarDecl, Assign, IntLit,
    )

PROFILE_COMPAT_V1 = "secp256k1_mixed_add_compat_v1"

# Register layout for jacobian<secp256k1> (12 regs: R0..R11)
_JAC_FIELD_OFFSETS = {"x": (0, 1, 2, 3), "y": (4, 5, 6, 7), "z": (8, 9, 10, 11)}
# Register layout for affine<secp256k1> (8 regs: base..base+7)
_AFF_FIELD_OFFSETS = {"x": (0, 1, 2, 3), "y": (4, 5, 6, 7)}


def try_expand(motor: MotorDecl, tx: TransmissionDecl) -> Optional[str]:
    """
    If motor is a plain-operator secp256k1 motor with the compat profile,
    return a complete expanded MTL source string.  Returns None otherwise.
    """
    if not motor.params:
        return None
    if getattr(tx, "profile", None) != PROFILE_COMPAT_V1:
        return None
    return _Secp256k1CompatV1Expander(motor, tx).expand()


# ---------------------------------------------------------------------------
# Internal expander
# ---------------------------------------------------------------------------

class _Secp256k1CompatV1Expander:
    """Expands point_add_mixed_plain motor through profile secp256k1_mixed_add_compat_v1."""

    def __init__(self, motor: MotorDecl, tx: TransmissionDecl):
        self.motor = motor
        self.tx = tx
        # param name -> {'x': [l0,l1,l2,l3], 'y': [...], 'z': [...]}
        self.param_limbs: Dict[str, Dict[str, List[str]]] = {}
        # field var name -> [l0, l1, l2, l3]
        self.field_var_limbs: Dict[str, List[str]] = {}
        self._tmp_counter = 0

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def expand(self) -> str:
        self._build_param_limbs()
        self._collect_field_var_names()

        out = []
        out += self._gen_motor()
        out.append("")
        out += self._gen_transmission()
        return "\n".join(out)

    # ------------------------------------------------------------------
    # Param -> limb mapping
    # ------------------------------------------------------------------

    def _build_param_limbs(self) -> None:
        """Map each MotorParam to its physical register-named u64 limbs."""
        for p in self.motor.params:
            # Resolve the register range from the matching transmission contract param
            reg_range = self._contract_reg_range(p.name)
            base = int(reg_range[0][1:])  # 'R0' -> 0

            if p.type.startswith("jacobian"):
                limbs: Dict[str, List[str]] = {}
                for field, offs in _JAC_FIELD_OFFSETS.items():
                    limbs[field] = [f"p_{p.name}_{field}_{i}" for i in range(4)]
                    # Store the @register assignment for each limb
                    for i, off in enumerate(offs):
                        limbs[field][i] = _reg_limb_name(p.name, field, i)
                self.param_limbs[p.name] = limbs
                # Also expose physical reg numbers for locked declarations
                self.param_limbs[p.name]["_base"] = base
                self.param_limbs[p.name]["_regs"] = [
                    base + off
                    for f in ("x", "y", "z")
                    for off in _JAC_FIELD_OFFSETS[f]
                ]

            elif p.type.startswith("affine"):
                limbs = {}
                for field, offs in _AFF_FIELD_OFFSETS.items():
                    limbs[field] = [_reg_limb_name(p.name, field, i) for i in range(4)]
                self.param_limbs[p.name] = limbs
                self.param_limbs[p.name]["_base"] = base
                self.param_limbs[p.name]["_regs"] = [
                    base + off
                    for f in ("x", "y")
                    for off in _AFF_FIELD_OFFSETS[f]
                ]

    def _contract_reg_range(self, param_name: str) -> Tuple[str, str]:
        """Return (start_reg, end_reg) for a param from the transmission contract."""
        mc = self.tx.motor_contract
        if mc is None:
            raise ValueError("Transmission has no motor contract")
        for p in mc.params:
            if p.name == param_name and p.register_range is not None:
                return p.register_range.start, p.register_range.end
        raise ValueError(f"No register range found for param '{param_name}'")

    # ------------------------------------------------------------------
    # Field variable collection
    # ------------------------------------------------------------------

    def _collect_field_var_names(self) -> None:
        """
        Pre-allocate limb names for every field variable that will appear,
        including temporaries created during compound expression decomposition.
        """
        # Explicit vars from body
        for name in _body_field_var_names(self.motor.body):
            self.field_var_limbs[name] = _field_limbs(name)

        # Temps needed for x3 compound: xh2_dbl, and in-place x3 reuse is fine
        self.field_var_limbs["_xh2_dbl"] = _field_limbs("xh2_dbl")

        # Temps needed for y3 compound
        self.field_var_limbs["_dx"]  = _field_limbs("dx")
        self.field_var_limbs["_t0m"] = _field_limbs("t0m")   # r * dx
        self.field_var_limbs["_t1m"] = _field_limbs("t1m")   # Y1 * h_cubed

    # ------------------------------------------------------------------
    # Motor generation
    # ------------------------------------------------------------------

    def _gen_motor(self) -> List[str]:
        lines = [f"motor {self.motor.name} {{"]
        lines += self._gen_locked_reg_decls()
        lines += self._gen_shared_temp_decls()
        lines += self._gen_field_var_decls()
        lines += self._gen_const_inits()
        lines += self._gen_body_expansion()
        lines += self._gen_write_back()
        lines.append("}")
        return lines

    def _gen_locked_reg_decls(self) -> List[str]:
        out = []
        out.append("    // -- Locked register declarations -------------------------------------")
        for p in self.motor.params:
            info = self.param_limbs[p.name]
            base = info["_base"]
            if p.type.startswith("jacobian"):
                fields = ["x", "y", "z"]
            else:
                fields = ["x", "y"]
            for field in fields:
                for i, limb in enumerate(info[field]):
                    reg_n = base + _JAC_FIELD_OFFSETS.get(field, _AFF_FIELD_OFFSETS.get(field, (0,)))[i] if p.type.startswith("jacobian") else base + _AFF_FIELD_OFFSETS[field][i]
                    out.append(f"    @register(R{reg_n}) u64 {limb};")
        return out

    def _gen_shared_temp_decls(self) -> List[str]:
        return [
            "",
            "    // -- Shared multiply temps (reused across all field muls) -------------",
            "    u64 mt0; u64 mt1; u64 mt2; u64 mt3;",
            "    u64 mt4; u64 mt5; u64 mt6; u64 mt7;",
            "    u64 mt4l; u64 mt4h; u64 mt5l; u64 mt5h;",
            "    u64 mt6l; u64 mt6h; u64 mt7l; u64 mt7h;",
            "    u64 c977; u64 c32; u64 cC; u64 PL0;",
            "    u64 t4C_lo; u64 t4C_hi;",
            "    u64 sb; u64 ap0;",
            "    u64 dbl_cy; u64 dbl_ds0; u64 dbl_ds1; u64 dbl_ds2; u64 dbl_ds3;",
            "    u64 dbl_db; u64 dbl_dnb;",
        ]

    def _gen_field_var_decls(self) -> List[str]:
        out = ["", "    // -- Field intermediate variables ------------------------------------"]
        for name, limbs in self.field_var_limbs.items():
            display = name.lstrip("_")
            out.append(f"    // {display}")
            decls = "    " + "  ".join(f"u64 {l};" for l in limbs)
            out.append(decls)
        return out

    def _gen_const_inits(self) -> List[str]:
        return [
            "",
            "    c977 = 977;",
            "    c32  = 32;",
            "    cC   = 0x1000003D1;",
            "    PL0  = 0xFFFFFFFEFFFFFC2F;",
        ]

    # ------------------------------------------------------------------
    # Body expansion: walk plain Motor stmts, dispatch to templates
    # ------------------------------------------------------------------

    def _gen_body_expansion(self) -> List[str]:
        out = []
        for stmt in self.motor.body.stmts:
            # Plain Motor body stmts: Block([VarDecl(field), Assign(name, expr)])
            if isinstance(stmt, Block) and len(stmt.stmts) == 2:
                decl, asn = stmt.stmts
                if isinstance(decl, VarDecl) and decl.type == "field" and isinstance(asn, Assign):
                    var_name = decl.name
                    expr = asn.value
                    if var_name == "x3":
                        out += self._expand_x3(expr)
                    elif var_name == "y3":
                        out += self._expand_y3(expr)
                    else:
                        out += self._expand_simple_field_op(var_name, expr)
                    continue
            # write-back: Assign(Var('walker'), Call('jacobian', [x3, y3, z3]))
            if isinstance(stmt, Assign):
                if (isinstance(stmt.lvalue, Var) and stmt.lvalue.name in self.param_limbs
                        and isinstance(stmt.value, Call)):
                    # handled in _gen_write_back
                    continue
        return out

    def _expand_simple_field_op(self, var_name: str, expr: Expr) -> List[str]:
        out_limbs = self.field_var_limbs[var_name]
        if isinstance(expr, BinOp) and expr.op == "*":
            a = self._resolve_limbs(expr.left)
            b = self._resolve_limbs(expr.right)
            lines = [f"", f"    // -- {var_name} = field_mul ------------------------------------------"]
            lines += _field_mul_tmpl(a, b, out_limbs)
            return lines
        if isinstance(expr, BinOp) and expr.op == "-":
            # Detect field negate: (0 - expr) is cheaper to lower as negate than sub.
            if isinstance(expr.left, IntLit) and expr.left.value == 0:
                b = self._resolve_limbs(expr.right)
                lines = [f"", f"    // -- {var_name} = field_negate -------------------------------------"]
                lines += _field_negate_tmpl(b, out_limbs)
                return lines
            a = self._resolve_limbs(expr.left)
            b = self._resolve_limbs(expr.right)
            lines = [f"", f"    // -- {var_name} = field_sub ------------------------------------------"]
            lines += _field_sub_tmpl(a, b, out_limbs)
            return lines
        raise NotImplementedError(f"Cannot expand field op for {var_name}: {type(expr).__name__} op={getattr(expr,'op',None)}")

    def _expand_x3(self, expr: Expr) -> List[str]:
        """
        x3 = r_squared - h_cubed - (xh2 + xh2)
        Step a: x3 = r_squared - h_cubed
        Step b: xh2_dbl = xh2 + xh2  (field double)
        Step c: x3 = x3 - xh2_dbl
        """
        out = ["", "    // -- x3 = r_squared - h_cubed - 2*xh2 ---------------------------------"]
        x3  = self.field_var_limbs["x3"]
        dbl = self.field_var_limbs["_xh2_dbl"]

        # expr is  (r_squared - h_cubed) - (xh2 + xh2)
        # left  = r_squared - h_cubed
        # right = xh2 + xh2
        lhs = expr.left   # BinOp(r_squared, '-', h_cubed)
        rhs = expr.right  # BinOp(xh2, '+', xh2)
        # Step a: x3 = r_squared - h_cubed
        a_limbs = self._resolve_limbs(lhs.left)
        b_limbs = self._resolve_limbs(lhs.right)
        out += [f"    // step a: x3 = r_squared - h_cubed"]
        out += _field_sub_tmpl(a_limbs, b_limbs, x3)

        # Step b: xh2_dbl = xh2 + xh2
        xh2_limbs = self._resolve_limbs(rhs.left)
        out += [f"    // step b: xh2_dbl = 2*xh2"]
        out += _field_double_tmpl(xh2_limbs, dbl)

        # Step c: x3 = x3 - xh2_dbl  (in-place)
        out += [f"    // step c: x3 -= xh2_dbl"]
        out += _field_sub_tmpl(x3, dbl, x3)
        return out

    def _expand_y3(self, expr: Expr) -> List[str]:
        """
        y3 = r * (xh2 - x3) - walker.y * h_cubed
        Step a: dx  = xh2 - x3
        Step b: t0m = r * dx
        Step c: t1m = walker.y * h_cubed
        Step d: y3  = t0m - t1m
        """
        out = ["", "    // -- y3 = r*(xh2-x3) - walker.y*h_cubed -------------------------------"]
        dx  = self.field_var_limbs["_dx"]
        t0m = self.field_var_limbs["_t0m"]
        t1m = self.field_var_limbs["_t1m"]
        y3  = self.field_var_limbs["y3"]

        # expr: BinOp(BinOp(r, '*', BinOp(xh2,'-',x3)),  '-',  BinOp(walker.y,'*',h_cubed))
        mul_left  = expr.left   # r * (xh2 - x3)
        mul_right = expr.right  # walker.y * h_cubed

        # Step a: dx = xh2 - x3
        xh2_limbs = self._resolve_limbs(mul_left.right.left)
        x3_limbs  = self._resolve_limbs(mul_left.right.right)
        out += ["    // step a: dx = xh2 - x3"]
        out += _field_sub_tmpl(xh2_limbs, x3_limbs, dx)

        # Step b: t0m = r * dx
        r_limbs = self._resolve_limbs(mul_left.left)
        out += ["    // step b: t0m = r * dx"]
        out += _field_mul_tmpl(r_limbs, dx, t0m)

        # Step c: t1m = walker.y * h_cubed
        wy_limbs = self._resolve_limbs(mul_right.left)
        hc_limbs = self._resolve_limbs(mul_right.right)
        out += ["    // step c: t1m = walker.y * h_cubed"]
        out += _field_mul_tmpl(wy_limbs, hc_limbs, t1m)

        # Step d: y3 = t0m - t1m
        out += ["    // step d: y3 = t0m - t1m"]
        out += _field_sub_tmpl(t0m, t1m, y3)
        return out

    def _gen_write_back(self) -> List[str]:
        """Move x3, y3, z3 limbs back into the locked walker registers."""
        walker = self.param_limbs["walker"]
        x3 = self.field_var_limbs["x3"]
        y3 = self.field_var_limbs["y3"]
        z3 = self.field_var_limbs["z3"]
        out = ["", "    // -- Write-back: walker <- jacobian(x3, y3, z3) -------------------------"]
        for i, (dst, src) in enumerate(zip(walker["x"], x3)):
            out.append(f"    {dst} = {src};")
        for i, (dst, src) in enumerate(zip(walker["y"], y3)):
            out.append(f"    {dst} = {src};")
        for i, (dst, src) in enumerate(zip(walker["z"], z3)):
            out.append(f"    {dst} = {src};")
        return out

    # ------------------------------------------------------------------
    # Expression resolver
    # ------------------------------------------------------------------

    def _resolve_limbs(self, expr: Expr) -> List[str]:
        """Return the 4 u64 limb names for a field-valued expression.
        IntLit(0) represents the zero field element: all limbs are the literal "0"."""
        if isinstance(expr, IntLit) and expr.value == 0:
            return ["0", "0", "0", "0"]
        if isinstance(expr, MemberAccess):
            obj_name = expr.obj.name if isinstance(expr.obj, Var) else str(expr.obj)
            return self.param_limbs[obj_name][expr.member]
        if isinstance(expr, Var):
            name = expr.name
            if name in self.field_var_limbs:
                return self.field_var_limbs[name]
            raise ValueError(f"Unknown field variable: '{name}'")
        raise ValueError(f"Cannot resolve limbs from {type(expr).__name__}")

    # ------------------------------------------------------------------
    # Transmission generation
    # ------------------------------------------------------------------

    def _gen_transmission(self) -> List[str]:
        """Generate a compatible launch transmission for the expanded motor."""
        lines = [f"transmission {self.tx.name} {{"]
        if getattr(self.tx, "target", None):
            lines.append(f"    target {self.tx.target};")
        if getattr(self.tx, "profile", None):
            lines.append(f"    profile {self.tx.profile};")
        for v in getattr(self.tx, "verify_clauses", []):
            arg = f"({v.arg})" if v.arg else ""
            lines.append(f"    verify {v.kind}{arg};")
        for r in getattr(self.tx, "require_clauses", []):
            arg = f"({r.arg})" if r.arg else ""
            lines.append(f"    require {r.kind}{arg};")
        # Add a minimal grid/block so the PTX emitter has launch geometry
        lines.append("    grid (1, 1, 1);")
        lines.append("    block (256, 1, 1);")
        lines.append(f"    motor {self.motor.name} {{")
        lines += self._gen_contract_params()
        lines.append("    }")
        lines.append("}")
        return lines

    def _gen_contract_params(self) -> List[str]:
        out = []
        for p in self.motor.params:
            info = self.param_limbs[p.name]
            if p.type.startswith("jacobian"):
                fields = [("x", "inout"), ("y", "inout"), ("z", "inout")]
            else:
                dir_kw = "inout" if p.direction == "inout" else "in"
                fields = [("x", dir_kw), ("y", dir_kw)]
            for field, dir_kw in fields:
                for limb in info[field]:
                    out.append(f"        {dir_kw} u64 {limb};")
        return out


# ---------------------------------------------------------------------------
# Helper: limb naming
# ---------------------------------------------------------------------------

def _reg_limb_name(param: str, field: str, idx: int) -> str:
    """Canonical limb name for a register-mapped param field."""
    # walker.x -> X1_0..X1_3, walker.y -> Y1_0..Y1_3, walker.z -> Z1_0..Z1_3
    # table_point.x -> x2_0..x2_3, table_point.y -> y2_0..y2_3
    MAP = {
        ("walker", "x"): "X1", ("walker", "y"): "Y1", ("walker", "z"): "Z1",
        ("table_point", "x"): "x2", ("table_point", "y"): "y2",
    }
    prefix = MAP.get((param, field), f"{param}_{field}")
    return f"{prefix}_{idx}"

def _field_limbs(name: str) -> List[str]:
    return [f"{name}_{i}" for i in range(4)]

def _body_field_var_names(body) -> List[str]:
    """Collect field VarDecl names from the motor body (Block stmts)."""
    names = []
    for stmt in body.stmts:
        if isinstance(stmt, Block) and len(stmt.stmts) >= 1:
            if isinstance(stmt.stmts[0], VarDecl) and stmt.stmts[0].type == "field":
                names.append(stmt.stmts[0].name)
    return names

def _is_sub(expr: Expr) -> bool:
    return isinstance(expr, BinOp) and expr.op == "-"

def _left(expr: Expr) -> Expr:
    return expr.left

def _right(expr: Expr) -> Expr:
    return expr.right


# ---------------------------------------------------------------------------
# Template functions -- emit MTL source lines
# ---------------------------------------------------------------------------

def _field_mul_tmpl(a: List[str], b: List[str], out: List[str]) -> List[str]:
    """
    Schoolbook 4x4 multiply a[0..3] * b[0..3] + unconditional Tier-3
    pseudo-Mersenne reduction.  Result in out[0..3].
    Shared temps mt0..mt7, mt4l/mt4h..mt7l/mt7h must be pre-declared.
    """
    a0,a1,a2,a3 = a
    b0,b1,b2,b3 = b
    r0,r1,r2,r3 = out
    L = lambda s: f"    {s}"
    ls = [
        # Row 0
        L(f"mt0 = mul_lo64({a0}, {b0});"),
        L(f"mt1 = mulhi64({a0}, {b0});"),
        L(f"mt1 = mad_lo_cc64({a0}, {b1}, mt1);"),
        L(f"mt2 = madc_hi64({a0}, {b1}, 0);"),
        L(f"mt2 = mad_lo_cc64({a0}, {b2}, mt2);"),
        L(f"mt3 = madc_hi64({a0}, {b2}, 0);"),
        L(f"mt3 = mad_lo_cc64({a0}, {b3}, mt3);"),
        L(f"mt4 = madc_hi64({a0}, {b3}, 0);"),
        # Row 1
        L(f"mt1 = mad_lo_cc64({a1}, {b0}, mt1);"),
        L(f"mt2 = madc_lo_cc64({a1}, {b1}, mt2);"),
        L(f"mt3 = madc_lo_cc64({a1}, {b2}, mt3);"),
        L(f"mt4 = madc_lo_cc64({a1}, {b3}, mt4);"),
        L(f"mt5 = addc64(0, 0);"),
        L(f"mt2 = mad_hi_cc64({a1}, {b0}, mt2);"),
        L(f"mt3 = madc_hi_cc64({a1}, {b1}, mt3);"),
        L(f"mt4 = madc_hi_cc64({a1}, {b2}, mt4);"),
        L(f"mt5 = madc_hi64({a1}, {b3}, mt5);"),
        # Row 2
        L(f"mt2 = mad_lo_cc64({a2}, {b0}, mt2);"),
        L(f"mt3 = madc_lo_cc64({a2}, {b1}, mt3);"),
        L(f"mt4 = madc_lo_cc64({a2}, {b2}, mt4);"),
        L(f"mt5 = madc_lo_cc64({a2}, {b3}, mt5);"),
        L(f"mt6 = addc64(0, 0);"),
        L(f"mt3 = mad_hi_cc64({a2}, {b0}, mt3);"),
        L(f"mt4 = madc_hi_cc64({a2}, {b1}, mt4);"),
        L(f"mt5 = madc_hi_cc64({a2}, {b2}, mt5);"),
        L(f"mt6 = madc_hi64({a2}, {b3}, mt6);"),
        # Row 3
        L(f"mt3 = mad_lo_cc64({a3}, {b0}, mt3);"),
        L(f"mt4 = madc_lo_cc64({a3}, {b1}, mt4);"),
        L(f"mt5 = madc_lo_cc64({a3}, {b2}, mt5);"),
        L(f"mt6 = madc_lo_cc64({a3}, {b3}, mt6);"),
        L(f"mt7 = addc64(0, 0);"),
        L(f"mt4 = mad_hi_cc64({a3}, {b0}, mt4);"),
        L(f"mt5 = madc_hi_cc64({a3}, {b1}, mt5);"),
        L(f"mt6 = madc_hi_cc64({a3}, {b2}, mt6);"),
        L(f"mt7 = madc_hi64({a3}, {b3}, mt7);"),
        # Reduction Step A: fold t4..t7 * 977
        L(f"mt4h = mt4;"),
        L(f"mt0 = mad_lo_cc64(mt4, c977, mt0);"),
        L(f"mt1 = madc_lo_cc64(mt5, c977, mt1);"),
        L(f"mt2 = madc_lo_cc64(mt6, c977, mt2);"),
        L(f"mt3 = madc_lo_cc64(mt7, c977, mt3);"),
        L(f"mt4l = addc64(0, 0);"),
        L(f"mt1 = mad_hi_cc64(mt4, c977, mt1);"),
        L(f"mt2 = madc_hi_cc64(mt5, c977, mt2);"),
        L(f"mt3 = madc_hi_cc64(mt6, c977, mt3);"),
        L(f"mt4 = madc_hi64(mt7, c977, mt4l);"),
        # Reduction Step B: fold t4..t7 << 32
        L(f"mt4l = mt4h << c32; mt4h = mt4h >> c32;"),
        L(f"mt5l = mt5 << c32; mt5h = mt5 >> c32;"),
        L(f"mt6l = mt6 << c32; mt6h = mt6 >> c32;"),
        L(f"mt7l = mt7 << c32; mt7h = mt7 >> c32;"),
        L(f"mt0 = add_cc64(mt0, mt4l);"),
        L(f"mt1 = addc_cc64(mt1, mt4h);"),
        L(f"mt1 = addc_cc64(mt1, mt5l);"),
        L(f"mt2 = addc_cc64(mt2, mt5h);"),
        L(f"mt2 = addc_cc64(mt2, mt6l);"),
        L(f"mt3 = addc_cc64(mt3, mt6h);"),
        L(f"mt3 = addc_cc64(mt3, mt7l);"),
        L(f"mt4 = addc64(mt7h, mt4);"),
        # Reduction Step C: fold overflow limb by C via exact 64x64->128 multiply.
        # No truncating shift — t4*C captured fully in lo/hi halves.
        L(f"t4C_lo = mul_lo64(mt4, cC);"),
        L(f"t4C_hi = mulhi64(mt4, cC);"),
        L(f"mt0 = add_cc64(mt0, t4C_lo);"),
        L(f"mt1 = addc_cc64(mt1, t4C_hi);"),
        L(f"mt2 = addc_cc64(mt2, 0);"),
        L(f"mt3 = addc64(mt3, 0);"),
        # Save result
        L(f"{r0} = mt0; {r1} = mt1; {r2} = mt2; {r3} = mt3;"),
    ]
    return ls


def _field_sub_tmpl(a: List[str], b: List[str], out: List[str]) -> List[str]:
    """
    Modular subtraction a - b mod P.
    Uses shared vars sb, ap0, PL0 (pre-declared).
    Writes result to out[0..3].  Safe for in-place (out == a).
    """
    a0,a1,a2,a3 = a
    b0,b1,b2,b3 = b
    r0,r1,r2,r3 = out
    L = lambda s: f"    {s}"
    return [
        L(f"{r0} = sub_cc64({a0}, {b0});"),
        L(f"{r1} = subc_cc64({a1}, {b1});"),
        L(f"{r2} = subc_cc64({a2}, {b2});"),
        L(f"{r3} = subc_cc64({a3}, {b3});"),
        L(f"sb  = subc64(0, 0);"),
        L(f"ap0 = PL0 & sb;"),
        L(f"{r0} = add_cc64({r0}, ap0);"),
        L(f"{r1} = addc_cc64({r1}, sb);"),
        L(f"{r2} = addc_cc64({r2}, sb);"),
        L(f"{r3} = addc64({r3}, sb);"),
    ]


def _field_negate_tmpl(y: List[str], out: List[str]) -> List[str]:
    """
    Modular field negation: out = -y mod P = (0 - y) mod P.
    Branchless via subtract-from-zero + conditional P add-back.
    - y == 0: 0-0=0, no borrow, add-back=0, result=0. Correct.
    - y != 0: 0-y borrows; add-back P gives P-y in (0,P). Correct.
    Uses shared vars sb, ap0, PL0 (pre-declared).
    """
    y0, y1, y2, y3 = y
    r0, r1, r2, r3 = out
    L = lambda s: f"    {s}"
    return [
        L(f"{r0} = sub_cc64(0, {y0});"),
        L(f"{r1} = subc_cc64(0, {y1});"),
        L(f"{r2} = subc_cc64(0, {y2});"),
        L(f"{r3} = subc_cc64(0, {y3});"),
        L(f"sb  = subc64(0, 0);"),
        L(f"ap0 = PL0 & sb;"),
        L(f"{r0} = add_cc64({r0}, ap0);"),
        L(f"{r1} = addc_cc64({r1}, sb);"),
        L(f"{r2} = addc_cc64({r2}, sb);"),
        L(f"{r3} = addc64({r3}, sb);"),
    ]


def _field_double_tmpl(a: List[str], out: List[str]) -> List[str]:
    """
    Modular doubling: out = 2*a mod P.
    Branchless: add-with-carry + conditional subtract-P.
    Uses shared vars dbl_cy, dbl_ds0..3, dbl_db, dbl_dnb (pre-declared).
    """
    a0,a1,a2,a3 = a
    r0,r1,r2,r3 = out
    L = lambda s: f"    {s}"
    return [
        L(f"{r0} = add_cc64({a0}, {a0});"),
        L(f"{r1} = addc_cc64({a1}, {a1});"),
        L(f"{r2} = addc_cc64({a2}, {a2});"),
        L(f"{r3} = addc_cc64({a3}, {a3});"),
        L(f"dbl_cy = addc64(0, 0);"),
        L(f"dbl_ds0 = sub_cc64({r0}, 0xFFFFFFFEFFFFFC2F);"),
        L(f"dbl_ds1 = subc_cc64({r1}, 0xFFFFFFFFFFFFFFFF);"),
        L(f"dbl_ds2 = subc_cc64({r2}, 0xFFFFFFFFFFFFFFFF);"),
        L(f"dbl_ds3 = subc_cc64({r3}, 0xFFFFFFFFFFFFFFFF);"),
        L(f"dbl_db  = subc64(dbl_cy, 0);"),
        L(f"dbl_dnb = ~dbl_db;"),
        L(f"{r0} = ({r0} & dbl_db) | (dbl_ds0 & dbl_dnb);"),
        L(f"{r1} = ({r1} & dbl_db) | (dbl_ds1 & dbl_dnb);"),
        L(f"{r2} = ({r2} & dbl_db) | (dbl_ds2 & dbl_dnb);"),
        L(f"{r3} = ({r3} & dbl_db) | (dbl_ds3 & dbl_dnb);"),
    ]
