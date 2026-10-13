"""
Ingot IR (Phase 2 - Real Lowering)

This is the bridge between the high-level motor AST and the PTX backend.

Key guarantees we maintain:
- Locked registers (@register(Rn)) are carried as hard constraints.
- The code generator (PTX emitter) must never allocate compiler temporaries
  into the physical registers the programmer explicitly locked.
- Control flow, memory ops, and arithmetic are represented explicitly.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set
try:
    from .ast_nodes import *
except ImportError:
    from ast_nodes import *

class CompileError(Exception):
    pass

class ShapeError(CompileError):
    """Index rank or declared-shape error.

    This is a language error, not a mutant the wall-breaker should swallow.
    The compiler re-raises it as a compile failure.
    """

@dataclass
class RegConstraint:
    name: str          # "R7"
    ptx_name: str      # "%R7"
    ptx_type: str      # ".f32" ".u64" etc.

@dataclass
class Op:
    pass

# --- Core IR operations (will grow) ---
@dataclass
class LoadParam(Op):
    dest: str
    param_name: str
    ptx_type: str

@dataclass
class LoadGlobal(Op):
    dest: str
    base: str          # register holding base pointer
    offset: str        # register holding byte offset (or u32 element index if byte_stride > 0)
    ptx_type: str
    byte_stride: int = 0  # if > 0 and offset is u32, emitter uses mad.wide.u32
    space: str = "global"  # "global" or "shared"

@dataclass
class StoreGlobal(Op):
    base: str
    offset: str
    src: str
    ptx_type: str
    byte_stride: int = 0  # if > 0 and offset is u32, emitter uses mad.wide.u32
    space: str = "global"

@dataclass
class BinOpIR(Op):
    dest: str
    left: str
    op: str            # '+', '*', etc. (will map to PTX)
    right: str
    ptx_type: str

@dataclass
class CmpIR(Op):
    dest_pred: str     # %p0 etc
    left: str
    op: str
    right: str

@dataclass
class Mov(Op):
    dest: str
    src: str
    ptx_type: str

@dataclass
class Label(Op):
    name: str

@dataclass
class Bra(Op):
    target: str
    pred: Optional[str] = None   # None = unconditional

@dataclass
class ReturnIR(Op):
    pass

@dataclass
class UnaryOpIR(Op):
    dest: str
    op: str       # "~", "-", "!"
    src: str
    ptx_type: str

@dataclass
class CvtIR(Op):
    """Explicit type-widening/narrowing conversion. Emits cvt.dst_ty.src_ty dest, src."""
    dest: str
    src: str
    dest_type: str   # e.g. ".u64"
    src_type: str    # e.g. ".u32"

@dataclass
class IntrinsicOp(Op):
    """Named PTX intrinsic: add.cc, addc, mul.hi, mad.lo.cc, etc."""
    dest: str
    ptx_instr: str   # e.g. "add.cc.u64"
    src1: str
    src2: str
    ptx_type: str
    src3: Optional[str] = None   # for 3-arg ops: mad.lo.cc dest, a, b, c

@dataclass
class ClockReadIR(Op):
    """Hardware cycle counter read.

    clock64() -> mov.u64 dest, %globaltimer;   (64-bit nanosecond timer on sm_89)
    clock32() -> mov.u32 dest, %clock;         (32-bit SM cycle counter)

    These are zero-argument intrinsics; the PTX emitter lowers them to a single
    mov from the appropriate special register.  No carry state is modified.
    """
    dest: str
    width: int   # 64 or 32

@dataclass
class BarrierOp(Op):
    """bar.sync N -- CTA-wide thread barrier (syncthreads)."""
    barrier_id: int = 0

@dataclass
class SharedDeclIR(Op):
    """Declarations for PTX .shared memory space arrays.

    nbytes is the allocation in bytes. array_size stays the element count so
    older constructors still describe a length. The emitter prints .b8[nbytes].
    """
    name: str
    elem_type: str
    array_size: int
    nbytes: int = 0

@dataclass
class CvtaSharedIR(Op):
    """mov.u32 dest, symbol -- shared-space offset of a .shared array.

    cvta.shared.u64 would be the generic address. On this GPU that value is
    the window base 0x4c000000, and st.shared / atom.shared treat the register
    as a shared offset, so the access lands on that boundary.
    """
    dest: str
    symbol: str

@dataclass
class AtomicAddIR(Op):
    """atom.{global,shared}.add.{u32,u64,f32} dest, [addr], val.

    ptr_reg is a .u64 address. offset is a constant byte displacement or a
    register. byte_stride scales a u32 element index with mad.wide when it
    is nonzero. The addressed element's type selects the opcode and the
    value register. A shared element uses space "shared".
    """
    dest: str
    ptr_reg: str
    val_reg: str
    ptx_type: str = ".u32"
    space: str = "global"
    offset: str = "0"
    byte_stride: int = 0

@dataclass
class ShflOp(Op):
    """Warp shuffle -- shfl.sync.{mode}.b32 dest, src, sel, clamp, membermask.

    mode:       "idx" | "up" | "down" | "bfly"
    src:        register holding the value to broadcast
    sel:        register holding lane-index / delta / XOR-mask
    membermask: register holding participating-lane mask (typically 0xffffffff for full warp)
    clamp is derived from mode at emission time:
        idx/down/bfly -> 0x1f  (maxLane = 31, full-warp, no segment)
        up            -> 0     (no upper-lane clamp, delta spans full warp)
    """
    dest: str
    mode: str
    src: str
    sel: str
    membermask: str

@dataclass
class Lop3Op(Op):
    """3-input logic operation: lop3.b32 dest, a, b, c, immLut

    immLut is a 0-255 integer encoding the 3-input truth table.  Common values:
      0xCA = choice(a,b,c)   = (a & b) | (~a & c)   — SHA256 Ch, RMD f1
      0xE8 = majority(a,b,c) = (a & b) | (a & c) | (b & c) — SHA256 Maj
      0x96 = XOR(a,b,c)      = a ^ b ^ c             — RMD f0/f4
      0x59 = (a | ~b) ^ c                             — RMD f2
      0xE4 = (a & c) | (b & ~c)                      — RMD f3
      0x2D = a ^ (b | ~c)                             — RMD f4 (left)
    """
    dest: str
    a: str
    b: str
    c: str
    imm: int   # 0-255

@dataclass
class PointNegateIR(Op):
    """Semantic secp256k1 Jacobian point negation: Y_regs <- -Y_regs in-place.
    X and Z register sets are not touched.

    Lowered during PTX emission to the branchless 10-instruction field_negate
    sequence (4x subtract-from-0 + conditional P add-back).

    curve must be "secp256k1".
    y_regs: [Y0, Y1, Y2, Y3] -- the 4 Y limb register names (limb-little-endian).
    neg_regs: [n0, n1, n2, n3] -- 4 scratch registers for the result before writeback.
    sb_reg: scratch for borrow collection.
    ap0_reg: scratch for P[0]-mask product.
    pl0_reg: register pre-loaded with secp256k1 P[0] = 0xFFFFFFFEFFFFFC2F.

    Proof contract (all verified in test_point_negate_kat.py):
      Q + PointNegate(Q) = infinity
      PointNegate(Q).x   = Q.x
      PointNegate(PointNegate(Q)).y = Q.y
      field_negate preserves curve membership
    """
    curve:    str
    y_regs:   List[str]
    neg_regs: List[str]
    sb_reg:   str
    ap0_reg:  str
    pl0_reg:  str

    def __post_init__(self):
        if self.curve != "secp256k1":
            raise ValueError(f"PointNegateIR: unsupported curve '{self.curve}'")
        if len(self.y_regs) != 4:
            raise ValueError(f"PointNegateIR: expected 4 Y registers, got {len(self.y_regs)}")
        if len(self.neg_regs) != 4:
            raise ValueError(f"PointNegateIR: expected 4 neg scratch registers, got {len(self.neg_regs)}")


@dataclass
class CanonicalizeNegationPairIR(Op):
    """Semantic canonicalization of a negation pair {Q, -Q} onto a target Y sheet.

    Selects the member of the pair whose affine Y has the requested parity.
    Returns the selected point and a negated_flag indicating whether Q was
    replaced by -Q.

    This op is NOT lowered to PTX by the current emitter. It is a semantic
    annotation that the profile expander uses to generate the branchless
    canonicalization schedule (direct-sheet warp vs negated-sheet warp).

    REJECT: use of the Jacobian Y register parity as affine Y parity without
    proof. Only an affine-boundary conversion gives the correct parity.
    """
    curve:           str
    y_regs:          List[str]
    target_parity:   int       # 0 = even (02), 1 = odd (03)
    neg_flag_reg:    str       # output: 0 or 1
    pl0_reg:         str


@dataclass
class MotorIR:
    name: str
    locked_regs: Dict[str, RegConstraint] = field(default_factory=dict)
    params: List[dict] = field(default_factory=list)
    ops: List[Op] = field(default_factory=list)
    next_temp: int = 0
    next_label: int = 0
    # Track which physical registers are claimed so the allocator can avoid them
    locked_physical: Set[str] = field(default_factory=set)

    # Machine understanding (for bloat elimination and Native Execution Strokes)
    target_warp_size: int = 32
    max_registers_per_thread: int = 255   # conservative for many architectures

    # Explicit handoff boundary (Precision Bellhousing / splined shaft) -- populated when motor declares register_ports
    # These physical names must be used directly for all dataflow crossing the motor->transmission interface.
    # The emitter is forbidden from inserting generic temps on paths involving these registers.
    boundary_physical: Set[str] = field(default_factory=set)   # e.g. {"R2", "R3", "R4"}
    boundary_drain_targets: Dict[str, str] = field(default_factory=dict)  # physical -> "global" | "shared" | "warp_lane"
    explicit_handoff: bool = False
    reg_types: Dict[str, str] = field(default_factory=dict)      # reg name -> ptx type
    ptr_elem_types: Dict[str, str] = field(default_factory=dict) # var name -> element ptx type for pointer vars

    def new_temp(self, prefix: str = "r") -> str:
        # P16 zero-slippage enforcement: when explicit_handoff is active, skip any
        # index N where R{N} is a declared register_port physical register.
        # This ensures generic temps never share a slot number with a locked physical.
        n = self.next_temp
        self.next_temp += 1
        if self.explicit_handoff and self.locked_physical:
            while f"R{n}" in self.locked_physical:
                n = self.next_temp
                self.next_temp += 1
        reg = f"%{prefix}{n}"
        # Record type so _ptx_type_for_reg can look it up
        ty = {
            "r": ".u32", "f": ".f32", "rd": ".u64", "p": ".pred"
        }.get(prefix, ".u32")
        self.reg_types[reg] = ty
        return reg

    def new_label(self, prefix: str = "L") -> str:
        n = self.next_label
        self.next_label += 1
        return f"{prefix}_{n}"

    def claim_locked(self, physical: str):
        self.locked_physical.add(physical)


class IRBuilder:
    """
    Real AST -> IR lowering for motors.

    Produces a flat list of ops with explicit register demands.
    Locked registers from @register(Rn) are recorded so later codegen
    can treat them as hard physical names that must not be reused for temps.
    """

    def __init__(self):
        self.var_to_reg: Dict[str, str] = {}   # source var name -> IR register name
        self.mir: Optional[MotorIR] = None
        self._lit_cache: Dict[tuple, str] = {}  # (type, value) -> register holding that literal
        self._loop_stack: List[tuple] = []      # stack of (head_label, exit_label) for break/continue
        self._reg_const: Dict[str, int] = {}   # register -> known integer value (MUL-BY-CONST)
        # name -> {dims: list[int|None], elem_ty: str, space: "global"|"shared"}
        self.shapes: Dict[str, dict] = {}
        self._shared_addr: Dict[str, str] = {}  # shared name -> cvta dest register

    def lower_motor(self, motor: MotorDecl, contract: Optional[MotorContract], machine_model: dict = None) -> MotorIR:
        self.var_to_reg.clear()
        self._lit_cache.clear()
        self._reg_const.clear()
        self.shapes.clear()
        self._shared_addr.clear()
        mm = machine_model or {}
        self.mir = MotorIR(
            name=motor.name,
            target_warp_size=mm.get("warp_size", 32),
            max_registers_per_thread=mm.get("max_regs", 255)
        )

        # 1. Record all explicitly locked registers (these are sacred)
        for var_name, phys in motor.locked_regs.items():
            ptx_ty = self._guess_ptx_type_for_var(motor, var_name)
            rc = RegConstraint(phys, "%" + phys, ptx_ty)
            self.mir.locked_regs[var_name] = rc
            self.mir.claim_locked(phys)
            # Track element type for pointer vars (u32* -> .u32, float* -> .f32)
            if ptx_ty == ".u64":
                self.mir.ptr_elem_types[var_name] = self._elem_ptx_type(motor, var_name)
            # Map the source variable directly to the locked physical name
            self.var_to_reg[var_name] = "%" + phys

        # 2b. Explicit handoff boundary (splined shafts) -- treat boundary physicals as additional sacred locked set
        # When present, IR lowering must never introduce new_temp for expressions whose home is a boundary physical.
        # The emitter will later enforce immediate-drain (value written to %R_n must be consumed by final fabric op).
        if getattr(motor, 'explicit_handoff_active', False):
            self.mir.explicit_handoff = True
            self.mir.boundary_physical = getattr(motor, 'boundary_physical_regs', set()).copy()
            for phys in self.mir.boundary_physical:
                self.mir.claim_locked(phys)  # never let allocator touch these
            # Also map any spline vars we can see so _expr_to_reg returns the physical directly
            for vname in getattr(motor, 'boundary_spline_vars', set()):
                if vname in motor.locked_regs:
                    self.var_to_reg[vname] = "%" + motor.locked_regs[vname]

        # 2. Record transmission contract parameters
        if contract:
            for i, p in enumerate(contract.params):
                self.mir.params.append({
                    "idx": i, "name": p.name, "direction": p.direction, "type": p.type
                })

        # 2c. Emit LoadParam ops for non-pointer scalar contract params with no locked reg.
        #     These are host-passed kernel args (e.g. `int n`, `float a`) that become usable IR registers.
        if contract:
            for p in contract.params:
                if '*' not in p.type and p.name not in self.var_to_reg:
                    if "f32" in p.type or "float" in p.type:
                        ptx_ty, bank = ".f32", "f"
                    elif "f64" in p.type or "double" in p.type:
                        ptx_ty, bank = ".f64", "rd"
                    else:
                        ptx_ty, bank = ".u32", "r"
                    reg = self.mir.new_temp(bank)
                    self.mir.ops.append(LoadParam(dest=reg, param_name=p.name, ptx_type=ptx_ty))
                    self.var_to_reg[p.name] = reg
                    self.mir.reg_types[reg] = ptx_ty

        # 3. Lower the actual body statements (machine understanding active)
        self.mir.ops.append(Label(self.mir.new_label("entry")))
        for stmt in motor.body.stmts:
            self._lower_stmt(stmt)

        # 4. Dead-code elimination: remove ops whose results are never read.
        # This eliminates bloat from dead local variables (e.g. vars assigned in a {} block
        # that are never referenced after -- a common mutation pattern that inflates temp banks).
        self._dce(self.mir)

        # 5. Store block_size from TX geometry for register-budget awareness downstream.
        self.mir.block_size = mm.get("block_size", 256)

        # P16: Zero-slippage post-allocation assertion.
        # Scan every allocated temp register and verify none share a slot number with
        # a locked_physical (register_port) entry.  new_temp() skips these slots at
        # allocation time, but this assertion catches any path that bypassed new_temp()
        # or where locked_physical was populated after some temps were already issued.
        if self.mir.explicit_handoff and self.mir.locked_physical:
            # Build reverse map: bare physical name -> port variable name for error messages
            _port_name_map: Dict[str, str] = {}
            for vname, rc in self.mir.locked_regs.items():
                _port_name_map[rc.name] = vname  # rc.name is bare e.g. "R2"
            for bp in self.mir.boundary_physical:
                if bp not in _port_name_map:
                    _port_name_map[bp] = bp

            import re as _re_p16
            for reg in list(self.mir.reg_types.keys()):
                # reg is e.g. "%r5", "%rd3", "%f2", "%p1" -- skip locked physicals like "%R2"
                bare = reg.lstrip('%')
                if bare.startswith('R'):
                    continue  # this IS a locked physical, not a generic temp
                # Extract numeric suffix: "rd12"->12, "r5"->5, "f3"->3, "p0"->0
                m = _re_p16.match(r'^(?:rd|r|f|p)(\d+)$', bare)
                if not m:
                    continue
                slot = int(m.group(1))
                locked_name = f"R{slot}"
                if locked_name in self.mir.locked_physical:
                    port_display = _port_name_map.get(locked_name, locked_name)
                    raise CompileError(
                        f"Zero-slippage violation: temp register {reg} assigned to "
                        f"locked physical {locked_name} (register_port {port_display})"
                    )

        return self.mir

    def _dce(self, mir: MotorIR) -> None:
        """
        Single-pass dead-code elimination over the op list.
        Removes BinOpIR / Mov / UnaryOpIR whose dest is never used as a source
        anywhere in the op list, and is not a locked physical register.
        Iterates until convergence so chains of dead ops collapse fully.
        """
        PURGEABLE = (BinOpIR, Mov, UnaryOpIR, CvtIR)

        def _read_set(ops):
            rs = set()
            for op in ops:
                if isinstance(op, BinOpIR):
                    rs.add(op.left); rs.add(op.right)
                elif isinstance(op, Mov):
                    rs.add(op.src)
                elif isinstance(op, UnaryOpIR):
                    rs.add(op.src)
                elif isinstance(op, LoadGlobal):
                    rs.add(op.base); rs.add(op.offset)
                elif isinstance(op, StoreGlobal):
                    rs.add(op.base); rs.add(op.offset); rs.add(op.src)
                elif isinstance(op, CmpIR):
                    rs.add(op.left); rs.add(op.right)
                elif isinstance(op, CvtIR):
                    rs.add(op.src)
                elif isinstance(op, IntrinsicOp):
                    rs.add(op.src1); rs.add(op.src2)
                    if op.src3 is not None:
                        rs.add(op.src3)
                elif isinstance(op, Bra):
                    if op.pred:
                        rs.add(op.pred.lstrip('!').lstrip('@'))
                elif isinstance(op, AtomicAddIR):
                    rs.add(op.ptr_reg)
                    rs.add(op.val_reg)
                    if isinstance(op.offset, str):
                        rs.add(op.offset)
            return rs

        locked = mir.locked_physical  # e.g. {"R0", "R2"} -- no % prefix

        changed = True
        while changed:
            changed = False
            reads = _read_set(mir.ops)
            new_ops = []
            for op in mir.ops:
                if isinstance(op, PURGEABLE):
                    dest = op.dest  # e.g. "%f3" or "%R2"
                    dest_bare = dest.lstrip('%')
                    if dest_bare in locked or dest in reads:
                        new_ops.append(op)
                    else:
                        changed = True  # dropped; re-scan for newly dead producers
                else:
                    new_ops.append(op)
            mir.ops = new_ops

    def _guess_ptx_type_for_var(self, motor: MotorDecl, var_name: str) -> str:
        for s in motor.body.stmts:
            if isinstance(s, VarDecl) and s.name == var_name:
                t = s.type.lower()
                if "*" in t or "float*" in t or "double*" in t:
                    return ".u64"
                if "float" in t or "f32" in t:
                    return ".f32"
                if "double" in t or "f64" in t:
                    return ".f64"
                if "u64" in t or "i64" in t:
                    return ".u64"
                if "u32" in t or "uint" in t or "int" in t:
                    return ".u32"
                return ".u32"
        return ".f32"

    def _elem_ptx_type(self, motor: MotorDecl, var_name: str) -> str:
        """Return the element PTX type for a pointer variable (strip the *)."""
        for s in motor.body.stmts:
            if isinstance(s, VarDecl) and s.name == var_name:
                t = s.type.lower().replace("*", "").strip()
                if "u32" in t or "uint" in t:
                    return ".u32"
                if "u64" in t:
                    return ".u64"
                if "float" in t or "f32" in t:
                    return ".f32"
                if "double" in t or "f64" in t:
                    return ".f64"
        return ".f32"

    def _detect_simple_for(self, stmt: ForStmt):
        """
        Detect for(var=0; var<limit; var=var+1) -- the unrollable pattern.
        Returns (counter_name, limit_expr) or None.
        """
        if not stmt.init or not stmt.cond or not stmt.update or not stmt.body:
            return None
        # init: Assign(Var(counter), IntLit(0))
        if not (isinstance(stmt.init, Assign) and isinstance(stmt.init.lvalue, Var)
                and isinstance(stmt.init.value, IntLit) and stmt.init.value.value == 0):
            return None
        counter = stmt.init.lvalue.name
        # cond: counter < expr
        if not (isinstance(stmt.cond, BinOp) and stmt.cond.op == '<'
                and isinstance(stmt.cond.left, Var) and stmt.cond.left.name == counter):
            return None
        limit_expr = stmt.cond.right
        # update: counter = counter + 1
        upd = stmt.update
        if not (isinstance(upd, Assign) and isinstance(upd.lvalue, Var)
                and upd.lvalue.name == counter and isinstance(upd.value, BinOp)
                and upd.value.op == '+' and isinstance(upd.value.left, Var)
                and upd.value.left.name == counter and isinstance(upd.value.right, IntLit)
                and upd.value.right.value == 1):
            return None
        return (counter, limit_expr)

    def _lower_for_unrolled(self, stmt: ForStmt, counter: str, limit_expr, factor: int):
        """
        4x unrolled ForStmt lowering with scalar cleanup epilogue.

        Structure mirrors what nvcc -O3 emits:
          n_main = n - (n & (factor-1))
          while i < n_main:  [body+update] x factor
          while i < n:       [body+update] x 1   (cleanup: 0..factor-1 iters)
        """
        if stmt.init:
            self._lower_stmt(stmt.init)

        i_reg  = self.var_to_reg[counter]
        n_reg  = self._expr_to_reg(limit_expr)
        mask   = factor - 1          # 3 for factor=4

        # n_main = n - (n & mask)
        rem_reg    = self.mir.new_temp("r")
        n_main_reg = self.mir.new_temp("r")
        self.mir.ops.append(BinOpIR(dest=rem_reg,    left=n_reg, op='&', right=str(mask),   ptx_type='.u32'))
        self.mir.ops.append(BinOpIR(dest=n_main_reg, left=n_reg, op='-', right=rem_reg,     ptx_type='.u32'))

        # -- main loop: (body + update) x factor per branch ------------------
        head_m = self.mir.new_label("for_head")
        exit_m = self.mir.new_label("for_exit")
        self.mir.ops.append(Label(head_m))
        p_main = self.mir.new_temp("p")
        self.mir.ops.append(BinOpIR(dest=p_main, left=i_reg, op='<', right=n_main_reg, ptx_type='.u32'))
        self.mir.ops.append(Bra(target=exit_m, pred='!' + p_main))

        self._loop_stack.append((head_m, exit_m))
        for _ in range(factor):
            for s in stmt.body.stmts:
                self._lower_stmt(s)
            if stmt.update:
                self._lower_stmt(stmt.update)
        self._loop_stack.pop()

        self.mir.ops.append(Bra(target=head_m))
        self.mir.ops.append(Label(exit_m))

        # -- cleanup loop: handles n % factor remaining iterations ------------
        head_c = self.mir.new_label("for_head")
        exit_c = self.mir.new_label("for_exit")
        self.mir.ops.append(Label(head_c))
        p_clean = self.mir.new_temp("p")
        self.mir.ops.append(BinOpIR(dest=p_clean, left=i_reg, op='<', right=n_reg, ptx_type='.u32'))
        self.mir.ops.append(Bra(target=exit_c, pred='!' + p_clean))

        self._loop_stack.append((head_c, exit_c))
        for s in stmt.body.stmts:
            self._lower_stmt(s)
        if stmt.update:
            self._lower_stmt(stmt.update)
        self._loop_stack.pop()

        self.mir.ops.append(Bra(target=head_c))
        self.mir.ops.append(Label(exit_c))

    def _lower_for_scalar(self, stmt: ForStmt):
        """Original scalar (1x) ForStmt lowering -- fallback for complex loops."""
        if stmt.init:
            self._lower_stmt(stmt.init)
        head = self.mir.new_label("for_head")
        exit_lbl = self.mir.new_label("for_exit")
        self.mir.ops.append(Label(head))
        if stmt.cond:
            cond_reg = self._expr_to_reg(stmt.cond)
            if not cond_reg.startswith('%p'):
                pred = self.mir.new_temp("p")
                self.mir.ops.append(CmpIR(dest_pred=pred, left=cond_reg, op="ne", right="0"))
                cond_reg = pred
            self.mir.ops.append(Bra(target=exit_lbl, pred="!" + cond_reg))
        if stmt.body:
            self._loop_stack.append((head, exit_lbl))
            for s in stmt.body.stmts:
                self._lower_stmt(s)
            self._loop_stack.pop()
        if stmt.update:
            self._lower_stmt(stmt.update)
        self.mir.ops.append(Bra(target=head))
        self.mir.ops.append(Label(exit_lbl))

    def _elem_bytes(self, elem_ty: str) -> int:
        # Same width rule as the unshaped 1D path: only .u64 is 8 bytes.
        return 8 if elem_ty == ".u64" else 4

    def _elem_ty_from_str(self, type_str: str, shared: bool = False) -> str:
        t = (type_str or "").lower().replace("*", "").strip()
        if "u32" in t or t == "uint":
            return ".u32"
        if "u64" in t or "i64" in t:
            return ".u64"
        if "float" in t or "f32" in t:
            return ".f32"
        if "double" in t or "f64" in t:
            return ".f64"
        # Shared `int` is the language's .u32. A global `int*` keeps the
        # historical element type, which this helper's fall-through is.
        if shared and t in ("int", "i32"):
            return ".u32"
        return ".f32"

    def _const_dims(self, name: str, dims_expr, allow_leading_unsized: bool) -> list:
        dims = []
        for i, dim in enumerate(dims_expr):
            if dim is None:
                if allow_leading_unsized and i == 0:
                    dims.append(None)
                    continue
                raise ShapeError(
                    f"dimension {i} of '{name}' must be a positive integer literal"
                )
            if not isinstance(dim, IntLit) or dim.value <= 0 or dim.value > 0xFFFFFFFF:
                raise ShapeError(
                    f"dimension {i} of '{name}' must be a positive integer literal"
                )
            dims.append(dim.value)
        return dims

    def _note_var_shape(self, stmt: VarDecl) -> None:
        dims_expr = list(getattr(stmt, "dims", None) or [])
        if not dims_expr:
            return
        is_ptr = "*" in (getattr(stmt, "type", "") or "")
        multi = len(dims_expr) > 1 or any(dim is None for dim in dims_expr)
        if not is_ptr:
            if multi:
                raise ShapeError(
                    f"'{stmt.name}' has a multi-dimensional shape but is not a pointer. "
                    f"Declare it as T* {stmt.name}[][...] or __shared__ T {stmt.name}[...]."
                )
            # A single concrete local `float buf[16]` has no pointer shape.
            return
        dims = self._const_dims(stmt.name, dims_expr, allow_leading_unsized=True)
        elem = self._elem_ty_from_str(stmt.type)
        self.shapes[stmt.name] = {"dims": dims, "elem_ty": elem, "space": "global"}
        self.mir.ptr_elem_types[stmt.name] = elem

    def _lower_shared(self, stmt: SharedDeclStmt) -> None:
        dims_expr = list(getattr(stmt, "dims", None) or [])
        if not dims_expr and getattr(stmt, "array_size", None) is not None:
            dims_expr = [stmt.array_size]
        if not dims_expr:
            raise ShapeError(f"shared array '{stmt.name}' needs a constant bound")
        dims = self._const_dims(stmt.name, dims_expr, allow_leading_unsized=False)
        elem = self._elem_ty_from_str(stmt.elem_type, shared=True)
        count = 1
        for extent in dims:
            count *= extent
        nbytes = count * self._elem_bytes(elem)
        self.shapes[stmt.name] = {"dims": dims, "elem_ty": elem, "space": "shared"}
        self.mir.ptr_elem_types[stmt.name] = elem
        self.mir.ops.append(SharedDeclIR(
            name=stmt.name, elem_type=elem, array_size=count, nbytes=nbytes,
        ))

    def _shared_base(self, name: str) -> str:
        hit = self._shared_addr.get(name)
        if hit:
            return hit
        dest = self.mir.new_temp("r")
        self.mir.ops.append(CvtaSharedIR(dest=dest, symbol=name))
        self._shared_addr[name] = dest
        return dest

    def _peel_indexes(self, expr: Expr):
        indices = []
        cur = expr
        while isinstance(cur, Index):
            indices.append(cur.index)
            cur = cur.base
        indices.reverse()
        return cur, indices

    def _shaped_address(self, expr: Expr):
        """Address of a declared multi-dimensional index, or None for legacy 1D.

        Returning None leaves the existing single-bracket path untouched, which
        is what the hashed kernels use. A chain whose root has no shape is an
        error: that used to lower as a pointer load at every bracket.
        """
        if not isinstance(expr, Index):
            return None
        root, indices = self._peel_indexes(expr)
        if not isinstance(root, Var):
            if len(indices) == 1:
                return None
            raise ShapeError("multi-dimensional index requires a named array")
        shape = self.shapes.get(root.name)
        if shape is None:
            if len(indices) == 1:
                return None
            raise ShapeError(
                f"'{root.name}' has no declared shape and was indexed with "
                f"{len(indices)} subscripts. Declare T* {root.name}[][rows][cols] "
                f"or __shared__ T {root.name}[rows][cols]."
            )
        dims = shape["dims"]
        if len(indices) != len(dims):
            raise ShapeError(
                f"'{root.name}' is rank {len(dims)} but was indexed with "
                f"{len(indices)} subscripts"
            )
        return self._flatten_shaped(root.name, indices, shape)

    def _flatten_shaped(self, name: str, indices, shape: dict):
        """Row-major element index, then the existing mad.wide byte scale.

        strides[last] = 1, strides[j] = strides[j+1] * dims[j+1].
        `[][12][4]` therefore multiplies the outer index by 48 and the row by 4.
        """
        dims = shape["dims"]
        rank = len(dims)
        strides = [1] * rank
        for j in range(rank - 2, -1, -1):
            strides[j] = strides[j + 1] * dims[j + 1]
            if strides[j] > 0xFFFFFFFF:
                raise ShapeError(f"stride of '{name}' dimension {j} does not fit in u32")
        acc_const = 0
        acc_reg = None
        for idx_expr, stride in zip(indices, strides):
            if isinstance(idx_expr, IntLit):
                term = idx_expr.value * stride
                if acc_reg is None:
                    acc_const += term
                elif term:
                    dest = self.mir.new_temp("r")
                    self.mir.ops.append(BinOpIR(
                        dest=dest, left=acc_reg, op="+", right=str(term), ptx_type=".u32",
                    ))
                    acc_reg = dest
                continue
            idx_reg = self._expr_to_reg(idx_expr)
            ty = self._ptx_type_for_reg(idx_reg)
            if ty in (".f32", ".f64"):
                raise ShapeError(f"index of '{name}' must be an integer")
            if ty == ".u64":
                narrow = self.mir.new_temp("r")
                self.mir.ops.append(CvtIR(
                    dest=narrow, src=idx_reg, dest_type=".u32", src_type=".u64",
                ))
                idx_reg = narrow
            elif ty != ".u32":
                raise ShapeError(f"index of '{name}' must be an integer")
            if stride == 1:
                term_reg = idx_reg
            else:
                term_reg = self.mir.new_temp("r")
                self.mir.ops.append(BinOpIR(
                    dest=term_reg, left=idx_reg, op="*", right=str(stride), ptx_type=".u32",
                ))
            if acc_reg is None:
                if acc_const == 0:
                    acc_reg = term_reg
                else:
                    dest = self.mir.new_temp("r")
                    self.mir.ops.append(BinOpIR(
                        dest=dest, left=term_reg, op="+", right=str(acc_const), ptx_type=".u32",
                    ))
                    acc_reg = dest
            else:
                dest = self.mir.new_temp("r")
                self.mir.ops.append(BinOpIR(
                    dest=dest, left=acc_reg, op="+", right=term_reg, ptx_type=".u32",
                ))
                acc_reg = dest
        elem_ty = shape["elem_ty"]
        elem_size = self._elem_bytes(elem_ty)
        if shape["space"] == "shared":
            base = self._shared_base(name)
        else:
            if name not in self.var_to_reg:
                base_reg = self.mir.new_temp("rd")
                self.var_to_reg[name] = base_reg
            base = self.var_to_reg[name]
        if acc_reg is None:
            return base, str(acc_const * elem_size), 0, elem_ty, shape["space"]
        return base, acc_reg, elem_size, elem_ty, shape["space"]

    def _coerce_store_src(self, elem_ty: str, value: Expr) -> str:
        val_hint = self.mir.new_temp("rd") if elem_ty == ".u64" and isinstance(value, (IntLit, FloatLit)) else None
        src_reg = self._expr_to_reg(value, dest_hint=val_hint)
        if elem_ty == ".u64" and src_reg.startswith("%r") and not src_reg.startswith("%rd"):
            wide = self.mir.new_temp("rd")
            self.mir.ops.append(CvtIR(dest=wide, src=src_reg, dest_type=".u64", src_type=".u32"))
            src_reg = wide
        if elem_ty == ".u32" and self._ptx_type_for_reg(src_reg) == ".u64":
            narrow = self.mir.new_temp("r")
            self.mir.ops.append(CvtIR(dest=narrow, src=src_reg, dest_type=".u32", src_type=".u64"))
            src_reg = narrow
        if isinstance(src_reg, str) and src_reg.startswith("%"):
            src_ty = self._ptx_type_for_reg(src_reg)
            if src_ty != elem_ty and (elem_ty == ".f32" or src_ty == ".f32"):
                src_reg = self._coerce_reg(src_reg, elem_ty)
        return src_reg

    def _atomic_target(self, expr: Expr):
        """Address of an atomicAdd target.

        Returns (base_reg, byte_offset, byte_stride, elem_ty, space).
        `atomicAdd(&p, v)` and `atomicAdd(p, v)` name the same address.
        A shared array uses the .u32 shared-space offset of the symbol.
        A shaped index uses the row-major product. An unshaped `p[i]`
        stays on the 1D byte scale.
        """
        if isinstance(expr, UnaryOp) and expr.op == "&":
            expr = expr.expr
        if isinstance(expr, Index):
            shaped = self._shaped_address(expr)
            if shaped is not None:
                return shaped
            base_name = expr.base.name if isinstance(expr.base, Var) else None
            elem_ty = self.mir.ptr_elem_types.get(base_name, ".f32") if base_name else ".f32"
            base_reg = self._expr_to_reg(expr.base)
            if self._ptx_type_for_reg(base_reg) != ".u64":
                raise ShapeError("atomicAdd index base is not a pointer")
            elem_size = self._elem_bytes(elem_ty)
            if isinstance(expr.index, IntLit):
                return base_reg, str(expr.index.value * elem_size), 0, elem_ty, "global"
            idx_reg = self._expr_to_reg(expr.index)
            idx_ty = self._ptx_type_for_reg(idx_reg)
            if idx_ty in (".f32", ".f64"):
                raise ShapeError("atomicAdd index must be an integer")
            if idx_ty == ".u32":
                return base_reg, idx_reg, elem_size, elem_ty, "global"
            if idx_ty != ".u64":
                wide = self.mir.new_temp("rd")
                self.mir.ops.append(CvtIR(
                    dest=wide, src=idx_reg, dest_type=".u64", src_type=idx_ty,
                ))
                idx_reg = wide
            byte_offset = self.mir.new_temp("rd")
            self.mir.ops.append(BinOpIR(
                dest=byte_offset, left=idx_reg, op="*", right=str(elem_size), ptx_type=".u64",
            ))
            return base_reg, byte_offset, 0, elem_ty, "global"
        if isinstance(expr, Var):
            shape = self.shapes.get(expr.name)
            if shape is not None and shape["space"] == "shared":
                return self._shared_base(expr.name), "0", 0, shape["elem_ty"], "shared"
            if shape is not None:
                if expr.name not in self.var_to_reg:
                    base = self.mir.new_temp("rd")
                    self.var_to_reg[expr.name] = base
                else:
                    base = self.var_to_reg[expr.name]
                if self._ptx_type_for_reg(base) != ".u64":
                    raise ShapeError(f"atomicAdd address '{expr.name}' is not a pointer")
                return base, "0", 0, shape["elem_ty"], shape["space"]
            elem_ty = self.mir.ptr_elem_types.get(expr.name, ".f32")
            base = self._expr_to_reg(expr)
            if self._ptx_type_for_reg(base) != ".u64":
                raise ShapeError(
                    f"atomicAdd address '{expr.name}' is not a pointer or a shared array"
                )
            return base, "0", 0, elem_ty, "global"
        raise ShapeError("atomicAdd address must be a pointer, an index, or a shared array")

    def _lower_atomic_add(self, stmt: AtomicAddStmt) -> None:
        """Element type selects atom.add. The value is converted to that type.

        .f32, .u32, and .u64 have register banks, so those are the opcodes
        this backend can print. .f64 stays a compile error. Plain int* keeps
        the historical .f32 element type from _elem_ptx_type.
        """
        base, offset, stride, elem_ty, space = self._atomic_target(stmt.target_ptr)
        if elem_ty == ".f64":
            raise ShapeError("atomicAdd of f64 has no .f64 register bank in this backend")
        if elem_ty not in (".u32", ".u64", ".f32"):
            raise ShapeError(f"atomicAdd does not support element type {elem_ty}")
        val_reg = self._coerce_store_src(elem_ty, stmt.val_expr)
        bank = {".u32": "r", ".u64": "rd", ".f32": "f"}[elem_ty]
        dest = self.mir.new_temp(bank)
        self.mir.ops.append(AtomicAddIR(
            dest=dest, ptr_reg=base, val_reg=val_reg,
            ptx_type=elem_ty, space=space, offset=str(offset), byte_stride=stride,
        ))

    def _lower_stmt(self, stmt: Stmt):
        if isinstance(stmt, VarDecl):
            self._note_var_shape(stmt)
            # Local variable declaration (not a locked parameter)
            if stmt.name not in self.var_to_reg:
                t = (getattr(stmt, 'type', '') or '').lower()
                if 'float' in t or 'f32' in t or 'double' in t or 'f64' in t:
                    reg = self.mir.new_temp("f")
                elif 'u64' in t or 'i64' in t or 'int64' in t:
                    reg = self.mir.new_temp("rd")
                else:
                    reg = self.mir.new_temp("r")
                self.var_to_reg[stmt.name] = reg
                # Bank-Aligned Manifold Staging (P12) note:
                # When lowering shared memory declarations that come from
                # the transmission, we will force 32-bank alignment here.

        elif isinstance(stmt, Assign):
            if isinstance(stmt.lvalue, Index):
                shaped = self._shaped_address(stmt.lvalue)
                if shaped is not None:
                    base_reg, byte_offset, byte_stride_val, elem_ty, space = shaped
                    src_reg = self._coerce_store_src(elem_ty, stmt.value)
                    self.mir.ops.append(StoreGlobal(
                        base=base_reg, offset=byte_offset, src=src_reg,
                        ptx_type=elem_ty, byte_stride=byte_stride_val, space=space,
                    ))
                else:
                    # Handle y[i] = expr  -> compute address and emit store.
                    # Unshaped single bracket. Hashed 1D kernels stay on this path.
                    base_name = stmt.lvalue.base.name if isinstance(stmt.lvalue.base, Var) else None
                    elem_ty = self.mir.ptr_elem_types.get(base_name, ".f32") if base_name else ".f32"
                    base_reg = self._expr_to_reg(stmt.lvalue.base)
                    # Constant-index fast path: use [ptr+N] addressing, same as rvalue Index.
                    elem_size = 8 if elem_ty == ".u64" else 4
                    if isinstance(stmt.lvalue.index, IntLit):
                        byte_offset = str(stmt.lvalue.index.value * elem_size)
                        byte_stride_val = 0
                    else:
                        idx_reg = self._expr_to_reg(stmt.lvalue.index)
                        if self._ptx_type_for_reg(idx_reg) == ".u32":
                            byte_offset = idx_reg
                            byte_stride_val = elem_size
                        else:
                            if self._ptx_type_for_reg(idx_reg) != ".u64":
                                wide = self.mir.new_temp("rd")
                                self.mir.ops.append(CvtIR(dest=wide, src=idx_reg,
                                                           dest_type=".u64", src_type=self._ptx_type_for_reg(idx_reg)))
                                idx_reg = wide
                            byte_offset_reg = self.mir.new_temp("rd")
                            self.mir.ops.append(BinOpIR(dest=byte_offset_reg, left=idx_reg, op="*", right=str(elem_size), ptx_type=".u64"))
                            byte_offset = byte_offset_reg
                            byte_stride_val = 0
                    # Evaluate the value with a u64 hint when storing to a u64 pointer --
                    # literals like `0` or `1` are then born at the right width with no CvtIR.
                    val_hint = self.mir.new_temp("rd") if elem_ty == ".u64" and isinstance(stmt.value, (IntLit, FloatLit)) else None
                    src_reg = self._expr_to_reg(stmt.value, dest_hint=val_hint)
                    # Fallback widen for non-literal u32 values (loop vars, etc.) -- necessary, not wasted.
                    if elem_ty == ".u64" and src_reg.startswith('%r') and not src_reg.startswith('%rd'):
                        wide = self.mir.new_temp("rd")
                        self.mir.ops.append(CvtIR(dest=wide, src=src_reg, dest_type=".u64", src_type=".u32"))
                        src_reg = wide
                    if elem_ty == ".u32" and self._ptx_type_for_reg(src_reg) == ".u64":
                        narrow = self.mir.new_temp("r")
                        self.mir.ops.append(CvtIR(dest=narrow, src=src_reg, dest_type=".u32", src_type=".u64"))
                        src_reg = narrow
                    # Float memory slots take float registers. An integer source here
                    # is the same mismatch as mul.f32 with a %r operand.
                    if isinstance(src_reg, str) and src_reg.startswith('%'):
                        src_ty = self._ptx_type_for_reg(src_reg)
                        if src_ty != elem_ty and (elem_ty == '.f32' or src_ty == '.f32'):
                            src_reg = self._coerce_reg(src_reg, elem_ty)
                    self.mir.ops.append(StoreGlobal(base=base_reg, offset=byte_offset, src=src_reg, ptx_type=elem_ty, byte_stride=byte_stride_val))
            else:
                # Clutch fix: if destination is a locked physical variable, route directly
                # to its %Rn register. var_to_reg already maps locked vars to %Rn from
                # lower_motor setup. This keeps locked variables in their physical registers
                # across all loop iterations -- the core fix for SHA256-shaped kernels.
                lvalue_name = getattr(stmt.lvalue, 'name', None)
                if lvalue_name and lvalue_name in self.var_to_reg and \
                   self.var_to_reg[lvalue_name].startswith('%R'):
                    dest_reg = self.var_to_reg[lvalue_name]
                else:
                    dest_reg = self._expr_to_reg(stmt.lvalue, as_lvalue=True)
                src_reg = self._expr_to_reg(stmt.value, dest_hint=dest_reg)
                if src_reg != dest_reg:
                    self.mir.ops.append(Mov(dest=dest_reg, src=src_reg, ptx_type=self._ptx_type_for_reg(dest_reg)))
                # MUL-BY-CONST: propagate constant value into dest register.
                val = self._reg_const.get(src_reg)
                if val is not None:
                    self._reg_const[dest_reg] = val
                else:
                    self._reg_const.pop(dest_reg, None)

        elif isinstance(stmt, ForStmt):
            info = self._detect_simple_for(stmt)
            if info:
                self._lower_for_unrolled(stmt, *info, factor=4)
            else:
                self._lower_for_scalar(stmt)

        elif isinstance(stmt, WhileStmt):
            head = self.mir.new_label("while_head")
            exit = self.mir.new_label("while_exit")
            self.mir.ops.append(Label(head))
            cond_reg = self._expr_to_reg(stmt.cond)
            if not cond_reg.startswith('%p'):
                pred = self.mir.new_temp("p")
                self.mir.ops.append(CmpIR(dest_pred=pred, left=cond_reg, op="ne", right="0"))
                cond_reg = pred
            self.mir.ops.append(Bra(target=exit, pred="!" + cond_reg))
            self._loop_stack.append((head, exit))
            for s in stmt.body.stmts:
                self._lower_stmt(s)
            self._loop_stack.pop()
            self.mir.ops.append(Bra(target=head))
            self.mir.ops.append(Label(exit))

        elif isinstance(stmt, BreakStmt):
            if self._loop_stack:
                self.mir.ops.append(Bra(target=self._loop_stack[-1][1]))

        elif isinstance(stmt, ContinueStmt):
            if self._loop_stack:
                self.mir.ops.append(Bra(target=self._loop_stack[-1][0]))

        elif isinstance(stmt, IfStmt):
            then_lbl = self.mir.new_label("if_then")
            else_lbl = self.mir.new_label("if_else") if stmt.else_body else None
            end_lbl = self.mir.new_label("if_end")

            cond_reg = self._expr_to_reg(stmt.cond)
            if not cond_reg.startswith('%p'):
                pred = self.mir.new_temp("p")
                self.mir.ops.append(CmpIR(dest_pred=pred, left=cond_reg, op="ne", right="0"))
                cond_reg = pred
            if else_lbl:
                self.mir.ops.append(Bra(target=else_lbl, pred="!" + cond_reg))
            else:
                self.mir.ops.append(Bra(target=end_lbl, pred="!" + cond_reg))

            self.mir.ops.append(Label(then_lbl))
            for s in stmt.then_body.stmts:
                self._lower_stmt(s)
            self.mir.ops.append(Bra(target=end_lbl))

            if stmt.else_body:
                self.mir.ops.append(Label(else_lbl))
                for s in stmt.else_body.stmts:
                    self._lower_stmt(s)
                self.mir.ops.append(Bra(target=end_lbl))

            self.mir.ops.append(Label(end_lbl))

        elif isinstance(stmt, SharedDeclStmt):
            self._lower_shared(stmt)

        elif isinstance(stmt, SyncThreadsStmt):
            self.mir.ops.append(BarrierOp(barrier_id=stmt.barrier_id))

        elif isinstance(stmt, AtomicAddStmt):
            self._lower_atomic_add(stmt)

        elif isinstance(stmt, ReturnStmt):
            self.mir.ops.append(ReturnIR())

        elif isinstance(stmt, ExprStmt):
            # syncthreads() is void -- emit BarrierOp directly, no dest register needed.
            if isinstance(stmt.expr, Call) and stmt.expr.callee in ("syncthreads", "__syncthreads"):
                self.mir.ops.append(BarrierOp(barrier_id=0))
            elif isinstance(stmt.expr, AtomicAddStmt):
                self._lower_atomic_add(stmt.expr)
            else:
                self._expr_to_reg(stmt.expr)

        elif isinstance(stmt, Block):
            for s in stmt.stmts:
                self._lower_stmt(s)

    def _ptx_type_for_reg(self, reg: str) -> str:
        """Return the PTX type for a register."""
        if reg in self.mir.reg_types:
            return self.mir.reg_types[reg]
        if reg and reg.startswith('%R'):
            for rc in self.mir.locked_regs.values():
                if rc.ptx_name == reg:
                    return rc.ptx_type
        return ".f32"

    def _coerce_reg(self, reg: str, want_ty: str) -> str:
        """Return a register of PTX type want_ty holding reg's value.

        Inserts one CvtIR when the types differ. The conversion is the value's
        real representation change (int bits are not float bits). Immediates
        and already-matching registers pass through.
        """
        if not isinstance(reg, str) or not reg.startswith('%'):
            return reg
        have = self._ptx_type_for_reg(reg)
        if have == want_ty:
            return reg
        bank = {'.u32': 'r', '.s32': 'r', '.f32': 'f', '.u64': 'rd', '.s64': 'rd'}.get(want_ty, 'f')
        dest = self.mir.new_temp(bank)
        if self.mir.reg_types.get(dest) != want_ty:
            self.mir.reg_types[dest] = want_ty
        self.mir.ops.append(CvtIR(dest=dest, src=reg, dest_type=want_ty, src_type=have))
        return dest

    def _expr_to_reg(self, expr: Expr, as_lvalue: bool = False, dest_hint: str = None) -> str:
        """Returns the IR register name holding the value (or address for lvalue)."""
        if isinstance(expr, Var):
            if expr.name not in self.var_to_reg:
                # Assume it's a parameter or will be loaded
                reg = self.mir.new_temp("r")
                self.var_to_reg[expr.name] = reg
            return self.var_to_reg[expr.name]

        if isinstance(expr, IntLit):
            # Literal deduplication with type-aware cache.
            # We keep separate cache entries for u32 and u64 uses so a literal that
            # is first encountered in a u32 context doesn't poison later u64 uses.
            # dest_hint drives the width decision only -- we never cache into dest_hint
            # because those slots belong to variables and will be overwritten.
            want_u64 = dest_hint is not None and dest_hint.startswith('%rd')
            key = ('int_u64' if want_u64 else 'int', expr.value)
            if key in self._lit_cache:
                return self._lit_cache[key]
            reg = self.mir.new_temp("rd") if want_u64 else self.mir.new_temp("r")
            ptx_ty = ".u64" if want_u64 else ".u32"
            self.mir.ops.append(Mov(dest=reg, src=str(expr.value), ptx_type=ptx_ty))
            self._lit_cache[key] = reg
            self._reg_const[reg] = expr.value
            return reg

        if isinstance(expr, FloatLit):
            key = ('float', expr.value)
            if key in self._lit_cache:
                return self._lit_cache[key]
            reg = self.mir.new_temp("f")
            self.mir.ops.append(Mov(dest=reg, src=str(expr.value), ptx_type=".f32"))
            self._lit_cache[key] = reg
            return reg

        if isinstance(expr, BinOp):
            # Identity constant folding before emitting any ops:
            #   x + 0, x - 0, x * 1, x / 1  ->  x  (no instruction needed)
            #   0 + x, 1 * x                  ->  x
            def _is_lit(e, val):
                return (isinstance(e, IntLit) and e.value == val) or \
                       (isinstance(e, FloatLit) and float(e.value) == float(val))

            def _identity_keeps_type(literal, other) -> bool:
                # An integer identity (x+0, x*1) does not change the other operand's type.
                # A float identity (x+0.0, x*1.0) promotes an integer operand to f32.
                # Folding that case away drops the promotion and the later float
                # instruction is left holding the integer register.
                if isinstance(literal, IntLit):
                    return True
                if not isinstance(literal, FloatLit):
                    return False
                if isinstance(other, FloatLit):
                    return True
                if isinstance(other, Var) and other.name in self.var_to_reg:
                    return self._ptx_type_for_reg(self.var_to_reg[other.name]) in ('.f32', '.f64')
                return False

            if expr.op in ('+', '-') and _is_lit(expr.right, 0) and _identity_keeps_type(expr.right, expr.left):
                result = self._expr_to_reg(expr.left, dest_hint=dest_hint)
                return result
            if expr.op == '+' and _is_lit(expr.left, 0) and _identity_keeps_type(expr.left, expr.right):
                return self._expr_to_reg(expr.right, dest_hint=dest_hint)
            if expr.op in ('*', '/') and _is_lit(expr.right, 1) and _identity_keeps_type(expr.right, expr.left):
                return self._expr_to_reg(expr.left, dest_hint=dest_hint)
            if expr.op == '*' and _is_lit(expr.left, 1) and _identity_keeps_type(expr.left, expr.right):
                return self._expr_to_reg(expr.right, dest_hint=dest_hint)

            # Evaluate operands with width-propagating hints so literals are born
            # at the right width and no CvtIR is ever needed.
            # Pass 1: evaluate left (no hint yet -- we don't know width until we see it).
            _u64_hint = dest_hint if (dest_hint and dest_hint.startswith('%rd')) else None
            l_hint = _u64_hint if isinstance(expr.left,  (IntLit, FloatLit)) else None
            l = self._expr_to_reg(expr.left, dest_hint=l_hint)
            # Pass 2: if left resolved to u64 and right is a literal, give right a u64 hint.
            # This handles nested BinOps like (x*x + 1) where x*x produces u64 but
            # the `+1` BinOp has no dest_hint -- the `1` still needs to be born u64.
            l_ty = self._ptx_type_for_reg(l)
            if l_ty == ".u64" and isinstance(expr.right, (IntLit, FloatLit)):
                r_hint = self.mir.new_temp("rd")
            elif _u64_hint and isinstance(expr.right, (IntLit, FloatLit)):
                r_hint = _u64_hint
            else:
                r_hint = None
            r = self._expr_to_reg(expr.right, dest_hint=r_hint)

            # MUL-BY-CONST: fold using cross-statement register constant tracking.
            # Only fold multiply -- +/- zero folds are unsafe across loop back-edges
            # (e.g. i=0 before loop, i=i+1 inside loop would fold to i=1 forever).
            # The +/- literal folds above (_is_lit path) handle the safe AST cases.
            lv = self._reg_const.get(l)
            rv = self._reg_const.get(r)
            if expr.op == '*':
                if lv == 0 or rv == 0:
                    return self._expr_to_reg(IntLit(0), dest_hint=dest_hint)
                if lv == 1:
                    return r
                if rv == 1:
                    return l

            l_ty = self._ptx_type_for_reg(l)
            r_ty = self._ptx_type_for_reg(r)
            # Usual arithmetic conversions. If either operand is f32, the operation
            # is performed in f32. The integer operand is converted first. The
            # destination type does not pull the operation back down to integer:
            # `int y = current_expert * 0.001953` is a float multiply, then a
            # conversion of the result. Pure integer pairs never enter this path.
            _float_ops = {'+', '-', '*', '/', '%', '<', '>', '<=', '>=', '==', '!='}
            if expr.op in _float_ops and (l_ty == '.f32' or r_ty == '.f32') and '.f64' not in (l_ty, r_ty):
                l = self._coerce_reg(l, '.f32')
                r = self._coerce_reg(r, '.f32')
                if expr.op in ('<', '>', '<=', '>=', '==', '!='):
                    dest = self.mir.new_temp('p')
                    self.mir.ops.append(BinOpIR(dest=dest, left=l, op=expr.op, right=r, ptx_type='.f32'))
                    return dest
                if dest_hint and self._ptx_type_for_reg(dest_hint) == '.f32':
                    dest = dest_hint
                else:
                    dest = self.mir.new_temp('f')
                self.mir.ops.append(BinOpIR(dest=dest, left=l, op=expr.op, right=r, ptx_type='.f32'))
                return dest

            if expr.op in ('<', '>', '<=', '>=', '==', '!=', '&&', '||'):
                dest = self.mir.new_temp("p")
                l_ty = self._ptx_type_for_reg(l)
                self.mir.ops.append(BinOpIR(dest=dest, left=l, op=expr.op, right=r, ptx_type=l_ty))
                return dest
            if dest_hint:
                hint_ty = self._ptx_type_for_reg(dest_hint)
                # A float destination does not retype an integer operation.
                # `float x = a * b` with a, b integer is an integer multiply
                # followed by a conversion. Doing the mul in .f32 would emit
                # mul.f32 with %r operands. u64 destinations still widen, below.
                if hint_ty in ('.f32', '.f64') and l_ty not in ('.f32', '.f64') and r_ty not in ('.f32', '.f64'):
                    dest_hint = None
            if dest_hint:
                dest = dest_hint
            else:
                l_ty = self._ptx_type_for_reg(l)
                r_ty = self._ptx_type_for_reg(r)
                # Integer pairs only. Float pairs returned above.
                # u64 dominates u32. Two f32 operands are equal and stay f32.
                if l_ty == r_ty:
                    result_ty = l_ty
                elif ".u64" in (l_ty, r_ty):
                    result_ty = ".u64"
                elif ".f32" in (l_ty, r_ty) or ".f64" in (l_ty, r_ty):
                    result_ty = ".f32"
                else:
                    result_ty = ".u32"
                bank = {".u32": "r", ".f32": "f", ".u64": "rd"}.get(result_ty, "r")
                dest = self.mir.new_temp(bank)
            ptx_ty = self._ptx_type_for_reg(dest)
            # Widen any operands that don't match the instruction type.
            # Necessary when a cached u32 literal is reused in a u64 arithmetic context.
            if ptx_ty == ".u64":
                for attr in ('l', 'r'):
                    reg = l if attr == 'l' else r
                    if self._ptx_type_for_reg(reg) != ".u64" and reg.startswith('%'):
                        wide = self.mir.new_temp("rd")
                        self.mir.ops.append(CvtIR(dest=wide, src=reg,
                                                   dest_type=".u64", src_type=self._ptx_type_for_reg(reg)))
                        if attr == 'l': l = wide
                        else: r = wide
            elif ptx_ty == ".u32":
                for attr in ('l', 'r'):
                    reg = l if attr == 'l' else r
                    if self._ptx_type_for_reg(reg) == ".u64" and reg.startswith('%'):
                        narrow = self.mir.new_temp("r")
                        self.mir.ops.append(CvtIR(dest=narrow, src=reg,
                                                  dest_type=".u32", src_type=".u64"))
                        if attr == 'l': l = narrow
                        else: r = narrow
            # PTX shift amount must be u32 -- downcast if it's u64
            r_final = r
            if expr.op in ('<<', '>>') and self._ptx_type_for_reg(r) == ".u64":
                r32 = self.mir.new_temp("r")
                self.mir.ops.append(CvtIR(dest=r32, src=r, dest_type=".u32", src_type=".u64"))
                r_final = r32
            self.mir.ops.append(BinOpIR(dest=dest, left=l, op=expr.op, right=r_final, ptx_type=ptx_ty))
            return dest

        if isinstance(expr, Index):
            shaped = self._shaped_address(expr)
            if shaped is not None:
                base, byte_offset, byte_stride_val, elem_ty, space = shaped
                bank = {".u32": "r", ".f32": "f", ".u64": "rd"}.get(elem_ty, "f")
                dest = dest_hint if dest_hint else self.mir.new_temp(bank)
                ptx_ty = self._ptx_type_for_reg(dest) if dest_hint else elem_ty
                self.mir.ops.append(LoadGlobal(
                    dest=dest, base=base, offset=byte_offset, ptx_type=ptx_ty,
                    byte_stride=byte_stride_val, space=space,
                ))
                return dest
            base = self._expr_to_reg(expr.base)
            # Constant-index fast path: compute byte offset at compile time.
            # Passes a numeric string to LoadGlobal/StoreGlobal so the emitter
            # can use [ptr+N] addressing instead of cvt+add+ld.
            base_name = expr.base.name if isinstance(expr.base, Var) else None
            elem_ty = self.mir.ptr_elem_types.get(base_name, ".f32") if base_name else ".f32"
            elem_size = 8 if elem_ty == ".u64" else 4
            if isinstance(expr.index, IntLit):
                byte_offset = str(expr.index.value * elem_size)
                byte_stride_val = 0
            else:
                idx = self._expr_to_reg(expr.index)
                if self._ptx_type_for_reg(idx) == ".u32":
                    # U64-INDEX: keep u32 element index for mad.wide.u32 emission.
                    # Emitter issues: mad.wide.u32 addr, idx_u32, elem_size, base
                    byte_offset = idx
                    byte_stride_val = elem_size
                else:
                    if self._ptx_type_for_reg(idx) != ".u64":
                        wide = self.mir.new_temp("rd")
                        self.mir.ops.append(CvtIR(dest=wide, src=idx,
                                                  dest_type=".u64", src_type=self._ptx_type_for_reg(idx)))
                        idx = wide
                    byte_offset_reg = self.mir.new_temp("rd")
                    self.mir.ops.append(BinOpIR(dest=byte_offset_reg, left=idx, op="*", right=str(elem_size), ptx_type=".u64"))
                    byte_offset = byte_offset_reg
                    byte_stride_val = 0
            bank = {".u32": "r", ".f32": "f", ".u64": "rd"}.get(elem_ty, "f")
            dest = dest_hint if dest_hint else self.mir.new_temp(bank)
            ptx_ty = self._ptx_type_for_reg(dest) if dest_hint else elem_ty
            self.mir.ops.append(LoadGlobal(dest=dest, base=base, offset=byte_offset, ptx_type=ptx_ty, byte_stride=byte_stride_val))
            return dest

        if isinstance(expr, UnaryOp):
            src = self._expr_to_reg(expr.expr)
            src_ty = self._ptx_type_for_reg(src)
            bank = {".u32": "r", ".f32": "f", ".u64": "rd"}.get(src_ty, "r")
            dest = dest_hint if dest_hint else self.mir.new_temp(bank)
            self.mir.ops.append(UnaryOpIR(dest=dest, op=expr.op, src=src, ptx_type=src_ty))
            return dest

        if isinstance(expr, Call):
            # Two-argument PTX intrinsics -- carry chain + mulhi
            INTRINSICS_2 = {
                "add_cc":      ("add.cc.u32",  ".u32", "r"),
                "addc":        ("addc.u32",    ".u32", "r"),
                "addc_cc":     ("addc.cc.u32", ".u32", "r"),
                "sub_cc":      ("sub.cc.u32",  ".u32", "r"),
                "subc":        ("subc.u32",    ".u32", "r"),
                "subc_cc":     ("subc.cc.u32", ".u32", "r"),
                "mulhi":       ("mul.hi.u32",  ".u32", "r"),
                "add_cc64":    ("add.cc.u64",  ".u64", "rd"),
                "addc64":      ("addc.u64",    ".u64", "rd"),
                "addc_cc64":   ("addc.cc.u64", ".u64", "rd"),
                "sub_cc64":    ("sub.cc.u64",  ".u64", "rd"),
                "subc64":      ("subc.u64",    ".u64", "rd"),
                "subc_cc64":   ("subc.cc.u64", ".u64", "rd"),
                "mulhi64":     ("mul.hi.u64",  ".u64", "rd"),
                "mul_lo64":    ("mul.lo.u64",  ".u64", "rd"),
            }
            if expr.callee in INTRINSICS_2 and len(expr.args) == 2:
                ptx_instr, ptx_ty, bank = INTRINSICS_2[expr.callee]
                # For literal args: only pre-allocate a hint if the value isn't cached yet.
                # This lets repeated constants (0, 0xFFFFFFFFFFFFFFFF) reuse one register.
                def _lit_hint(arg):
                    if bank != "rd" or not isinstance(arg, (IntLit, FloatLit)):
                        return None
                    # Always allocate an rd hint so _expr_to_reg uses the ('int_u64',...)
                    # cache key. On a cache hit _expr_to_reg returns the cached register
                    # and the hint is allocated but never written -- negligible overhead.
                    return self.mir.new_temp(bank)
                h1 = _lit_hint(expr.args[0])
                h2 = _lit_hint(expr.args[1])
                s1 = self._expr_to_reg(expr.args[0], dest_hint=h1)
                s2 = self._expr_to_reg(expr.args[1], dest_hint=h2)
                # Use dest_hint directly if type matches -- eliminates copy-on-write mov
                if dest_hint and self._ptx_type_for_reg(dest_hint) == ptx_ty:
                    dest = dest_hint
                else:
                    dest = self.mir.new_temp(bank)
                self.mir.ops.append(IntrinsicOp(dest=dest, ptx_instr=ptx_instr,
                                                src1=s1, src2=s2, ptx_type=ptx_ty))
                self.mir.reg_types[dest] = ptx_ty
                return dest

            # Three-argument fused multiply-add intrinsics (for field multiplication)
            INTRINSICS_3 = {
                "mad_lo_cc64":   ("mad.lo.cc.u64",  ".u64", "rd"),
                "madc_lo_cc64":  ("madc.lo.cc.u64", ".u64", "rd"),
                "mad_hi_cc64":   ("mad.hi.cc.u64",  ".u64", "rd"),
                "madc_hi64":     ("madc.hi.cc.u64", ".u64", "rd"),
                "madc_hi_cc64":  ("madc.hi.cc.u64", ".u64", "rd"),
                "mad_lo_cc":     ("mad.lo.cc.u32",  ".u32", "r"),
                "madc_lo_cc":    ("madc.lo.cc.u32", ".u32", "r"),
                "mad_hi_cc":     ("mad.hi.cc.u32",  ".u32", "r"),
                "madc_hi":       ("madc.hi.u32",    ".u32", "r"),
            }
            if expr.callee in INTRINSICS_3 and len(expr.args) == 3:
                ptx_instr, ptx_ty, bank = INTRINSICS_3[expr.callee]
                def _lit_hint3(arg):
                    if bank != "rd" or not isinstance(arg, (IntLit, FloatLit)):
                        return None
                    return self.mir.new_temp(bank)
                h1 = _lit_hint3(expr.args[0])
                h2 = _lit_hint3(expr.args[1])
                h3 = _lit_hint3(expr.args[2])
                s1 = self._expr_to_reg(expr.args[0], dest_hint=h1)
                s2 = self._expr_to_reg(expr.args[1], dest_hint=h2)
                s3 = self._expr_to_reg(expr.args[2], dest_hint=h3)
                if dest_hint and self._ptx_type_for_reg(dest_hint) == ptx_ty:
                    dest = dest_hint
                else:
                    dest = self.mir.new_temp(bank)
                self.mir.ops.append(IntrinsicOp(dest=dest, ptx_instr=ptx_instr,
                                                src1=s1, src2=s2, ptx_type=ptx_ty, src3=s3))
                self.mir.reg_types[dest] = ptx_ty
                return dest

            BUILTINS = {
                "__builtin_threadIdx_x": ("%tid.x",   ".u32"),
                "__builtin_threadIdx_y": ("%tid.y",   ".u32"),
                "__builtin_threadIdx_z": ("%tid.z",   ".u32"),
                "__builtin_blockIdx_x":  ("%ctaid.x", ".u32"),
                "__builtin_blockIdx_y":  ("%ctaid.y", ".u32"),
                "__builtin_blockIdx_z":  ("%ctaid.z", ".u32"),
                "__builtin_blockDim_x":  ("%ntid.x",   ".u32"),
                "__builtin_blockDim_y":  ("%ntid.y",   ".u32"),
                "__builtin_blockDim_z":  ("%ntid.z",   ".u32"),
                "__builtin_gridDim_x":   ("%nctaid.x", ".u32"),
                "__builtin_gridDim_y":   ("%nctaid.y", ".u32"),
            }
            if expr.callee in BUILTINS:
                ptx_src, ptx_ty = BUILTINS[expr.callee]
                dest = self.mir.new_temp("r")
                self.mir.ops.append(Mov(dest=dest, src=ptx_src, ptx_type=ptx_ty))
                return dest

            # -- Warp shuffle intrinsics -------------------------------------------
            # shfl_sync_u32(mask, src, lane)      -> shfl.sync.idx.b32
            # shfl_up_sync_u32(mask, src, delta)  -> shfl.sync.up.b32
            # shfl_down_sync_u32(mask, src, delta)-> shfl.sync.down.b32
            # shfl_bfly_sync_u32(mask, src, xmask)-> shfl.sync.bfly.b32
            SHFL_MODES = {
                "shfl_sync_u32":       "idx",
                "shfl_up_sync_u32":    "up",
                "shfl_down_sync_u32":  "down",
                "shfl_bfly_sync_u32":  "bfly",
            }
            if expr.callee in SHFL_MODES and len(expr.args) == 3:
                mode = SHFL_MODES[expr.callee]
                mask_reg = self._expr_to_reg(expr.args[0])
                src_reg  = self._expr_to_reg(expr.args[1])
                sel_reg  = self._expr_to_reg(expr.args[2])
                dest = dest_hint if (dest_hint and dest_hint.startswith('%r')
                                     and not dest_hint.startswith('%rd')) else self.mir.new_temp("r")
                self.mir.ops.append(ShflOp(dest=dest, mode=mode, src=src_reg,
                                           sel=sel_reg, membermask=mask_reg))
                self.mir.reg_types[dest] = ".u32"
                return dest

            # -- lane_id() -> %laneid ---------------------------------------------
            if expr.callee == "lane_id" and not expr.args:
                dest = dest_hint if (dest_hint and dest_hint.startswith('%r')
                                     and not dest_hint.startswith('%rd')) else self.mir.new_temp("r")
                self.mir.ops.append(Mov(dest=dest, src="%laneid", ptx_type=".u32"))
                self.mir.reg_types[dest] = ".u32"
                return dest

            # -- clock64() -> mov.u64 dest, %globaltimer --------------------------
            # Returns a 64-bit hardware cycle/timer counter (sm_89: %globaltimer is
            # a nanosecond-resolution global timer shared across the device).
            if expr.callee == "clock64" and not expr.args:
                dest = dest_hint if (dest_hint and dest_hint.startswith('%rd')) else self.mir.new_temp("rd")
                self.mir.ops.append(ClockReadIR(dest=dest, width=64))
                self.mir.reg_types[dest] = ".u64"
                return dest

            # -- clock32() / clock() -> mov.u32 dest, %clock ----------------------
            # Returns a 32-bit SM cycle counter (wraps every ~4 billion cycles).
            if expr.callee in ("clock32", "clock") and not expr.args:
                dest = dest_hint if (dest_hint and dest_hint.startswith('%r')
                                     and not dest_hint.startswith('%rd')) else self.mir.new_temp("r")
                self.mir.ops.append(ClockReadIR(dest=dest, width=32))
                self.mir.reg_types[dest] = ".u32"
                return dest

            # -- lop3(a, b, c, immLut) -> lop3.b32 dest, a, b, c, imm ----------
            # 3-input logic operation.  immLut encodes the 8-entry truth table.
            # Common: 0xCA=choice, 0xE8=majority, 0x96=XOR3, 0x59=(a|~b)^c,
            #         0xE4=(a&c)|(b&~c), 0x2B=a^(b|~c)
            if expr.callee == "lop3" and len(expr.args) == 4:
                ra = self._expr_to_reg(expr.args[0])
                rb = self._expr_to_reg(expr.args[1])
                rc = self._expr_to_reg(expr.args[2])
                # 4th arg must be an integer literal
                imm_expr = expr.args[3]
                if not isinstance(imm_expr, IntLit):
                    raise CompileError("lop3(): 4th argument (immLut) must be an integer literal")
                imm_val = imm_expr.value
                if not (0 <= imm_val <= 255):
                    raise CompileError(f"lop3(): immLut {imm_val} out of range 0-255")
                dest = dest_hint if (dest_hint and dest_hint.startswith('%r')
                                     and not dest_hint.startswith('%rd')) else self.mir.new_temp("r")
                self.mir.ops.append(Lop3Op(dest=dest, a=ra, b=rb, c=rc, imm=imm_val))
                self.mir.reg_types[dest] = ".u32"
                return dest

            raise CompileError(f"Unknown call '{expr.callee}' in motor -- use builtins (threadIdx.x, blockIdx.x) or motor-contract calls only")

        # Fallback
        reg = self.mir.new_temp("r")
        return reg

if __name__ == "__main__":
    print("IRBuilder ready for real AST lowering.")
