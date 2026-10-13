"""
Ingot AST Node Definitions (Phase 1 + WP1 Semantic Layer)

Clean, minimal AST that preserves the strict motor vs transmission separation.
No references to any agent/Hermes concepts.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# ---------------- Expressions ----------------

@dataclass
class Expr:
    pass

@dataclass
class Var(Expr):
    name: str

@dataclass
class IntLit(Expr):
    value: int
    raw: str = ""

@dataclass
class FloatLit(Expr):
    value: float
    raw: str = ""

@dataclass
class BoolLit(Expr):
    value: bool

@dataclass
class BinOp(Expr):
    left: Expr
    op: str
    right: Expr

@dataclass
class UnaryOp(Expr):
    op: str
    expr: Expr

@dataclass
class Index(Expr):
    base: Expr
    index: Expr

@dataclass
class Call(Expr):
    callee: str
    args: List[Expr]

@dataclass
class MemberAccess(Expr):
    """walker.z  or  field.one  -- postfix dot member access."""
    obj: Expr
    member: str

# ---------------- Statements ----------------

@dataclass
class Stmt:
    pass

@dataclass
class VarDecl(Stmt):
    name: str
    type: str
    register: Optional[str] = None   # e.g. "R7" or None
    array_size: Optional[Expr] = None
    # One entry per bracket. None is an unsized leading dimension: T* name[][12][4].
    # A single concrete [N] also sets array_size, so older readers stay valid.
    dims: List[Optional[Expr]] = field(default_factory=list)

@dataclass
class Assign(Stmt):
    lvalue: Expr   # Var or Index
    value: Expr

@dataclass
class ForStmt(Stmt):
    init: Optional[Stmt]
    cond: Optional[Expr]
    update: Optional[Stmt]
    body: 'Block'

@dataclass
class WhileStmt(Stmt):
    cond: Expr
    body: 'Block'

@dataclass
class BreakStmt(Stmt):
    pass

@dataclass
class ContinueStmt(Stmt):
    pass

@dataclass
class IfStmt(Stmt):
    cond: Expr
    then_body: 'Block'
    else_body: Optional['Block'] = None

@dataclass
class ReturnStmt(Stmt):
    value: Optional[Expr] = None

@dataclass
class ExprStmt(Stmt):
    expr: Expr

@dataclass
class SharedDeclStmt(Stmt):
    elem_type: str
    name: str
    array_size: Optional[Expr] = None
    # Every shared dimension is a constant. Rank > 1 leaves array_size empty.
    dims: List[Optional[Expr]] = field(default_factory=list)

@dataclass
class SyncThreadsStmt(Stmt):
    barrier_id: int = 0

@dataclass
class WarpShuffleExpr(Expr):
    mode: str             # "up" or "down"
    mask_expr: Expr
    val_expr: Expr
    delta_expr: Expr

@dataclass
class AtomicAddStmt(Stmt):
    target_ptr: Expr
    val_expr: Expr

@dataclass
class Block(Stmt):
    stmts: List[Stmt]

@dataclass
class MotorParam(Stmt):
    """Direction-typed motor parameter: inout jacobian<secp256k1> walker."""
    direction: str   # 'in' | 'out' | 'inout'
    type: str        # 'field', 'jacobian<secp256k1>', 'affine<secp256k1>', or primitive
    name: str

# ---------------- Top level ----------------

@dataclass
class WindowDecl(Stmt):
    """Register-windowed array: @window(R0..R3) u64 X_limbs[4]
    Declares consecutive physical registers as a named array.
    Array access must use compile-time constant indices -- desugared to locked VarDecl nodes
    before semantic checking."""
    name: str
    elem_type: str   # 'u64', 'u32', etc.
    count: int       # number of elements (= end_reg_num - start_reg_num + 1)
    start_reg: str   # 'R0'
    end_reg: str     # 'R3'

@dataclass
class ProfileDecl:
    """Top-level sealed profile: profile NAME { max_registers = 96; spill_allowed = false; }
    Referenced by name from transmission profile clauses."""
    name: str
    properties: Dict[str, Any]  # 'max_registers': 96, 'spill_allowed': False, etc.

@dataclass
class PacketDecl:
    """Top-level packet: a named register contract shared between motors.

    Fields are @window arrays or @register scalars -- the same physical register layout
    that both motors will use.  A motor references a packet with `use packet_name;` in
    its body.  A transmission contract references one with `packet packet_name;` instead
    of listing every field manually.

    Example:
        packet candidate_state {
            @window(R0..R3)  u64 x[4];
            @window(R4..R7)  u64 y[4];
            @register(R8)    u64 y_parity;
            @register(R9)    u64 route_flags;
            @register(R10)   u64 candidate_offset;
        }
    """
    name: str
    fields: List  # List[Union[WindowDecl, VarDecl(register=...)]]

@dataclass
class PacketUse(Stmt):
    """use NAME; inside a motor body -- generic contract-use statement.
    NAME may refer to a packet, rail, target_latch, or route_type.
    The desugar pass inspects prog.packets / prog.rails / etc. to dispatch correctly.
    For packets: grafts locked VarDecl nodes + motor.windows entry.
    For rails: injects NAME_width constant.
    For route_types: injects variant name constants (e.g. LIVE=0, DEAD=1).
    For target_latches: records the latch in motor.used_latches (no injected constants)."""
    packet_name: str  # generic contract name despite the field name

@dataclass
class PacketRef:
    """packet packet_name; inside a motor contract.
    The desugar pass expands it to the equivalent TxParam / RegisterPort entries.
    Direction is always inout (all packet fields are read-write by default)."""
    packet_name: str

@dataclass
class RouteVariant:
    """One variant of a route_type enum: NAME = integer_value."""
    name: str
    value: int

@dataclass
class RouteTypeDecl:
    """route_type NAME { LIVE = 0; DEAD = 1; EXACT_HIT = 2; }

    Typed route states for candidates.  A motor that `use`s a route_type gets each
    variant injected as a compile-time integer constant.  Once a candidate's
    route_flags field equals DEAD the compiler (future: enforced) forbids scheduling
    further work for that lane.

    Example:
        route_type CandidateStatus {
            LIVE      = 0;
            DEAD      = 1;
            EXACT_HIT = 2;
        }
    """
    name: str
    variants: List['RouteVariant']

@dataclass
class RailDecl:
    """rail NAME { packet: PKT; width: N; stride: S; }

    Range-and-stride contract for a keyspace scan.  The width is a compile-time
    constant injected as NAME_width in any motor that `use`s this rail.  The
    packet binding records which fields carry the walking state.

    Example:
        rail PointRail {
            packet: candidate_state;
            width:  1073741824;   // 2^30
            stride: G;            // secp256k1 generator
        }
    """
    name: str
    packet_name: str   # packet that carries the walking state
    width: int         # compile-time number of candidates
    stride: str        # symbolic stride name (e.g. "G")

@dataclass
class TargetLatchDecl:
    """target_latch NAME { target_bytes: N; compare_at: [stage1, stage2, ...]; }

    Staged comparison descriptor for a streaming hash pipeline.  Each name in
    compare_at is a natural early-exit checkpoint -- the motor checks that stage's
    partial digest word before computing the rest.  This removes useless hash rounds
    when a prefix word already fails.

    Example:
        target_latch Hash160Match {
            target_bytes: 20;
            compare_at:   [ ripemd_word0, ripemd_word1, ripemd_full ];
        }
    """
    name: str
    target_bytes: int
    compare_at: List[str]   # stage names, earliest exit first

@dataclass
class MotorDecl:
    name: str
    params: List['MotorParam'] = field(default_factory=list)  # WP1: direction-typed inputs
    register_lock: Optional[str] = None  # motor-level lock if present
    body: Block = field(default_factory=lambda: Block([]))
    # Collected during parse for convenience
    locked_regs: Dict[str, str] = field(default_factory=dict)  # varname -> "R7"
    windows: Dict[str, 'WindowDecl'] = field(default_factory=dict)  # name -> WindowDecl (populated by desugar)
    used_packets: List[str] = field(default_factory=list)    # packet names grafted by desugar
    used_rails: List[str] = field(default_factory=list)      # rail names used by this motor
    used_latches: List[str] = field(default_factory=list)    # target_latch names used by this motor
    used_route_types: List[str] = field(default_factory=list)  # route_type names used by this motor
    used_attractors: List[str] = field(default_factory=list)   # attractor names used by this motor (WP9)
    used_manifolds: List[str] = field(default_factory=list)    # manifold names used by this motor (WP9)

    # Explicit handoff boundary metadata (Precision Bellhousing) -- populated by semantic when register_ports present
    explicit_handoff_active: bool = False
    boundary_physical_regs: set = field(default_factory=set)  # e.g. {"R2", "R3", "R4"}
    boundary_spline_vars: set = field(default_factory=set)    # var names bound to those physicals

@dataclass
class RegisterRange:
    """Register range in a motor contract: registers(R0..R11)."""
    start: str   # 'R0'
    end: str     # 'R11'

@dataclass
class TxParam:
    direction: str   # in | out | inout
    type: str
    name: str
    align: Optional[int] = None
    register_range: Optional['RegisterRange'] = None  # WP1: registers(R0..R11)

@dataclass
class MotorContract:
    motor_name: str
    params: List[TxParam]
    register_ports: List['RegisterPort'] = field(default_factory=list)  # explicit handoff splines (Precision Bellhousing)
    boundary_lock_policy: Optional[str] = None  # e.g. "strict_zero_temps" from @boundary_lock(...)
    packet_refs: List['PacketRef'] = field(default_factory=list)  # packet packet_name; expanded by desugar

@dataclass
class RegisterPort:
    """Explicit register spline at the Motor-Transmission boundary (Precision Bellhousing)."""
    direction: str          # 'in' | 'out' | 'inout'
    register: str           # e.g. "R0"
    type: str               # e.g. "float*"
    name: str               # e.g. "port_a"  (for documentation / future use)

@dataclass
class VerifyClause:
    """verify semantic_equivalence;  or  verify native_text_equal(ref_name);"""
    kind: str            # 'semantic_equivalence', 'native_text_equal', ...
    arg: Optional[str] = None

@dataclass
class RequireClause:
    """require no_spills;  or  require max_registers(96);  or  require input_invariant(expr);"""
    kind: str
    arg: Optional[Any] = None   # int for max_registers, Expr for input_invariant

@dataclass
class TransmissionDecl:
    name: str
    grid: List[Expr]
    block: List[Expr]
    stream: Optional[str] = None
    motor_contract: Optional[MotorContract] = None
    # WP1: sealed lowering profile fields
    target: Optional[str] = None              # e.g. 'nvidia.sm_89'
    profile: Optional[str] = None            # e.g. 'secp256k1_mixed_add_compat_v1'
    verify_clauses: List['VerifyClause'] = field(default_factory=list)
    require_clauses: List['RequireClause'] = field(default_factory=list)

@dataclass
class TensorDecl:
    """tensor name : dtype @align(N); -- physical tensor coordinate in a manifold."""
    name: str
    dtype: str
    alignment: Optional[int] = 16

@dataclass
class ManifoldDecl:
    """manifold Name { tensor t1: q8_0 @align(16); ... } -- physical neural weight manifold."""
    name: str
    tensors: List[TensorDecl] = field(default_factory=list)

@dataclass
class AttractorDecl:
    """attractor Name { pin: [tok1, tok2]; threshold: float; exit_on_hit: bool; }
    Teleological target pinning descriptor: pins target tokens or subspaces at the output."""
    name: str
    target_tokens: List[int] = field(default_factory=list)
    threshold: float = 0.0
    exit_on_hit: bool = True

@dataclass
class Program:
    motors: List[MotorDecl] = field(default_factory=list)
    transmissions: List[TransmissionDecl] = field(default_factory=list)
    profiles: List['ProfileDecl'] = field(default_factory=list)
    packets: List['PacketDecl'] = field(default_factory=list)
    rails: List['RailDecl'] = field(default_factory=list)
    target_latches: List['TargetLatchDecl'] = field(default_factory=list)
    route_types: List['RouteTypeDecl'] = field(default_factory=list)
    manifolds: List['ManifoldDecl'] = field(default_factory=list)      # WP9: AI manifolds
    attractors: List['AttractorDecl'] = field(default_factory=list)    # WP9: AI attractors
    # host_code ignored for Phase 1 compilation
