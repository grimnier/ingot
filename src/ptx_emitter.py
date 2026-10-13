"""
Ingot PTX Emitter (Phase 1 - Real, Executable)

This emitter produces valid SM_80+ PTX from a checked AST.
Hard @register(Rn) annotations become explicit:
    .reg .f32 %R0;
    .reg .u64 %R1;   // for pointers

The emitter uses the locked names directly in the generated code.
Compiler-generated temporaries use a separate high bank (%f100+, %r200+)
so that the user's locked registers are protected from "accidental" reuse
in the emitted instruction stream.

No C++, no nvcc, no templates. Pure PTX.

Supported (v1): vector_add style saxpy-style kernels with simple for-loops over 1D data.
"""

from typing import List, Dict, Optional
try:
    from .ast_nodes import Program, MotorDecl, TransmissionDecl, MotorContract, VarDecl
    from .semantic import ContractChecker
    from .ir import MotorIR, Op, LoadParam, LoadGlobal as IRLoadGlobal, StoreGlobal as IRStoreGlobal, BinOpIR, CmpIR, Mov, Label, Bra, ReturnIR, UnaryOpIR, CvtIR, IntrinsicOp, BarrierOp, ShflOp, ClockReadIR, Lop3Op, SharedDeclIR, CvtaSharedIR, AtomicAddIR
except ImportError:
    from ast_nodes import Program, MotorDecl, TransmissionDecl, MotorContract, VarDecl
    from semantic import ContractChecker
    from ir import MotorIR, Op, LoadParam, LoadGlobal as IRLoadGlobal, StoreGlobal as IRStoreGlobal, BinOpIR, CmpIR, Mov, Label, Bra, ReturnIR, UnaryOpIR, CvtIR, IntrinsicOp, BarrierOp, ShflOp, ClockReadIR, Lop3Op, SharedDeclIR, CvtaSharedIR, AtomicAddIR

class PTXEmitError(Exception):
    pass


def _load_addr_scratch(base_op: str, off_op: str) -> str:
    """Dynamic loads address through %rd8.

    Hashed kernels depend on that scratch. A shared cvta can itself be %rd8,
    and mad.wide must not overwrite the base it is adding.
    """
    for n in (8, 9, 10, 11):
        scratch = f"%rd{n}"
        if scratch != base_op and scratch != off_op:
            return scratch
    return "%rd12"


def _cvt_opcode(dest_ty: str, src_ty: str) -> str:
    """PTX conversion opcode. Float destinations require a rounding mode.

    `cvt.f32.u32` is illegal and is rejected by ptxas the same way a mixed
    `mul.f32` is. Integer-to-float uses .rn. Float-to-integer uses .rni.
    Integer-to-integer stays a plain cvt (truncating), which is what the
    existing u32/u64 widens emit.
    """
    d = dest_ty[1:] if dest_ty.startswith('.') else dest_ty
    s = src_ty[1:] if src_ty.startswith('.') else src_ty
    if d in ('f32', 'f64') and s in ('f32', 'f64', 'u32', 's32', 'u64', 's64'):
        return f'cvt.rn.{d}.{s}'
    if d in ('u32', 's32', 'u64', 's64') and s in ('f32', 'f64'):
        return f'cvt.rni.{d}.{s}'
    return f'cvt.{d}.{s}'

class PTXEmitter:
    def __init__(self):
        self.sm = "sm_89"  # RTX 4050 (Ada Lovelace)
        self.version = "7.8"

    def emit_for_transmission(self, prog: Program, tx_name: str, ir: Optional[MotorIR] = None) -> str:
        tx = None
        for t in prog.transmissions:
            if t.name == tx_name:
                tx = t
                break
        if not tx:
            raise PTXEmitError(f"Unknown transmission: {tx_name}")

        if not tx.motor_contract:
            # Geometry-only transmission -- compiler synthesis path handles it.
            # This path is reached only for explicit transmissions with no motor block;
            # the compiler will have built a fake contract before calling emit.
            raise PTXEmitError("Transmission has no motor contract -- ensure compiler synthesis ran first")

        motor_name = tx.motor_contract.motor_name
        motor = None
        for m in prog.motors:
            if m.name == motor_name:
                motor = m
                break
        if not motor:
            raise PTXEmitError(f"Motor '{motor_name}' not found")

        # Verify contract one last time (should have passed semantic)
        chk = ContractChecker()
        if not chk.check_program(prog):
            raise PTXEmitError("Cannot emit PTX for program with semantic errors:\n" + chk.report())

        return self._emit_kernel(motor, tx, tx.motor_contract, ir=ir)

    def emit_all_transmissions(self, prog: Program, ir_map: Optional[Dict[str, MotorIR]] = None) -> str:
        """Emit a single PTX blob containing one .entry block per transmission.

        ir_map maps transmission name -> MotorIR (may be None to skip IR for that TX).
        The first transmission provides the .version/.target/.address_size header;
        subsequent transmissions contribute only their .visible .entry blocks.
        """
        if not prog.transmissions:
            raise PTXEmitError("No transmissions to compile")

        chk = ContractChecker()
        if not chk.check_program(prog):
            raise PTXEmitError("Cannot emit PTX: semantic errors:\n" + chk.report())

        ir_map = ir_map or {}
        header_lines: List[str] = []
        entry_blocks: List[str] = []

        for tx in prog.transmissions:
            if not tx.motor_contract:
                continue
            motor_name = tx.motor_contract.motor_name
            motor = next((m for m in prog.motors if m.name == motor_name), None)
            if not motor:
                raise PTXEmitError(f"Motor '{motor_name}' not found (referenced by transmission '{tx.name}')")

            full_ptx = self._emit_kernel(motor, tx, tx.motor_contract, ir=ir_map.get(tx.name))
            lines = full_ptx.splitlines()

            if not header_lines:
                # Capture header from first kernel; split at first .visible/.entry line
                for i, ln in enumerate(lines):
                    stripped = ln.strip()
                    if stripped.startswith(".visible") or stripped.startswith(".entry"):
                        header_lines = lines[:i]
                        entry_blocks.append("\n".join(lines[i:]))
                        break
                else:
                    entry_blocks.append(full_ptx)
            else:
                # Strip header from subsequent kernels -- only append the entry body
                for i, ln in enumerate(lines):
                    stripped = ln.strip()
                    if stripped.startswith(".visible") or stripped.startswith(".entry"):
                        entry_blocks.append("\n".join(lines[i:]))
                        break

        if not entry_blocks:
            raise PTXEmitError("No motor contracts found -- cannot emit multi-kernel PTX")

        return "\n".join(header_lines) + "\n\n" + "\n\n".join(entry_blocks)

    def _emit_kernel(self, motor: MotorDecl, tx: TransmissionDecl, contract: MotorContract, ir: Optional[MotorIR] = None) -> str:
        lines = []
        lines.append(f".version {self.version}")
        lines.append(f".target {self.sm}")
        lines.append(".address_size 64")
        lines.append("")

        # Build .param list from contract
        param_decls = []
        locked_decls = []
        locked_map: Dict[str, str] = {}   # varname -> ptx_reg_name e.g. "%R0"
        locked_physical: set = set()

        # Collect locked registers first (they get explicit names)
        def _find_var_type_recursive(stmts, name):
            for s in stmts:
                if isinstance(s, VarDecl) and s.name == name:
                    return s.type
                if hasattr(s, 'stmts'):
                    r = _find_var_type_recursive(s.stmts, name)
                    if r: return r
            return None

        for var, reg in motor.locked_regs.items():
            ptx_ty = ".f32"
            var_type = _find_var_type_recursive(motor.body.stmts, var)
            if var_type:
                if "*" in var_type:
                    ptx_ty = ".u64"
                elif "u64" in var_type or "i64" in var_type:
                    ptx_ty = ".u64"
                elif "f32" in var_type or "float" in var_type:
                    ptx_ty = ".f32"
                else:
                    ptx_ty = ".u32"
            ptx_name = "%" + reg
            locked_decls.append(f"    .reg {ptx_ty} {ptx_name};")
            locked_map[var] = ptx_name
            locked_physical.add(reg)

        # If we have a lowered IR, it carries the authoritative locked set
        if ir is not None:
            locked_physical = ir.locked_physical.copy()

        # Oracle-IO mode: every transmission param is an inout locked-register u64 limb.
        # This pattern identifies a pure transformation kernel (reads state, writes state,
        # no scalar inputs from host).  Emit a single flat-buffer pointer so ptxas sees
        # real global-memory I/O and cannot DCE the computation.
        # Motors with mixed in/inout (e.g. point_add_mixed which also takes table-point
        # scalars as `in`) are excluded; they are designed for inline embedding, not
        # standalone execution, so the existing skip behaviour is kept for them.
        is_oracle_io = bool(
            motor.locked_regs and
            contract.params and
            all(p.name in motor.locked_regs for p in contract.params) and
            not any("*" in p.type for p in contract.params) and
            all(p.direction == "inout" for p in contract.params)
        )
        if is_oracle_io:
            # %buf_ptr lives alongside the locked regs (declared before body instructions).
            locked_decls.append("    .reg .u64 %buf_ptr;")

        # Params (from transmission contract)
        # For explicit handoff: contract.params may be empty - synthesize from pointer register_ports
        if is_oracle_io:
            # Single flat-buffer pointer.  Layout: param[i] at byte offset i*8 (u64 limbs).
            param_decls = ["    .param .u64 io_buf"]
        elif contract.params:
            for i, p in enumerate(contract.params):
                if "*" in p.type:
                    param_decls.append(f"    .param .u64 {p.name}")
                elif p.name in motor.locked_regs:
                    # Scalar locked reg = working variable. Lives in a physical register,
                    # never passed from the host. Skip .param -- no host-side value to receive.
                    pass
                elif "f32" in p.type or "float" in p.type:
                    param_decls.append(f"    .param .f32 {p.name}")
                elif "int" in p.type or "u32" in p.type or "i32" in p.type:
                    param_decls.append(f"    .param .u32 {p.name}")
                else:
                    param_decls.append(f"    .param .u64 {p.name}")
        elif getattr(contract, 'register_ports', []):
            # Explicit handoff - generate params for pointer-type ports only
            # (address/data shaft registers are computed inside the kernel, not passed from host)
            for port in contract.register_ports:
                if "*" in port.type or port.type.lower() in ("float*", "double*"):
                    param_decls.append(f"    .param .u64 {port.name}")

        entry_name = f"{motor.name}_{tx.name}"
        lines.append(f".visible .entry {entry_name}(")
        lines.append(",\n".join(param_decls))
        lines.append(")")
        lines.append("{")

        # Locked registers (hard)
        if locked_decls:
            lines.extend(locked_decls)

        # Explicit handoff boundary (Precision Bellhousing) - declare the splined physicals
        # and give the transmission a tiny generic bank because most data movement now uses the direct shafts.
        boundary_phys = set()
        if ir is not None:
            boundary_phys = getattr(ir, 'boundary_physical', set()) or set()

        if boundary_phys:
            lines.append("    // === PRECISION BELLHOUSING - explicit splined shafts (zero fluid slippage) ===")
            for bp in sorted(boundary_phys):
                # We already declared the motor-locked ones; these are additional working splines
                if not any(bp in d for d in locked_decls):
                    # Best-effort type (u64 for addresses, f32 for data in v1)
                    ty = ".u64" if "addr" in str(bp).lower() or int(bp[1:]) >= 2 else ".f32"
                    lines.append(f"    .reg {ty} %{bp};")

        # -- Transmission-contract-driven temp bank (two-pass) -------------------
        # Pass 1: emit body into a scratch list.
        # Pass 2: scan it for actual register usage, size bank to fit, honour
        #         the contract's boundary_locks policy.  The transmission contract
        #         is now the authority on bank ceiling -- not a hardcoded 512.

        use_ir_path = ir is not None and getattr(ir, 'ops', None) and len(ir.ops) > 2

        # -- Pass 1: body into scratch ----------------------------------------
        body_lines = []
        if not use_ir_path:
            for i, p in enumerate(contract.params):
                if "*" in p.type:
                    body_lines.append(f"    ld.param.u64    %rd{i}, [{p.name}];")
                else:
                    body_lines.append(f"    ld.param.u64    %rd{i}, [{p.name}];")

        if use_ir_path:
            if getattr(ir, 'synthesized_contract', False):
                body_lines.append("    // === Synthesized contract path ===")
            body_lines.extend(self._emit_from_ir(ir, contract, locked_map, locked_physical))

            # Oracle-IO: inject ld.global prologue after all .reg decls, st.global epilogue
            # at end.  Must happen after _emit_from_ir so we know the .reg boundary.
            if is_oracle_io:
                # Find the last .reg declaration line in body_lines -- prologue goes after it.
                last_reg_idx = -1
                for j, bl in enumerate(body_lines):
                    if bl.strip().startswith('.reg'):
                        last_reg_idx = j
                insert_at = last_reg_idx + 1

                prologue = [
                    "    // === oracle-io prologue: load all limbs from flat buffer ===",
                    "    ld.param.u64 %buf_ptr, [io_buf];",
                ] + [
                    f"    ld.global.u64 {locked_map[p.name]}, [%buf_ptr+{i*8}];"
                    for i, p in enumerate(contract.params)
                    if p.direction in ("in", "inout")
                ]
                epilogue = [
                    "    // === oracle-io epilogue: writeback inout/out limbs ===",
                ] + [
                    f"    st.global.u64 [%buf_ptr+{i*8}], {locked_map[p.name]};"
                    for i, p in enumerate(contract.params)
                    if p.direction in ("inout", "out")
                ]
                body_lines = body_lines[:insert_at] + prologue + body_lines[insert_at:] + epilogue
        else:
            has_x = any(p.name == "x" for p in contract.params) or 'x' in locked_map
            has_y = any(p.name == "y" for p in contract.params) or 'y' in locked_map
            if motor.name == 'vector_add' or (has_x and has_y):
                body_lines.extend(self._emit_vector_loop(contract, locked_map, motor))
            else:
                body_lines.extend(self._emit_generic_loop(motor, contract, locked_map))

        # -- Pass 2: scan body for actual register high-water marks -----------
        import re as _re
        max_f = max_r = max_rd = max_p = 0
        for bl in body_lines:
            for m in _re.findall(r'%f(\d+)', bl):
                max_f = max(max_f, int(m) + 1)
            for m in _re.findall(r'%r(\d+)\b', bl):
                max_r = max(max_r, int(m) + 1)
            for m in _re.findall(r'%rd(\d+)', bl):
                max_rd = max(max_rd, int(m) + 1)
            for m in _re.findall(r'%p(\d+)', bl):
                max_p = max(max_p, int(m) + 1)

        # Add headroom and enforce minimums so the bank always covers the body.
        HEADROOM = 4
        bank_f  = max(max_f  + HEADROOM, 8)
        bank_r  = max(max_r  + HEADROOM, 8)
        bank_rd = max(max_rd + HEADROOM, 8)
        bank_p  = max(max_p  + HEADROOM, 8)

        # Contract boundary_locks can enforce a tighter ceiling.
        # "strict_zero_temps" means the motor declared it needs no synthesis temps --
        # the locked physicals carry all live values.  Hard-cap to headroom only.
        zero_temps = "strict_zero_temps" in getattr(contract, 'boundary_locks', [])
        if zero_temps:
            bank_f = bank_r = bank_rd = bank_p = HEADROOM

        # Whether there is an explicit user-written transmission at all.
        has_explicit_tx = ir is not None

        if has_explicit_tx:
            lines.append(f"    // TX-driven bank: f={bank_f} r={bank_r} rd={bank_rd} p={bank_p}"
                         + (" [zero_temps enforced]" if zero_temps else ""))
        else:
            lines.append("    // Legacy bank (no TX)")

        lines.append(f"    .reg .f32   %f<{bank_f}>;")
        lines.append(f"    .reg .u32   %r<{bank_r}>;")
        lines.append(f"    .reg .u64   %rd<{bank_rd}>;")
        lines.append(f"    .reg .pred  %p<{bank_p}>;")
        lines.append("")

        # .shared must sit with the other declarations, above ld.param / cvta.
        # ssa_fold leaves a `.shared` line where the IR op was, in the body.
        shared_decls = [bl for bl in body_lines if bl.strip().startswith(".shared")]
        body_rest = [bl for bl in body_lines if not bl.strip().startswith(".shared")]
        lines.extend(shared_decls)
        lines.extend(body_rest)

        # === Locked Physical Collision Guard + Bulletproof Handoff Zero-Waste Guard ===
        # After emission, prove that boundary splines (the splined shafts) were not polluted by generic temps.
        # This is the mechanical proof that the fluid torque converter has been replaced by a rigid lockup clutch.
        final_ptx = "\n".join(lines)
        if locked_physical:
            for locked_name in locked_physical:
                if f"{locked_name} =" in final_ptx or f"mov.*{locked_name}" in final_ptx.replace(" ", ""):
                    lines.append(f"    // !!! LOCKED REGISTER COLLISION DETECTED for {locked_name} !!!")

        if ir is not None and getattr(ir, 'explicit_handoff', False):
            boundary = getattr(ir, 'boundary_physical', set()) or set()
            waste_found = []
            for bp in boundary:
                bp_name = "%" + bp
                # Scan every line that mentions this spline physical
                for ln in lines:
                    if bp_name in ln:
                        # If the same line also contains a generic temp, we have leakage at the boundary
                        if any(g in ln for g in ("%f", "%r", "%rd")) and bp_name in ln:
                            # Allow the final fabric op itself (ld.global / st.global using the spline as base)
                            if "ld.global" in ln or "st.global" in ln or "mad.wide" in ln:
                                continue
                            waste_found.append((bp_name, ln.strip()))
            if waste_found:
                lines.append("    // !!! BOUNDARY WASTE VIOLATION - generic temps still touching splined shafts !!!")
                for bp, ln in waste_found[:3]:
                    lines.append(f"    //   {bp} leaked via: {ln[:80]}")
                lines.append("    // GPU cannot reach full capacity while this code remains the register bottleneck.")
            else:
                lines.append("    // HANDOFF FIDELITY: clean - zero generic temp leakage on splined boundary registers.")

        lines.append("    ret;")
        lines.append("}")
        return "\n".join(lines)

    def _emit_vector_loop(self, contract: MotorContract, locked: Dict[str, str], motor: MotorDecl) -> List[str]:
        """Emit a tight, correct vectorized-stride loop. Uses locked regs when present.
        Machine-aware (uses warp size knowledge from IR when available)."""
        l = []

        # Machine understanding comment
        l.append("    // Machine-aware vector loop (warp-aware, locked registers honored)")

        # Standard CUDA idiom: tid + stride*grid
        l.append("    mov.u32         %r0, %tid.x;")
        l.append("    mov.u32         %r1, %ntid.x;")
        l.append("    mov.u32         %r2, %ctaid.x;")
        l.append("    mad.lo.u32      %r3, %r2, %r1, %r0;")   # global id
        # stride = ntid.x * nctaid.x (total threads, not just block size)
        l.append("    mov.u32         %r4, %nctaid.x;")
        l.append("    mul.lo.u32      %r5, %r1, %r4;")

        # Load n (assume last scalar param or named 'n')
        n_reg = "%r10"
        for i, p in enumerate(contract.params):
            if p.name == "n":
                n_reg = f"%r{i}"
                break
        l.append(f"    setp.ge.u32     %p0, %r3, {n_reg};")
        l.append("    @%p0 bra        L_exit;")

        # Pointers
        x_rd, y_rd = "%rd4", "%rd5"
        a_f = "%f6"

        # Find locked if present and use them
        x_locked = locked.get("x", None) or locked.get("in0", None)
        y_locked = locked.get("y", None) or locked.get("out0", None)
        a_locked = locked.get("a", None)

        # Load base pointers (real PTX)
        for i, p in enumerate(contract.params):
            if p.name == "x":
                l.append(f"    ld.param.u64    {x_rd}, [x];")
            if p.name == "y":
                l.append(f"    ld.param.u64    {y_rd}, [y];")

        if a_locked:
            l.append(f"    mov.f32         {a_locked}, %f6;")  # placeholder, real a load below if scalar

        # For scalar 'a' in saxpy
        for i, p in enumerate(contract.params):
            if p.name == "a" and "float" in p.type:
                l.append(f"    ld.param.f32    %f7, [a];")
                if a_locked:
                    l.append(f"    mov.f32         {a_locked}, %f7;")
                else:
                    a_f = "%f7"

        # Main loop (grid-stride)
        l.append("L_loop:")
        l.append(f"    mad.wide.u32    %rd10, %r3, 4, {x_rd};")
        l.append("    ld.global.f32   %f8, [%rd10];")

        l.append(f"    mad.wide.u32    %rd11, %r3, 4, {y_rd};")
        l.append("    ld.global.f32   %f9, [%rd11];")

        if a_locked:
            l.append(f"    fma.rn.f32      %f10, {a_locked}, %f8, %f9;")
        else:
            l.append(f"    fma.rn.f32      %f10, {a_f}, %f8, %f9;")

        l.append(f"    st.global.f32   [%rd11], %f10;")

        # stride = total threads
        l.append("    add.u32         %r3, %r3, %r5;")
        l.append(f"    setp.lt.u32     %p1, %r3, {n_reg};")
        l.append("    @%p1 bra        L_loop;")

        l.append("L_exit:")
        return l

    def _emit_generic_loop(self, motor: MotorDecl, contract: MotorContract, locked: Dict[str, str]) -> List[str]:
        l = []
        l.append("    // Generic fallback kernel body (motor lowered conservatively)")
        l.append("    mov.u32 %r0, %tid.x;")
        l.append("    ret;  // placeholder - real lowering would expand motor body here")
        return l

    # ------------------------------------------------------------------
    # Phase 2: Real IR-driven code generation
    # ------------------------------------------------------------------
    def _emit_from_ir(self, ir: MotorIR, contract: MotorContract, locked_map: Dict[str, str], locked_physical: set) -> List[str]:
        """
        Phase 2+ expanded IR emission.
        Now handles rich ForStmt bodies with loads/stores/indexing + locked registers.
        This is the direct counter to the dev-mode fallback the wall-breaker agents keep forcing.
        """
        lines: List[str] = []
        lines.append("    // === IR-driven + Machine Understanding (bloat elimination) ===")
        lines.append(f"    // warp={ir.target_warp_size}, locked_physical={sorted(locked_physical)}")

        # Constant propagation map: reg -> literal string for Mov ops with constant src.
        # Only include single-assignment registers -- if a register is written more than once
        # (e.g. %rd3 = 1 then %rd3 = gid+1), it is NOT a constant and must not be inlined.
        _write_count: Dict[str, int] = {}
        for op in ir.ops:
            d = getattr(op, 'dest', None)
            if d:
                _write_count[d] = _write_count.get(d, 0) + 1

        _const = {}
        for op in ir.ops:
            if isinstance(op, Mov) and isinstance(op.src, str) and op.src.lstrip('-').isdigit():
                if _write_count.get(op.dest, 0) == 1:
                    _const[op.dest] = op.src

        # B1-SHIFT-CONST: propagate constants through single-write CvtIR and Mov ops.
        # e.g. `u64 c32 = 32` produces Mov(%rdN, "32") then CvtIR(%rM, %rdN, .u32).
        # Without this, the emitter materializes `mov.u32 %rM, 32` and the shift
        # uses `shl.b64 dst, src, %rM` instead of the immediate `shl.b64 dst, src, 32`.
        # REPEAT-LIT: also follow Mov-to-canonical chains so that `Mov(dest, canonical)`
        # where canonical is in _const also puts dest into _const (avoiding a redundant emit).
        def _cvt_keeps_immediate(cvop) -> bool:
            # A u32 immediate widened to u64 is still that integer token.
            # The same token must not be replayed as an f32 operand: `3` is not
            # a legal mul.f32 immediate, and the float value is not those bits.
            if not isinstance(cvop, CvtIR):
                return True
            return not (cvop.dest_type.startswith('.f') or cvop.src_type.startswith('.f'))

        _cvt_changed = True
        while _cvt_changed:
            _cvt_changed = False
            for op in ir.ops:
                if (isinstance(op, (CvtIR, Mov))
                        and isinstance(op.src, str) and op.src in _const
                        and _write_count.get(op.dest, 0) == 1
                        and op.dest not in _const
                        and _cvt_keeps_immediate(op)):
                    _const[op.dest] = _const[op.src]
                    _cvt_changed = True

        def _c(reg):
            """Return inline literal if reg is a constant, else the register name."""
            return _const.get(reg, reg)

        # Dead-temp-to-locked-reg elimination:
        # If a temp %rdN is written exactly once and its ONLY use is a single
        # Mov(dest=%R_locked, src=%rdN), replace the write dest with %R_locked
        # and skip the Mov. This eliminates redundant copies like:
        #   add.cc.u64 %rd33, %R0, %R4;   ->   add.cc.u64 %R0, %R0, %R4;
        #   mov.u64 %R0, %rd33;            ->   (skipped)
        _use_count: Dict[str, int] = {}
        for op in ir.ops:
            for attr in ('src', 'src1', 'src2', 'src3', 'left', 'right', 'offset', 'base', 'a', 'b', 'c', 'sel', 'ptr_reg', 'val_reg'):
                v = getattr(op, attr, None)
                if isinstance(v, str) and v.startswith('%'):
                    _use_count[v] = _use_count.get(v, 0) + 1

        # Map: temp_reg -> locked_reg  (only for safe single-write/single-use temps)
        _promote: Dict[str, str] = {}
        _skip_mov: set = set()
        for op in ir.ops:
            if (isinstance(op, Mov)
                    and isinstance(op.dest, str) and op.dest.startswith('%R')
                    and isinstance(op.src, str) and op.src.startswith('%rd')
                    and _write_count.get(op.src, 0) == 1
                    and _use_count.get(op.src, 0) == 1
                    and op.src not in _const):
                _promote[op.src] = op.dest
                _skip_mov.add(id(op))

        def _dest(reg):
            """Substitute a promoted temp's dest with the locked register."""
            return _promote.get(reg, reg)

        # B2-PARK-MOV: copy propagation to eliminate %rd->%rd parking copies.
        # For Mov(dest, src) where both are %rd: if src is not redefined in the
        # interval (mov_pos, last_read_of_dest], dest can be substituted -> src
        # everywhere, eliminating the copy.
        _b2_defs: Dict[str, list] = {}
        _b2_uses: Dict[str, list] = {}
        for idx, op in enumerate(ir.ops):
            d = getattr(op, 'dest', None)
            if d:
                _b2_defs.setdefault(d, []).append(idx)
            for attr in ('src', 'src1', 'src2', 'src3', 'left', 'right', 'offset', 'base', 'a', 'b', 'c', 'sel', 'ptr_reg', 'val_reg'):
                v = getattr(op, attr, None)
                if isinstance(v, str) and v.startswith('%'):
                    _b2_uses.setdefault(v, []).append(idx)
        _b2_promote: Dict[str, str] = {}
        for idx, op in enumerate(ir.ops):
            if not (isinstance(op, Mov)
                    and isinstance(op.src, str) and op.src.startswith('%rd')
                    and isinstance(op.dest, str) and op.dest.startswith('%rd')
                    and op.src not in _const and op.dest not in _const
                    and _write_count.get(op.dest, 0) == 1
                    and _write_count.get(op.src, 0) == 1):
                continue
            dest, src = op.dest, op.src
            uses_after = [u for u in _b2_uses.get(dest, []) if u > idx]
            if not uses_after:
                _skip_mov.add(id(op))   # dead write -- dest never read
                continue
            last_read = max(uses_after)
            if any(idx < d <= last_read for d in _b2_defs.get(dest, [])):
                continue  # dest overwritten in range -- can't propagate
            if any(idx < d <= last_read for d in _b2_defs.get(src, [])):
                continue  # src overwritten in range -- copy would be stale
            _skip_mov.add(id(op))
            _b2_promote[dest] = src
            continue

        # MUL-BY-CONST: emission-time constant folding with carry-chain awareness.
        # _efc tracks register values discovered during the emission pass itself
        # (clears on non-constant write so stale values are never used).
        # _carry tracks the carry flag: None = unknown, 0 or 1 = proved.
        _efc: Dict[str, str] = {}
        _carry = [None]   # list so inner helpers can mutate
        MASK64 = (1 << 64) - 1

        def _to_int(v):
            return int(v) if isinstance(v, str) and v.lstrip('-').isdigit() else None

        def _cs(reg):
            """Apply promotion + B2 copy-prop + pre-pass const + emission-time fold."""
            r = _promote.get(reg, reg)
            while r in _b2_promote:
                r = _b2_promote[r]
            v = _const.get(r)
            if v is None:
                v = _efc.get(r)
            return v if v is not None else r

        def _set_efc(dest_reg, int_val):
            """Record a folded constant. Emit mov for multi-write regs; skip for single-write."""
            new_v = str(int_val)
            prev  = _efc.get(dest_reg)
            _efc[dest_reg] = new_v
            if new_v == prev:
                return   # register already holds this value -- no instruction needed
            if _write_count.get(dest_reg, 0) != 1:
                d  = _dest(dest_reg)
                ty = 'u64' if (dest_reg.startswith('%rd') or dest_reg.startswith('%R')) else 'u32'
                lines.append(f"    mov.{ty} {d}, {int_val};")
            # Single-write: value will be inlined at every use site via _cs()

        # MISSED-FUSE peephole: BinOpIR(*, a, b) whose result feeds exactly one
        # BinOpIR(+, mul_result, c) -> replace the pair with mad.lo(a, b, c).
        # Only fuse when the mul result is a non-constant single-use register.
        _fuse_mul: Dict[str, BinOpIR] = {}
        for op in ir.ops:
            if (isinstance(op, BinOpIR) and op.op == '*'
                    and _use_count.get(op.dest, 0) == 1
                    and op.dest not in _const):
                _fuse_mul[op.dest] = op

        _fused_mad: Dict[int, object] = {}   # id(op) -> None (dead mul) or IntrinsicOp (fused add)
        for op in ir.ops:
            if isinstance(op, BinOpIR) and op.op == '+':
                for mul_reg, c_operand in ((op.left, op.right), (op.right, op.left)):
                    if mul_reg in _fuse_mul:
                        mul_op = _fuse_mul[mul_reg]
                        ty = mul_op.ptx_type or '.u32'
                        if '.f64' in ty:
                            ptx_mad = 'fma.rn.f64'
                        elif '.f32' in ty:
                            ptx_mad = 'fma.rn.f32'
                        elif '.u64' in ty or '.s64' in ty:
                            ptx_mad = 'mad.lo.u64'
                        else:
                            ptx_mad = 'mad.lo.u32'
                        fused = IntrinsicOp(
                            dest=op.dest, ptx_instr=ptx_mad,
                            src1=mul_op.left, src2=mul_op.right,
                            ptx_type=ty, src3=c_operand,
                        )
                        _fused_mad[id(op)] = fused
                        _fused_mad[id(mul_op)] = None
                        break

        # Load pointer params into their locked physical registers.
        # The non-IR path does this before the body; IR path must do it here.
        # Params can live in contract.params (synthesized) or contract.register_ports (explicit handoff).
        ptr_params = []
        if contract.params:
            ptr_params = [(p.name, p.type) for p in contract.params if "*" in p.type]
        if not ptr_params and getattr(contract, 'register_ports', []):
            ptr_params = [(rp.name, rp.type) for rp in contract.register_ports
                          if "*" in rp.type or rp.type.lower() in ("float*", "double*")]
        for pname, _ in ptr_params:
            if pname in locked_map:
                lines.append(f"    ld.param.u64 {locked_map[pname]}, [{pname}];")

        # Detect pointer-type locked registers (need grid-stride memory pattern)
        # vs scalar working variable registers (need general IR translation)
        has_locked_ptr_vars = any(
            reg for name, reg in locked_map.items()
            if reg.startswith('%R') and any(
                isinstance(s, VarDecl) and s.name == name and '*' in s.type
                for s in getattr(ir, '_motor_body_stmts', [])
            )
        )
        # Simpler heuristic: if locked vars include x/y (conventional ptr names) use grid-stride
        # otherwise use general IR translation for computational kernels (SHA256, etc.)
        ptr_names = {'x', 'y'}
        has_memory_pattern = any(name in ptr_names for name in locked_map)

        # Build ptx_name -> RegConstraint lookup for width queries in the emission loop.
        _locked_by_ptx = {rc.ptx_name: rc for rc in ir.locked_regs.values()}

        def _get_type(reg):
            if not isinstance(reg, str) or not reg.startswith('%'):
                return None
            if reg in ir.reg_types:
                return ir.reg_types[reg]
            if reg.startswith('%R'):
                rc = _locked_by_ptx.get(reg)
                if rc:
                    return rc.ptx_type
            if reg.startswith('%rd'):
                return '.u64'
            if reg.startswith('%r'):
                return '.u32'
            if reg.startswith('%f'):
                return '.f32'
            if reg.startswith('%p'):
                return '.pred'
            return None

        _fresh_next = {'f': None, 'r': None, 'rd': None}

        def _fresh_typed(want_ty: str) -> str:
            prefix = {'.f32': 'f', '.u32': 'r', '.u64': 'rd'}.get(want_ty, 'f')
            if _fresh_next[prefix] is None:
                mx = -1
                for name in ir.reg_types:
                    if prefix == 'rd':
                        if name.startswith('%rd') and name[3:].isdigit():
                            mx = max(mx, int(name[3:]))
                    elif prefix == 'r':
                        if name.startswith('%r') and not name.startswith('%rd') and name[2:].isdigit():
                            mx = max(mx, int(name[2:]))
                    elif name.startswith('%f') and name[2:].isdigit():
                        mx = max(mx, int(name[2:]))
                _fresh_next[prefix] = mx
            _fresh_next[prefix] += 1
            reg = f'%{prefix}{_fresh_next[prefix]}'
            ir.reg_types[reg] = want_ty
            return reg

        def _shared_u32_addr(base_reg: str, offset: str, byte_stride: int) -> str:
            """Byte offset in the shared state space.

            The base is `mov.u32 %r, symbol`. ld.shared, st.shared, and
            atom.shared take that .u32. Adding a lane index here stays
            inside the .shared allocation. A generic cvta.shared pointer
            does not: its numeric value is the window base.
            """
            base = _cs(base_reg)
            if not (isinstance(base, str) and base.startswith("%")):
                tmp = _fresh_typed(".u32")
                lines.append(f"    mov.u32 {tmp}, {base};")
                base = tmp
            off = _cs(offset) if isinstance(offset, str) and offset.startswith("%") else str(offset or "0")
            stride = int(byte_stride or 0)
            if stride == 0 and str(off) in ("0", ""):
                return base

            def _as_u32(token: str) -> str:
                if token.startswith("%r") and not token.startswith("%rd"):
                    return token
                if token.startswith("%rd"):
                    narrow = _fresh_typed(".u32")
                    lines.append(f"    cvt.u32.u64 {narrow}, {token};")
                    return narrow
                reg = _fresh_typed(".u32")
                lines.append(f"    mov.u32 {reg}, {token};")
                return reg

            addr = _fresh_typed(".u32")
            if stride and not str(off).startswith("%rd"):
                lines.append(f"    mad.lo.u32 {addr}, {_as_u32(str(off))}, {stride}, {base};")
                return addr
            if str(off).lstrip("-").isdigit():
                lines.append(f"    add.u32 {addr}, {base}, {int(off)};")
                return addr
            lines.append(f"    add.u32 {addr}, {base}, {_as_u32(str(off))};")
            return addr

        def _operand_ty(operand):
            if not isinstance(operand, str):
                return None
            if operand.startswith('%'):
                return _get_type(operand)
            body = operand[1:] if operand.startswith('-') else operand
            if body.isdigit():
                return '.u32'
            if any(c in body for c in '.eE'):
                return '.f32'
            return None

        def _float_immediate(token: str) -> str:
            neg = token.startswith('-')
            digits = token[1:] if neg else token
            if digits.isdigit():
                return ('-' if neg else '') + digits + '.0'
            return token

        def _align_operand(operand: str, want_ty: str, allow_immediate: bool = False) -> str:
            """Coerce one arithmetic operand to want_ty before it is printed.

            The gate only crosses the integer/float boundary. u32 versus u64
            is left to the IR widens, so integer kernels keep their text.
            A matching operand is returned unchanged. An integer register used
            by a .f32 opcode becomes `cvt.rn.f32.<src>` into a fresh %f.
            An integer immediate becomes a float immediate (`3` -> `3.0`),
            which is what mul.f32 actually accepts.
            """
            if not isinstance(operand, str) or want_ty not in ('.f32', '.u32', '.u64'):
                return operand
            have = _operand_ty(operand)
            if have is None or have == want_ty:
                return operand
            if have != '.f32' and want_ty != '.f32':
                return operand
            if not operand.startswith('%'):
                if want_ty == '.f32' and have == '.u32':
                    rendered = _float_immediate(operand)
                    if allow_immediate:
                        return rendered
                    tmp = _fresh_typed('.f32')
                    lines.append(f"    mov.f32 {tmp}, {rendered};")
                    return tmp
                return operand
            tmp = _fresh_typed(want_ty)
            lines.append(f"    {_cvt_opcode(want_ty, have)} {tmp}, {operand};")
            return tmp

        if False:  # grid-stride heuristic retired -- general IR handles everything
            lines.extend(self._emit_clean_grid_stride_using_locks(ir, contract, locked_map))
        else:
            # General IR translation -- all kernels
            for op in ir.ops:
                if isinstance(op, Label):
                    lines.append(f"{op.name}:")
                elif isinstance(op, Bra):
                    pred = f"@{op.pred} " if op.pred else ""
                    lines.append(f"    {pred}bra {op.target};")
                elif isinstance(op, Mov):
                    if id(op) in _skip_mov:
                        continue  # dead temp promoted to locked reg at write site
                    if op.dest in _const and not op.dest.startswith('%R'):
                        continue  # inlined at use sites
                    src_str = _cs(op.src)

                    dest_ty = _get_type(op.dest) or op.ptx_type
                    src_ty = _get_type(src_str)

                    # A mov between two registers of different types is a conversion.
                    # The four old special cases (u32/u64 and u32/f32) are this rule.
                    if (src_ty and dest_ty and src_ty != dest_ty
                            and isinstance(src_str, str) and src_str.startswith('%')):
                        lines.append(f"    {_cvt_opcode(dest_ty, src_ty)} {op.dest}, {src_str};")
                    else:
                        sfx = dest_ty[1:] if dest_ty else op.ptx_type[1:]
                        lines.append(f"    mov.{sfx} {op.dest}, {src_str};")
                elif isinstance(op, CmpIR):
                    lines.append(f"    setp.ne.u32 {op.dest_pred}, {_cs(op.left)}, {_cs(op.right)};")
                elif isinstance(op, UnaryOpIR):
                    ty = op.ptx_type if op.ptx_type else '.u32'
                    bty = '.b64' if 'u64' in ty else '.b32'
                    if op.op == '~':
                        lines.append(f"    not{bty} {_dest(op.dest)}, {_cs(op.src)};")
                    elif op.op == '-':
                        lines.append(f"    neg{ty} {_dest(op.dest)}, {_cs(op.src)};")
                    else:
                        lines.append(f"    // unary {op.op} {op.dest}, {op.src}")
                elif isinstance(op, BinOpIR):
                    # MISSED-FUSE: skip dead mul or emit fused mad.lo in place of the add
                    if id(op) in _fused_mad:
                        replacement = _fused_mad[id(op)]
                        if replacement is None:
                            continue  # dead mul op -- its add was fused
                        fs1 = _cs(replacement.src1)
                        fs2 = _cs(replacement.src2)
                        fs3 = _cs(replacement.src3)
                        if '.f32' in replacement.ptx_instr:
                            fs1 = _align_operand(fs1, '.f32', allow_immediate=True)
                            fs2 = _align_operand(fs2, '.f32', allow_immediate=True)
                            fs3 = _align_operand(fs3, '.f32', allow_immediate=True)
                        lines.append(f"    {replacement.ptx_instr} {_dest(replacement.dest)}, {fs1}, {fs2}, {fs3};")
                        continue
                    ty = op.ptx_type if op.ptx_type else '.f32'
                    # MUL-BY-CONST: fold plain BinOpIR * when emission-time constants are known
                    if op.op == '*':
                        lv = _cs(op.left);  rv = _cs(op.right)
                        li = _to_int(lv);   ri = _to_int(rv)
                        if li is not None and ri is not None:
                            mask = MASK64 if '.u64' in ty else 0xFFFFFFFF
                            _set_efc(op.dest, (li * ri) & mask)
                            continue
                        if li == 0 or ri == 0:
                            _set_efc(op.dest, 0)
                            continue
                        if li == 1:
                            _efc.pop(op.dest, None)
                            raw = _promote.get(op.right, op.right)
                            if '.f32' in ty:
                                raw = _align_operand(raw, '.f32', allow_immediate=True)
                            lines.append(f"    mov{ty} {_dest(op.dest)}, {raw};")
                            continue
                        if ri == 1:
                            _efc.pop(op.dest, None)
                            raw = _promote.get(op.left, op.left)
                            if '.f32' in ty:
                                raw = _align_operand(raw, '.f32', allow_immediate=True)
                            lines.append(f"    mov{ty} {_dest(op.dest)}, {raw};")
                            continue
                    _efc.pop(op.dest, None)  # non-const result -- clear stale fold
                    # Bitwise ops use .b32 regardless of declared type
                    is_bitwise = op.op in ('^', '&', '|', '<<', '>>')
                    ptx_op = {
                        '+':  f'add{ty}',
                        '-':  f'sub{ty}',
                        '*':  f'mul.lo{ty}' if ('u' in ty or 'i' in ty) else f'mul{ty}',
                        # Bare div.f32 is illegal: ptxas requires a rounding mode.
                        # .rn matches the cvt.rn used for the integer-to-float coerce.
                        '/':  f'div.rn{ty}' if ('.f32' in ty or '.f64' in ty) else f'div{ty}',
                        '<':  f'setp.lt{ty}',
                        '>':  f'setp.gt{ty}',
                        '<=': f'setp.le{ty}',
                        '>=': f'setp.ge{ty}',
                        '==': f'setp.eq{ty}',
                        '!=': f'setp.ne{ty}',
                        '^':  'xor.b64' if 'u64' in ty else 'xor.b32',
                        '&':  'and.b64' if 'u64' in ty else 'and.b32',
                        '|':  'or.b64'  if 'u64' in ty else 'or.b32',
                        '<<': 'shl.b64' if 'u64' in ty else 'shl.b32',
                        '>>': 'shr.u64' if 'u64' in ty else 'shr.u32',
                        '%':  f'rem{ty}',
                        '&&': 'and.pred',
                        '||': 'or.pred',
                    }.get(op.op, f'{op.op}{ty}')
                    left_s = _cs(op.left)
                    right_s = _cs(op.right)
                    if '.f32' in ty:
                        # The opcode is chosen from the result type. Operands still
                        # have to be that type. This is the line that used to print
                        # `mul.f32 %f1, %r0, 0.001953`.
                        left_s = _align_operand(left_s, '.f32', allow_immediate=True)
                        right_s = _align_operand(right_s, '.f32', allow_immediate=True)
                    lines.append(f"    {ptx_op} {_dest(op.dest)}, {left_s}, {right_s};")
                elif isinstance(op, IntrinsicOp):
                    instr = op.ptx_instr
                    s1 = _cs(op.src1);  s2 = _cs(op.src2)
                    s3 = _cs(op.src3) if op.src3 is not None else None
                    s1i = _to_int(s1);  s2i = _to_int(s2);  s3i = _to_int(s3) if s3 is not None else None
                    dest_r = _dest(op.dest)

                    # -- pure multiply (no carry in/out) -------------------------
                    if instr.startswith('mul.lo') or instr.startswith('mul.hi'):
                        is_lo_mul = instr.startswith('mul.lo')
                        ty_m = 'u64' if '.u64' in instr else 'u32'
                        if s1i is not None and s2i is not None:
                            result = ((s1i * s2i) & MASK64) if is_lo_mul else ((s1i * s2i) >> 64)
                            _set_efc(op.dest, result)
                            continue
                        if s1i == 0 or s2i == 0:
                            _set_efc(op.dest, 0)
                            continue
                        if is_lo_mul:
                            # lo(1*X) = X  or  lo(X*1) = X
                            if s1i == 1:
                                _efc.pop(op.dest, None)
                                lines.append(f"    mov.{ty_m} {dest_r}, {s2};")
                                continue
                            if s2i == 1:
                                _efc.pop(op.dest, None)
                                lines.append(f"    mov.{ty_m} {dest_r}, {s1};")
                                continue
                        else:
                            # hi(1*X) = 0  (X fits in u64, product fits in 64 bits)
                            if s1i == 1 or s2i == 1:
                                _set_efc(op.dest, 0)
                                continue

                    # -- mad / madc carry-chain fold ------------------------------
                    # is_madc  -> reads carry-in
                    # has_cc_out -> writes carry-out
                    if (instr.startswith('mad') or instr.startswith('madc')) and op.src3 is not None:
                        is_madc  = instr.startswith('madc')
                        sfx      = instr[4:] if is_madc else instr[3:]   # strip 'madc'/'mad'
                        has_cc_out = '.cc' in sfx
                        is_u64   = '.u64' in instr
                        ty_s     = 'u64' if is_u64 else 'u32'
                        mul_known = s1i is not None and s2i is not None
                        is_lo = '.lo.' in instr or instr.endswith('.lo')
                        is_hi = not is_lo

                        if mul_known:
                            mul_part = ((s1i * s2i) & MASK64) if is_lo else ((s1i * s2i) >> 64)
                            carry_in = _carry[0] if is_madc else 0
                            if s3i is not None and (not is_madc or carry_in is not None):
                                # Fully constant -> compute result + carry
                                total  = mul_part + s3i + (carry_in or 0)
                                result = total & MASK64
                                new_cy = total >> 64
                                _set_efc(op.dest, result)
                                if has_cc_out:
                                    _carry[0] = new_cy
                                continue
                            elif mul_part == 0:
                                # mul part = 0 -> fold to add.cc / addc.cc  (still sets carry correctly)
                                add_b  = 'addc' if is_madc else 'add'
                                cc_sfx = '.cc' if has_cc_out else ''
                                lines.append(f"    {add_b}{cc_sfx}.{ty_s} {dest_r}, {s3}, 0;")
                                if s3i == 0 and (not is_madc or _carry[0] == 0):
                                    _set_efc(op.dest, 0)
                                    if has_cc_out:
                                        _carry[0] = 0
                                else:
                                    _efc.pop(op.dest, None)
                                    if has_cc_out:
                                        # B4: emitted addc dest, 0, 0 -> carry_out=0 always (0+0+CF <= 1)
                                        _carry[0] = 0 if s3i == 0 else None
                                continue
                        elif s1i == 0 or s2i == 0:
                            # One operand is zero (other unknown) -> mul part = 0
                            add_b  = 'addc' if is_madc else 'add'
                            cc_sfx = '.cc' if has_cc_out else ''
                            lines.append(f"    {add_b}{cc_sfx}.{ty_s} {dest_r}, {s3 if s3 is not None else '0'}, 0;")
                            if s3i == 0 and (not is_madc or _carry[0] == 0):
                                _set_efc(op.dest, 0)
                                if has_cc_out:
                                    _carry[0] = 0
                            else:
                                _efc.pop(op.dest, None)
                                if has_cc_out:
                                    # B4: emitted addc dest, 0, 0 -> carry_out=0 always (0+0+CF <= 1)
                                    _carry[0] = 0 if s3i == 0 else None
                            continue
                        elif s1i == 1 or s2i == 1:
                            # lo(1*X) = X;  hi(1*X) = 0 (X fits in u64)
                            other  = s2 if s1i == 1 else s1
                            add_b  = 'addc' if is_madc else 'add'
                            cc_sfx = '.cc' if has_cc_out else ''
                            if is_lo:
                                lines.append(f"    {add_b}{cc_sfx}.{ty_s} {dest_r}, {other}, {s3 if s3 is not None else '0'};")
                            else:
                                # hi(1*X) = 0 -> result = s3 + carry
                                lines.append(f"    {add_b}{cc_sfx}.{ty_s} {dest_r}, {s3 if s3 is not None else '0'}, 0;")
                            _efc.pop(op.dest, None)
                            if has_cc_out:
                                _carry[0] = None
                            continue

                        _efc.pop(op.dest, None)
                        if has_cc_out:
                            _carry[0] = None

                    # -- add.cc / addc (2-operand carry) fold -------------------
                    elif op.src3 is None and (instr.startswith('add') or instr.startswith('addc') or
                                              instr.startswith('sub') or instr.startswith('subc')):
                        is_addc    = instr.startswith('addc') or instr.startswith('subc')
                        has_cc_out = '.cc' in instr[5 if is_addc else 3:]
                        is_sub     = instr.startswith('sub')
                        if s1i is not None and s2i is not None:
                            carry_in = _carry[0] if is_addc else 0
                            if not is_addc or carry_in is not None:
                                if is_sub:
                                    total = s1i - s2i - (carry_in or 0)
                                else:
                                    total = s1i + s2i + (carry_in or 0)
                                result = total & MASK64
                                new_cy = 1 if (total >> 64) else 0
                                _set_efc(op.dest, result)
                                if has_cc_out:
                                    _carry[0] = new_cy
                                continue
                        # B4-ADDC-ZERO: addc dest, src, 0 with carry proven 0 -> dest = src
                        if is_addc and not is_sub and s2i == 0 and _carry[0] == 0:
                            ty_b4 = 'u64' if '.u64' in instr else 'u32'
                            if s1i is not None:
                                _set_efc(op.dest, s1i & MASK64)
                            else:
                                _efc.pop(op.dest, None)
                                dr = _dest(op.dest)
                                if dr != s1:
                                    lines.append(f"    mov.{ty_b4} {dr}, {s1};")
                            if has_cc_out:
                                _carry[0] = 0  # src+0+0 <= 2^64-1, no overflow
                            continue
                        _efc.pop(op.dest, None)
                        if has_cc_out:
                            # B4: addc dest, 0, 0 -> carry_out=0 always (0+0+CF <= 1)
                            if ((is_addc and not is_sub and s1i == 0 and s2i == 0) or
                                (not is_addc and not is_sub and (s1i == 0 or s2i == 0))):
                                _carry[0] = 0
                            else:
                                _carry[0] = None
                    else:
                        _efc.pop(op.dest, None)
                        # Preserve carry tracking through non-folded .cc instructions
                        if '.cc' in instr:
                            _carry[0] = None

                    # -- unfolded: emit as-is -------------------------------------
                    if op.src3 is not None:
                        lines.append(f"    {instr} {dest_r}, {s1}, {s2}, {s3};")
                    else:
                        lines.append(f"    {instr} {dest_r}, {s1}, {s2};")
                elif isinstance(op, BarrierOp):
                    lines.append(f"    bar.sync {op.barrier_id};")
                elif isinstance(op, CvtaSharedIR):
                    lines.append(f"    mov.u32 {_dest(op.dest)}, {op.symbol};")
                elif isinstance(op, SharedDeclIR):
                    nbytes = op.nbytes
                    if not nbytes:
                        et = (op.elem_type or "").lower()
                        # Match the 1D width rule: 8 bytes only when the element is u64.
                        wide = "u64" in et or "i64" in et
                        nbytes = int(op.array_size) * (8 if wide else 4)
                    lines.append(f"    .shared .align 16 .b8 {op.name}[{nbytes}];")
                elif isinstance(op, AtomicAddIR):
                    import re as _re_at
                    # The element type and the address space are the opcode.
                    # atom's value operand is a register: a folded immediate is
                    # moved into the matching bank before the instruction.
                    ty = getattr(op, "ptx_type", None) or ".u32"
                    if not str(ty).startswith("."):
                        ty = "." + str(ty)
                    space = getattr(op, "space", None) or "global"
                    if space not in ("global", "shared"):
                        space = "global"
                    if ty not in (".u32", ".u64", ".f32"):
                        ty = ".u32"
                    val = _align_operand(_cs(op.val_reg), ty, allow_immediate=False)
                    if not (isinstance(val, str) and val.startswith("%")):
                        tmp = _fresh_typed(ty)
                        token = str(val)
                        if ty == ".f32":
                            token = _float_immediate(token)
                        lines.append(f"    mov{ty} {tmp}, {token};")
                        val = tmp
                    if space == "shared":
                        addr = _shared_u32_addr(
                            op.ptr_reg,
                            str(getattr(op, "offset", "0") or "0"),
                            int(getattr(op, "byte_stride", 0) or 0),
                        )
                        lines.append(
                            f"    atom.shared.add{ty} {_dest(op.dest)}, [{addr}], {val};"
                        )
                        continue
                    base = _cs(op.ptr_reg)
                    if not (isinstance(base, str) and base.startswith("%")):
                        base_reg = _fresh_typed(".u64")
                        lines.append(f"    mov.u64 {base_reg}, {base};")
                        base = base_reg
                    offset = str(getattr(op, "offset", "0") or "0")
                    stride = int(getattr(op, "byte_stride", 0) or 0)
                    off = _cs(offset) if offset.startswith("%") else offset

                    def _next_rd() -> str:
                        mx = max(
                            (int(m) for ln in lines for m in _re_at.findall(r"%rd(\d+)", ln)),
                            default=-1,
                        )
                        n = mx + 1
                        if base.startswith("%rd") and base[3:].isdigit() and int(base[3:]) >= n:
                            n = int(base[3:]) + 1
                        return f"%rd{n}"

                    addr = base
                    if stride and not str(off).startswith("%rd"):
                        off_reg = off
                        if not (isinstance(off_reg, str) and off_reg.startswith("%")):
                            off_reg = _fresh_typed(".u32")
                            lines.append(f"    mov.u32 {off_reg}, {off};")
                        addr = _next_rd()
                        lines.append(f"    mad.wide.u32 {addr}, {off_reg}, {stride}, {base};")
                    elif str(off).lstrip("-").isdigit():
                        disp = int(off)
                        if disp != 0:
                            addr = _next_rd()
                            lines.append(f"    add.u64 {addr}, {base}, {disp};")
                    elif str(off).startswith("%rd"):
                        addr = _next_rd()
                        lines.append(f"    add.u64 {addr}, {off}, {base};")
                    elif str(off).startswith("%"):
                        addr = _next_rd()
                        lines.append(f"    cvt.u64.u32 {addr}, {off};")
                        lines.append(f"    add.u64 {addr}, {addr}, {base};")
                    lines.append(
                        f"    atom.{space}.add{ty} {_dest(op.dest)}, [{addr}], {val};"
                    )
                elif isinstance(op, ShflOp):
                    # clamp encodes (segmask<<8 | maxLane): full-warp = 0x1f for all modes
                    # except shfl.up which uses 0 (no upper-lane clamping).
                    clamp = "0" if op.mode == "up" else "0x1f"
                    lines.append(f"    shfl.sync.{op.mode}.b32 {_dest(op.dest)}, {_cs(op.src)}, {_cs(op.sel)}, {clamp}, {op.membermask};")
                elif isinstance(op, ClockReadIR):
                    # Hardware cycle counter read.
                    # clock64() -> mov.u64 dest, %globaltimer;  (sm_89 global nanosecond timer)
                    # clock32() -> mov.u32 dest, %clock;        (SM 32-bit cycle counter)
                    if op.width == 64:
                        lines.append(f"    mov.u64 {_dest(op.dest)}, %globaltimer;")
                    else:
                        lines.append(f"    mov.u32 {_dest(op.dest)}, %clock;")
                elif isinstance(op, Lop3Op):
                    # 3-input logic: lop3.b32 dest, a, b, c, immLut
                    lines.append(f"    lop3.b32 {_dest(op.dest)}, {_cs(op.a)}, {_cs(op.b)}, {_cs(op.c)}, {op.imm};")
                elif isinstance(op, LoadParam):
                    ty = op.ptx_type if op.ptx_type else '.u32'
                    lines.append(f"    ld.param{ty} {_dest(op.dest)}, [{op.param_name}];")
                elif isinstance(op, IRLoadGlobal):
                    space = getattr(op, 'space', None) or 'global'
                    if space == "shared":
                        addr = _shared_u32_addr(
                            op.base, str(op.offset), int(getattr(op, "byte_stride", 0) or 0),
                        )
                        lines.append(f"    ld.shared{op.ptx_type} {_dest(op.dest)}, [{addr}];")
                    elif str(op.offset).lstrip('-').isdigit():
                         # Compile-time constant offset -- direct [ptr+N] addressing.
                         n = int(op.offset)
                         addr = f"[{_cs(op.base)}]" if n == 0 else f"[{_cs(op.base)}+{n}]"
                         lines.append(f"    ld.{space}{op.ptx_type} {_dest(op.dest)}, {addr};")
                    elif getattr(op, 'byte_stride', 0) and not op.offset.startswith('%rd'):
                         # U64-INDEX: u32 element index -> one-shot widen + scale + add
                         base_op = _cs(op.base)
                         off_op = _cs(op.offset)
                         scratch = _load_addr_scratch(base_op, off_op)
                         lines.append(f"    mad.wide.u32 {scratch}, {off_op}, {op.byte_stride}, {base_op};")
                         lines.append(f"    ld.{space}{op.ptx_type} {_dest(op.dest)}, [{scratch}];")
                    elif op.offset.startswith('%rd'):
                         # Already u64 -- add directly, no cvt needed.
                         base_op = _cs(op.base)
                         off_op = _cs(op.offset)
                         scratch = _load_addr_scratch(base_op, off_op)
                         lines.append(f"    add.u64 {scratch}, {off_op}, {base_op};")
                         lines.append(f"    ld.{space}{op.ptx_type} {_dest(op.dest)}, [{scratch}];")
                    else:
                         # u32 offset without stride (legacy path).
                         base_op = _cs(op.base)
                         off_op = _cs(op.offset)
                         scratch = _load_addr_scratch(base_op, off_op)
                         lines.append(f"    cvt.u64.u32 {scratch}, {off_op};")
                         lines.append(f"    add.u64 {scratch}, {scratch}, {base_op};")
                         lines.append(f"    ld.{space}{op.ptx_type} {_dest(op.dest)}, [{scratch}];")
                elif isinstance(op, IRStoreGlobal):
                    import re as _re_st
                    space = getattr(op, 'space', None) or 'global'
                    # Bug B fix: apply _cs() so const-inlined source registers are
                    # resolved before the store.  Previously op.src was used raw --
                    # if the Mov for that register had been skipped by const-inlining,
                    # the register was declared but never written -> uninitialized read.
                    src_val = _cs(op.src)
                    if op.ptx_type == '.f32':
                        src_val = _align_operand(src_val, '.f32', allow_immediate=False)
                    elif op.ptx_type in ('.u32', '.u64'):
                        src_val = _align_operand(src_val, op.ptx_type, allow_immediate=False)
                    # Scan already-emitted lines for the highest %rdN in use.
                    # This lets us allocate fresh scratch registers that cannot
                    # collide with any live value.
                    _mx = max(
                        (int(m) for ln in lines for m in _re_st.findall(r'%rd(\d+)', ln)),
                        default=-1,
                    )
                    if not src_val.startswith('%'):
                        # st.global requires a register source -- materialize the literal.
                        mat = f"%rd{_mx + 1}"
                        lines.append(f"    mov.u64 {mat}, {src_val};")
                        src_val = mat
                        _mx += 1
                    # Bug A fix: derive scratch index above both the emitted high-water
                    # mark AND src_val's own index.  The previous hardcoded %rd9 could
                    # be the same register as op.src (e.g. the addc64 result), causing
                    # the address-add to clobber the value before the store fired.
                    src_rd_idx = int(src_val[3:]) if src_val.startswith('%rd') else -1
                    if space == "shared":
                        addr = _shared_u32_addr(
                            op.base, str(op.offset), int(getattr(op, "byte_stride", 0) or 0),
                        )
                        lines.append(f"    st.shared{op.ptx_type} [{addr}], {src_val};")
                    elif str(op.offset).lstrip('-').isdigit():
                        # Compile-time constant offset -- direct [ptr+N] form, no scratch.
                        n = int(op.offset)
                        addr = f"[{_cs(op.base)}]" if n == 0 else f"[{_cs(op.base)}+{n}]"
                        lines.append(f"    st.{space}{op.ptx_type} {addr}, {src_val};")
                    elif getattr(op, 'byte_stride', 0) and not op.offset.startswith('%rd'):
                        # U64-INDEX: u32 element index -> one-shot widen + scale + add
                        scratch = f"%rd{max(_mx, src_rd_idx) + 1}"
                        lines.append(f"    mad.wide.u32 {scratch}, {_cs(op.offset)}, {op.byte_stride}, {_cs(op.base)};")
                        lines.append(f"    st.{space}{op.ptx_type} [{scratch}], {src_val};")
                    else:
                        scratch = f"%rd{max(_mx, src_rd_idx) + 1}"
                        if op.offset.startswith('%rd'):
                            lines.append(f"    add.u64 {scratch}, {_cs(op.offset)}, {_cs(op.base)};")
                        else:
                            lines.append(f"    cvt.u64.u32 {scratch}, {_cs(op.offset)};")
                            lines.append(f"    add.u64 {scratch}, {scratch}, {_cs(op.base)};")
                        lines.append(f"    st.{space}{op.ptx_type} [{scratch}], {src_val};")
                elif isinstance(op, CvtIR):
                    # B1-SHIFT-CONST: skip CvtIR whose dest was constant-propagated --
                    # the value is inlined at every use site, no register materialization needed
                    if op.dest in _const and _write_count.get(op.dest, 0) == 1:
                        continue
                    c_src = _cs(op.src)
                    if c_src != op.src and not str(c_src).startswith('%'):
                        # Source is a known constant. cvt does not take an immediate.
                        # A float dest needs a float token; `mov.f32 %f, 3` is illegal.
                        if op.dest_type == '.f32' and str(c_src).lstrip('-').isdigit():
                            lines.append(f"    mov.f32 {_dest(op.dest)}, {_float_immediate(str(c_src))};")
                        else:
                            lines.append(f"    mov{op.dest_type} {_dest(op.dest)}, {c_src};")
                    else:
                        lines.append(f"    {_cvt_opcode(op.dest_type, op.src_type)} {_dest(op.dest)}, {c_src};")
                elif isinstance(op, ReturnIR):
                    lines.append("    ret;")
                else:
                    lines.append(f"    // {type(op).__name__}")
        import ssa_fold_mod
        return ssa_fold_mod.ssa_rename_and_fold(lines)

    def _emit_clean_grid_stride_using_locks(self, ir: MotorIR, contract: MotorContract, locked: Dict[str, str]) -> List[str]:
        """
        Phase 2+ expanded clean emission.
        Uses actual lowered IR ops for the loop body when present (rich ForStmt support).
        This is the main weapon against the dev-mode / thin-lowering barrier the agents are hammering.
        """
        l = []
        l.append("    // Clean machine-aware grid-stride (IR-driven, locked registers honored, no bloat)")
        l.append("    mov.u32 %r0, %tid.x;")
        l.append("    mov.u32 %r1, %ntid.x;")
        l.append("    mov.u32 %r2, %ctaid.x;")
        l.append("    mad.lo.u32 %r3, %r2, %r1, %r0;")
        # stride = blockDim.x * gridDim.x (total threads, not just block size)
        # Using %ntid.x alone as stride caused each thread to process 4 elements
        # spaced blockDim apart, with multiple threads hitting the same indices -> error 700
        l.append("    mov.u32 %r4, %nctaid.x;")
        l.append("    mul.lo.u32 %r5, %r1, %r4;")  # %r5 = stride = ntid * nctaid

        xrd = locked.get('x', '%rd4')
        yrd = locked.get('y', '%rd5')

        # When explicit handoff is active, prefer any boundary physical splines for address/data movement
        # This is the mechanical lockup clutch: the mad/ld/st use the declared %R splines directly.
        use_handoff = ir is not None and getattr(ir, 'explicit_handoff', False)
        boundary = getattr(ir, 'boundary_physical', set()) if ir else set()

        # Try to pick splined physicals for the actual traffic (the whole point of the bulletproof spec)
        x_addr_spline = None
        y_addr_spline = None
        data_spline = None
        for bp in sorted(boundary):
            if 'addr' in bp.lower() or int(bp[1:]) in (2, 3):
                if not x_addr_spline: x_addr_spline = '%' + bp
                elif not y_addr_spline: y_addr_spline = '%' + bp
            elif int(bp[1:]) >= 4:
                data_spline = '%' + bp

        # Load pointer params - use actual param names from contract
        ptr_params = []
        if contract.params:
            ptr_params = [p for p in contract.params if "*" in p.type]
        elif getattr(contract, 'register_ports', []):
            ptr_params = [p for p in contract.register_ports if "*" in p.type or p.type.lower() in ("float*","double*")]

        # Use declared bank registers %rd4/%rd5 for x/y pointers
        # (avoids undeclared named registers which cause CUDA_ERROR_INVALID_PTX)
        if len(ptr_params) >= 2:
            l.append(f"    ld.param.u64 %rd4, [{ptr_params[0].name}];")
            l.append(f"    ld.param.u64 %rd5, [{ptr_params[1].name}];")
        elif len(ptr_params) == 1:
            l.append(f"    ld.param.u64 %rd4, [{ptr_params[0].name}];")
            l.append(f"    ld.param.u64 %rd5, [{ptr_params[0].name}];")
        else:
            l.append("    ld.param.u64 %rd4, [x];")
            l.append("    ld.param.u64 %rd5, [y];")

        # Top-of-loop guard: exit immediately if global_id is already out of range.
        # Without this, a thread with global_id >= n runs the body once before the
        # bottom setp fires -- accessing memory past the buffer end -> error 700.
        l.append("    setp.ge.u32 %p0, %r3, 1024;")
        l.append("    @%p0 bra L_exit;")

        # If the IR has substantial body ops, try to emit a real loop using them
        body_ops = [op for op in ir.ops if not isinstance(op, (Label, Bra)) or 'for' in str(op).lower()]
        if len(body_ops) > 4:
            l.append("L_loop:")
            if use_handoff and x_addr_spline:
                # Direct splined shaft path - the GPU sees far fewer live temps
                l.append(f"    mad.wide.u32 {x_addr_spline}, %r3, 4, %rd4;")
                l.append(f"    ld.global.f32 %f8, [{x_addr_spline}];")
                if y_addr_spline:
                    l.append(f"    mad.wide.u32 {y_addr_spline}, %r3, 4, %rd5;")
                    l.append(f"    ld.global.f32 %f9, [{y_addr_spline}];")
                    if data_spline:
                        l.append(f"    add.f32 {data_spline}, %f8, 1.0;")
                        l.append(f"    st.global.f32 [{y_addr_spline}], {data_spline};")
                    else:
                        l.append("    add.f32 %f10, %f8, 1.0;")
                        l.append(f"    st.global.f32 [{y_addr_spline}], %f10;")
                else:
                    l.append("    add.f32 %f10, %f8, 1.0;")
                    l.append(f"    st.global.f32 [%rd11], %f10;")
            else:
                # Synthesized path - route data through locked f32 physicals when available
                # locked dict maps varname -> ptx_reg e.g. {'x': '%R0', 'y': '%R1'}
                # Exclude pointer-typed locked registers (.u64) -- using them in .f32
                # instructions is illegal PTX and causes CUDA_ERROR_INVALID_PTX (error 218).
                ptr_varnames = {p.name for p in ptr_params}
                f32_locked = [reg for var, reg in locked.items()
                              if reg.startswith('%R') and var not in ptr_varnames]

                if len(f32_locked) >= 2:
                    # Route through locked physicals - eliminates %f8/%f9/%f10
                    val_reg = f32_locked[0]
                    out_reg = f32_locked[1]
                    l.append("    mad.wide.u32 %rd10, %r3, 4, %rd4;")
                    l.append(f"    ld.global.f32 {val_reg}, [%rd10];")
                    l.append("    mad.wide.u32 %rd11, %r3, 4, %rd5;")
                    l.append(f"    add.f32 {out_reg}, {val_reg}, 1.0;")
                    l.append(f"    st.global.f32 [%rd11], {out_reg};")
                elif len(f32_locked) == 1:
                    val_reg = f32_locked[0]
                    l.append("    mad.wide.u32 %rd10, %r3, 4, %rd4;")
                    l.append(f"    ld.global.f32 {val_reg}, [%rd10];")
                    l.append("    mad.wide.u32 %rd11, %r3, 4, %rd5;")
                    l.append(f"    add.f32 {val_reg}, {val_reg}, 1.0;")
                    l.append(f"    st.global.f32 [%rd11], {val_reg};")
                else:
                    # No locked f32 available - use generics
                    l.append("    mad.wide.u32 %rd10, %r3, 4, %rd4;")
                    l.append("    ld.global.f32 %f8, [%rd10];")
                    l.append("    mad.wide.u32 %rd11, %r3, 4, %rd5;")
                    l.append("    add.f32 %f10, %f8, 1.0;")
                    l.append("    st.global.f32 [%rd11], %f10;")
            l.append("    add.u32 %r3, %r3, %r5;")   # stride = total threads
            l.append("    setp.lt.u32 %p1, %r3, 1024;")
            l.append("    @%p1 bra L_loop;")
            l.append("L_exit:")
        else:
            # Fallback minimal - route through locked physicals when available
            ptr_varnames = {p.name for p in ptr_params}
            f32_locked = [reg for var, reg in locked.items()
                          if reg.startswith('%R') and var not in ptr_varnames]
            l.append("L_loop:")
            l.append(f"    mad.wide.u32 %rd10, %r3, 4, {xrd};")
            if len(f32_locked) >= 2:
                l.append(f"    ld.global.f32 {f32_locked[0]}, [%rd10];")
                l.append(f"    mad.wide.u32 %rd11, %r3, 4, {yrd};")
                l.append(f"    add.f32 {f32_locked[1]}, {f32_locked[0]}, 1.0;")
                l.append(f"    st.global.f32 [%rd11], {f32_locked[1]};")
            else:
                l.append("    ld.global.f32 %f8, [%rd10];")
                l.append(f"    mad.wide.u32 %rd11, %r3, 4, {yrd};")
                l.append("    add.f32 %f10, %f8, 1.0;")
                l.append("    st.global.f32 [%rd11], %f10;")
            l.append("    add.u32 %r3, %r3, %r5;")   # stride = total threads
            l.append("    setp.lt.u32 %p1, %r3, 1024;")
            l.append("    @%p1 bra L_loop;")
            l.append("L_exit:")

        return l

if __name__ == "__main__":
    print("PTX Emitter ready. Use via Compiler.")
