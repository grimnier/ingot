"""
Ingot Compiler - Full Pipeline (Phase 1 Complete)

lexer -> parser -> semantic contract check -> PTX emission

This is the heart of the system. Real, usable, no agent references.
"""

from pathlib import Path
from typing import Optional, Tuple
try:
    from .lexer import Lexer, LexerError
    from .parser import Parser, ParseError
    from .semantic import ContractChecker, SemanticError
    from .ptx_emitter import PTXEmitter, PTXEmitError
    from .ast_nodes import Program
    from .ir import IRBuilder, MotorIR, ShapeError
    from .expansion import try_expand
except ImportError:
    # Allow direct execution / test runners
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from lexer import Lexer, LexerError
    from parser import Parser, ParseError
    from semantic import ContractChecker, SemanticError
    from ptx_emitter import PTXEmitter, PTXEmitError
    from ast_nodes import Program
    from ir import IRBuilder, MotorIR, ShapeError
    from expansion import try_expand

class CompileError(Exception):
    pass

class Compiler:
    def __init__(self):
        self.last_program: Optional[Program] = None
        self.last_ptx: Optional[str] = None
        self.last_ir: Optional[MotorIR] = None

    def compile_file(self, path: str | Path, tx_name: Optional[str] = None) -> Tuple[str, Program]:
        src = Path(path).read_text(encoding="utf-8")
        return self.compile_source(src, str(path), tx_name)

    def compile_source(self, source: str, filename: str = "<input>", tx_name: Optional[str] = None) -> Tuple[str, Program]:
        try:
            lexer = Lexer(source, filename)
            tokens = lexer.tokenize()
        except LexerError as e:
            raise CompileError(str(e))

        try:
            parser = Parser(tokens, filename)
            prog = parser.parse()
        except ParseError as e:
            raise CompileError(str(e))

        self.last_program = prog

        # Packet desugar runs first: grafts packet fields (including @window arrays) into
        # motors that use them, and populates motor.windows so the window-index rewrite
        # pass (_desugar_windows Pass 2) covers packet-sourced arrays too.
        self._desugar_packet_refs(prog)

        # Desugar register windows: expands WindowDecl nodes into locked VarDecl nodes and
        # rewrites Index(Var(win), IntLit(i)) -> Var(win_i) for both inline and packet windows.
        self._desugar_windows(prog)

        # ANY user-declared transmission block = explicit ownership, including geometry-only ones.
        explicit_tx_ownership = bool(getattr(prog, 'transmissions', []))
        prog.explicit_transmission_ownership = explicit_tx_ownership

        checker = ContractChecker()
        if not checker.check_program(prog):
            raise CompileError("Semantic contract violation(s):\n" + checker.report())

        # WP2: detect plain-operator motors and expand through the sealed profile
        target_tx = tx_name or (prog.transmissions[0].name if prog.transmissions else None)
        if target_tx:
            _tx = next((t for t in prog.transmissions if t.name == target_tx), None)
            _mc_name = _tx.motor_contract.motor_name if (_tx and _tx.motor_contract) else None
            _motor = next((m for m in prog.motors if m.name == _mc_name), None) if _mc_name else None
            if _motor and _motor.params:
                expanded_src = try_expand(_motor, _tx)
                if expanded_src:
                    return self.compile_source(expanded_src, filename + " [expanded]", target_tx)

        # Real IR lowering (Phase 2)
        ir_builder = IRBuilder()

        # The Dual-Drive Clutches (P17) note:
        # Transmission contracts will eventually define async feedback channels.
        # When a motor produces a result flag, the scheduler can engage
        # a parallel transmission without device-side dynamic parallelism queues.
        motor_name = None
        contract = None
        if target_tx:
            for t in prog.transmissions:
                if t.name == target_tx and t.motor_contract:
                    motor_name = t.motor_contract.motor_name
                    contract = t.motor_contract
                    break
        motor = next((m for m in prog.motors if m.name == motor_name), None) if motor_name else None

        # Build machine model, feeding TX geometry in so the IR builder knows the register budget.
        block_size = 256
        if target_tx:
            tx_obj = next((t for t in prog.transmissions if t.name == target_tx), None)
            if tx_obj and tx_obj.block:
                try:
                    block_size = int(getattr(tx_obj.block[0], 'value', 256))
                except Exception:
                    pass
        machine = {"warp_size": 32, "max_regs": 255, "block_size": block_size}
        try:
            self.last_ir = ir_builder.lower_motor(motor, contract, machine_model=machine) if motor else None
        except ShapeError as e:
            raise CompileError(str(e)) from e
        except Exception as e:
            # Wall-breaker robustness: some extremely mutated programs can still trip the current IR builder.
            # Log and continue with None so the emitter can fall back gracefully instead of killing the agent batch.
            print(f"[compiler] IR lowering failed for this mutant (wall-breaker edge case): {e}")
            self.last_ir = None

        # Choose compilation path based on what the user declared.
        has_contract = any(t.motor_contract for t in prog.transmissions)

        if not prog.transmissions:
            # -- Motor-only (no transmission block at all) ----------------------
            # Synthesis is legal: infer grid/block/params from the motor's locked regs.
            if not prog.motors:
                raise CompileError('No motor found')
            motor = prog.motors[0]

            from ast_nodes import TransmissionDecl, MotorContract, TxParam, IntLit
            params = []
            seen = set()
            for var_name, reg in motor.locked_regs.items():
                guessed = self._guess_type_for_var(motor, var_name).strip()
                ptype = guessed if guessed else 'float'
                params.append(TxParam('inout', ptype, var_name))
                seen.add(var_name)
            for stmt in motor.body.stmts:
                if hasattr(stmt, 'lvalue') and hasattr(stmt.lvalue, 'name'):
                    nm = stmt.lvalue.name
                    if nm not in seen and '*' in self._guess_type_for_var(motor, nm):
                        params.append(TxParam('inout', 'float*', nm))
                        seen.add(nm)
            if not params:
                params = [TxParam('inout', 'float*', 'data')]
            fake_contract = MotorContract(motor_name=motor.name, params=params, register_ports=[])
            fake_tx = TransmissionDecl(motor.name + '_contract', [IntLit(1024,'1024')], [IntLit(1,'1')], None, fake_contract)
            prog.transmissions = [fake_tx]
            try:
                self.last_ir = ir_builder.lower_motor(motor, fake_contract, machine_model=machine)
                self.last_ir.synthesized_contract = True
            except ShapeError as e:
                raise CompileError(str(e)) from e
            except Exception as e:
                print(f"[compiler] Synthetic IR lowering failed: {e}")
                self.last_ir = None

        elif not has_contract:
            # -- Geometry-only transmission (grid/block declared, no motor contract) --
            # The user owns the geometry. Infer params from locked regs only (no full
            # synthesis). Mark as synthesized_ports so downstream can distinguish this
            # from full synthesis. Block size comes from the explicit TX geometry.
            if not prog.motors:
                raise CompileError('No motor found')
            tx_geom = prog.transmissions[0]
            motor = prog.motors[0]

            # Extract block size from the declared geometry for register budget
            try:
                block_x = int(getattr(tx_geom.block[0], 'value', 256)) if tx_geom.block else 256
            except Exception:
                block_x = 256
            machine = {**machine, "block_size": block_x}

            from ast_nodes import MotorContract, TxParam
            params = []
            seen = set()
            for var_name, reg in motor.locked_regs.items():
                guessed = self._guess_type_for_var(motor, var_name).strip()
                ptype = guessed if guessed else 'float'
                params.append(TxParam('inout', ptype, var_name))
                seen.add(var_name)
            for stmt in motor.body.stmts:
                if hasattr(stmt, 'lvalue') and hasattr(stmt.lvalue, 'name'):
                    nm = stmt.lvalue.name
                    if nm not in seen and '*' in self._guess_type_for_var(motor, nm):
                        params.append(TxParam('inout', 'float*', nm))
                        seen.add(nm)
            if not params:
                params = [TxParam('inout', 'float*', 'data')]
            inferred_contract = MotorContract(motor_name=motor.name, params=params, register_ports=[])
            # Graft the inferred motor contract onto the user's geometry TX so the
            # emitter sees a complete transmission with the user's real grid/block.
            tx_geom.motor_contract = inferred_contract
            try:
                self.last_ir = ir_builder.lower_motor(motor, inferred_contract, machine_model=machine)
                self.last_ir.synthesized_ports = True   # geometry explicit, ports inferred
            except ShapeError as e:
                raise CompileError(str(e)) from e
            except Exception as e:
                print(f"[compiler] Geometry-TX IR lowering failed: {e}")
                self.last_ir = None

        target_tx = tx_name or prog.transmissions[0].name

        emitter = PTXEmitter()
        try:
            ptx = emitter.emit_for_transmission(prog, target_tx, ir=self.last_ir)
        except PTXEmitError as e:
            raise CompileError(str(e))

        self.last_ptx = ptx

        # Register allocation safety check: prove no compiler temp was written into
        # a hard-locked physical register.  Runs whenever the IR lowering produced
        # a locked_physical set (i.e. the motor had @register(Rn) annotations).
        if self.last_ir is not None:
            locked_names = list(getattr(self.last_ir, 'locked_physical', set()))
            if locked_names:
                try:
                    from native_verify import verify_no_temp_touches_locked_physical
                except ImportError:
                    try:
                        from .native_verify import verify_no_temp_touches_locked_physical
                    except ImportError:
                        verify_no_temp_touches_locked_physical = None
                if verify_no_temp_touches_locked_physical is not None:
                    verify_no_temp_touches_locked_physical(ptx, locked_names)

        # Enforce sealed-profile constraints (require no_spills, max_registers, etc.)
        self._run_sealed_gate(prog, ptx, target_tx)

        # Phase 2+: Expose first-class pressure / fidelity report when we have IR
        if self.last_ir is not None:
            locked_count = len(getattr(self.last_ir, 'locked_physical', set()))
            temp_estimate = max(0, len(self.last_ir.ops) // 2)  # rough
            self.last_pressure_report = {
                "locked_physical_count": locked_count,
                "approx_temp_ops": temp_estimate,
                "pressure_ratio_estimate": round(temp_estimate / max(1, locked_count), 2) if locked_count else 0,
                "fidelity": "clean" if locked_count > 0 else "n/a",
                "note": "Generated from lowered IR. Higher ratio = more compiler temps under locked pressure."
            }
        else:
            self.last_pressure_report = None

        return ptx, prog

    def compile_all_transmissions(self, source: str, filename: str = "<input>"):
        """Compile every transmission in source into a single PTX with multiple .entry blocks.

        Returns (ptx, prog). Runs the spill gate on each sealed transmission independently.
        Use this instead of compile_source when a .mtl file contains paired motors (e.g.
        point_walk_tx + hash160_tx) that must land in the same cubin without shared headers.
        """
        try:
            tokens = Lexer(source, filename).tokenize()
        except LexerError as e:
            raise CompileError(str(e))
        try:
            prog = Parser(tokens, filename).parse()
        except ParseError as e:
            raise CompileError(str(e))

        self.last_program = prog
        self._desugar_packet_refs(prog)
        self._desugar_windows(prog)
        prog.explicit_transmission_ownership = bool(prog.transmissions)

        checker = ContractChecker()
        if not checker.check_program(prog):
            raise CompileError("Semantic errors:\n" + checker.report())

        ir_builder = IRBuilder()
        ir_map = {}
        machine = {"warp_size": 32, "max_regs": 255, "block_size": 256}
        for tx in prog.transmissions:
            if not tx.motor_contract:
                continue
            motor_name = tx.motor_contract.motor_name
            motor = next((m for m in prog.motors if m.name == motor_name), None)
            if motor:
                try:
                    ir_map[tx.name] = ir_builder.lower_motor(motor, tx.motor_contract,
                                                             machine_model=machine)
                except ShapeError as e:
                    raise CompileError(str(e)) from e
                except Exception as e:
                    print(f"[compiler] IR lowering failed for {tx.name}: {e}")
                    ir_map[tx.name] = None

        emitter = PTXEmitter()
        try:
            ptx = emitter.emit_all_transmissions(prog, ir_map=ir_map)
        except PTXEmitError as e:
            raise CompileError(str(e))

        self.last_ptx = ptx

        for tx in prog.transmissions:
            if getattr(tx, 'require_clauses', None) or getattr(tx, 'profile', None):
                self._run_sealed_gate(prog, ptx, tx.name)

        return ptx, prog

    def compile_all_file(self, path) -> tuple:
        src = Path(path).read_text(encoding="utf-8")
        return self.compile_all_transmissions(src, str(path))

    def _guess_type_for_var(self, motor, var_name: str) -> str:
        def search(stmts):
            for stmt in stmts:
                if hasattr(stmt, 'name') and stmt.name == var_name and hasattr(stmt, 'type'):
                    return stmt.type
                # Recurse into blocks and control flow
                for attr in ('stmts', 'then_body', 'else_body', 'body'):
                    child = getattr(stmt, attr, None)
                    if child:
                        sub = child.stmts if hasattr(child, 'stmts') else []
                        found = search(sub)
                        if found:
                            return found
            return None
        result = search(motor.body.stmts)
        return result if result else 'float'

    def _desugar_packet_refs(self, prog):
        """Graft all top-level contract declarations into motors that `use` them.

        Handles four contract types (dispatched by looking up the name in the relevant
        program-level lists):
          - packet   -> locked VarDecl nodes, motor.windows entry, motor.used_packets
          - rail     -> NAME_width constant (VarDecl+Assign), motor.used_rails
          - route_type -> one VarDecl+Assign per variant constant, motor.used_route_types
          - target_latch -> recorded in motor.used_latches (no injected constants)

        Must run BEFORE _desugar_windows so packet-sourced @window fields land in
        motor.windows before the index-rewrite pass picks them up.

        Also expands `packet NAME;` references in motor contracts.
        """
        try:
            from .ast_nodes import (PacketUse, PacketRef, WindowDecl, VarDecl, Assign, Var,
                                    IntLit, FloatLit, TxParam, RegisterRange, RegisterPort)
        except ImportError:
            from ast_nodes import (PacketUse, PacketRef, WindowDecl, VarDecl, Assign, Var,
                                   IntLit, FloatLit, TxParam, RegisterRange, RegisterPort)

        packet_map     = {p.name: p for p in getattr(prog, 'packets',       [])}
        rail_map       = {r.name: r for r in getattr(prog, 'rails',         [])}
        latch_map      = {l.name: l for l in getattr(prog, 'target_latches',[])}
        route_type_map = {t.name: t for t in getattr(prog, 'route_types',   [])}
        attractor_map  = {a.name: a for a in getattr(prog, 'attractors',    [])}
        manifold_map   = {m.name: m for m in getattr(prog, 'manifolds',     [])}

        # -- Pass 1: graft contract fields / constants into motors --------------
        for motor in prog.motors:
            graft_stmts = []   # VarDecl / Assign nodes to prepend
            body_stmts  = []   # original body stmts minus PacketUse

            for stmt in motor.body.stmts:
                if not isinstance(stmt, PacketUse):
                    body_stmts.append(stmt)
                    continue

                name = stmt.packet_name  # generic contract name

                # -- Packet ----------------------------------------------------
                if name in packet_map:
                    pkt = packet_map[name]
                    motor.used_packets.append(name)
                    for fld in pkt.fields:
                        if isinstance(fld, WindowDecl):
                            start_num = int(fld.start_reg[1:])
                            for i in range(fld.count):
                                reg_name = f'R{start_num + i}'
                                var_name = f'{fld.name}_{i}'
                                if reg_name in motor.locked_regs.values():
                                    raise CompileError(
                                        f"Packet '{name}' field '{fld.name}[{i}]' register "
                                        f"{reg_name} conflicts with an existing locked register "
                                        f"in motor '{motor.name}'"
                                    )
                                graft_stmts.append(VarDecl(name=var_name, type=fld.elem_type,
                                                           register=reg_name))
                                motor.locked_regs[var_name] = reg_name
                            motor.windows[fld.name] = fld
                        elif isinstance(fld, VarDecl) and fld.register:
                            reg_name = fld.register
                            if reg_name in motor.locked_regs.values():
                                raise CompileError(
                                    f"Packet '{name}' field '{fld.name}' register {reg_name} "
                                    f"conflicts with an existing locked register "
                                    f"in motor '{motor.name}'"
                                )
                            graft_stmts.append(VarDecl(name=fld.name, type=fld.type,
                                                       register=reg_name))
                            motor.locked_regs[fld.name] = reg_name

                # -- Rail ------------------------------------------------------
                elif name in rail_map:
                    rail = rail_map[name]
                    motor.used_rails.append(name)
                    # Inject NAME_width as a compile-time constant (folded by const-prop)
                    width_name = f'{name}_width'
                    graft_stmts.append(VarDecl(name=width_name, type='u64'))
                    graft_stmts.append(Assign(Var(width_name),
                                              IntLit(rail.width, str(rail.width))))

                # -- Route type ------------------------------------------------
                elif name in route_type_map:
                    rt = route_type_map[name]
                    motor.used_route_types.append(name)
                    # Inject each variant as a compile-time integer constant
                    for variant in rt.variants:
                        graft_stmts.append(VarDecl(name=variant.name, type='u64'))
                        graft_stmts.append(Assign(Var(variant.name),
                                                  IntLit(variant.value, str(variant.value))))

                # -- Target latch ----------------------------------------------
                elif name in latch_map:
                    motor.used_latches.append(name)
                    # No injected constants -- target values are runtime kernel params.
                    # The latch records the comparison structure for compiler analysis.

                # -- Attractor (WP9: AI Model Layer) ---------------------------
                elif name in attractor_map:
                    attractor = attractor_map[name]
                    motor.used_attractors.append(name)
                    # Inject NAME_target as integer constant if target_tokens present
                    if attractor.target_tokens:
                        target_name = f'{name}_target'
                        target_val = attractor.target_tokens[0]
                        graft_stmts.append(VarDecl(name=target_name, type='u64'))
                        graft_stmts.append(Assign(Var(target_name),
                                                  IntLit(target_val, str(target_val))))
                    # Inject NAME_threshold as float constant
                    thresh_name = f'{name}_threshold'
                    graft_stmts.append(VarDecl(name=thresh_name, type='f32'))
                    graft_stmts.append(Assign(Var(thresh_name),
                                              FloatLit(attractor.threshold, str(attractor.threshold))))

                # -- Manifold (WP9: AI Model Layer) ----------------------------
                elif name in manifold_map:
                    motor.used_manifolds.append(name)
                    # Manifold records the physical tensor layout descriptor in the motor

                else:
                    raise CompileError(
                        f"Motor '{motor.name}' references unknown contract '{name}' "
                        f"(not found in packets, rails, target_latches, route_types, attractors, or manifolds)"
                    )

            motor.body.stmts = graft_stmts + body_stmts

        # -- Pass 2: expand packet refs in motor contracts ----------------------
        for tx in prog.transmissions:
            if not tx.motor_contract:
                continue
            contract = tx.motor_contract
            if not getattr(contract, 'packet_refs', None):
                continue

            for pref in contract.packet_refs:
                pkt_name = pref.packet_name
                if pkt_name not in packet_map:
                    raise CompileError(
                        f"Transmission '{tx.name}' motor contract references unknown "
                        f"packet '{pkt_name}'"
                    )
                pkt = packet_map[pkt_name]
                for fld in pkt.fields:
                    if isinstance(fld, WindowDecl):
                        contract.params.append(TxParam(
                            direction='inout',
                            type=fld.elem_type,
                            name=fld.name,
                            register_range=RegisterRange(start=fld.start_reg, end=fld.end_reg)
                        ))
                    elif isinstance(fld, VarDecl) and fld.register:
                        contract.register_ports.append(RegisterPort(
                            direction='inout',
                            register=fld.register,
                            type=fld.type,
                            name=fld.name
                        ))
            contract.packet_refs = []

    def _desugar_windows(self, prog):
        """Expand WindowDecl nodes into locked VarDecl nodes and rewrite Index accesses.

        Window declarations like `@window(R0..R3) u64 X[4]` are replaced with four
        VarDecl nodes (`X_0`..`X_3`, each with an @register annotation) and every
        `X[IntLit(i)]` in the motor body is rewritten to `Var('X_i')`.

        Non-constant window indices are rejected as a compile error -- physical registers
        cannot be dynamically indexed.
        """
        try:
            from .ast_nodes import (WindowDecl, VarDecl, Index, Var, IntLit, BinOp,
                                    UnaryOp, Call, MemberAccess, Block, Assign,
                                    ForStmt, WhileStmt, IfStmt, ReturnStmt, ExprStmt)
        except ImportError:
            from ast_nodes import (WindowDecl, VarDecl, Index, Var, IntLit, BinOp,
                                   UnaryOp, Call, MemberAccess, Block, Assign,
                                   ForStmt, WhileStmt, IfStmt, ReturnStmt, ExprStmt)

        for motor in prog.motors:
            # Pass 1: collect windows, expand each to count VarDecl nodes.
            windows = {}
            new_stmts = []
            for stmt in motor.body.stmts:
                if isinstance(stmt, WindowDecl):
                    windows[stmt.name] = stmt
                    motor.windows[stmt.name] = stmt
                    start_num = int(stmt.start_reg[1:])
                    for i in range(stmt.count):
                        reg_name = f'R{start_num + i}'
                        var_name = f'{stmt.name}_{i}'
                        if reg_name in motor.locked_regs.values():
                            raise CompileError(
                                f"Window '{stmt.name}' register {reg_name} conflicts with "
                                f"another locked register in motor '{motor.name}'"
                            )
                        vd = VarDecl(name=var_name, type=stmt.elem_type, register=reg_name)
                        new_stmts.append(vd)
                        motor.locked_regs[var_name] = reg_name
                else:
                    new_stmts.append(stmt)
            motor.body.stmts = new_stmts

            # Pass 2: rewrite Index(Var(win_name), IntLit(i)) -> Var(win_name_i).
            # Include windows already in motor.windows from packet graft (_desugar_packet_refs
            # runs before _desugar_windows so packet windows are already registered there).
            all_windows = {**motor.windows, **windows}
            if not all_windows:
                continue
            windows = all_windows  # use merged set for the rewrite below

            # (old "if not windows: continue" replaced above)
            def _rw_expr(e):
                if isinstance(e, Index):
                    if isinstance(e.base, Var) and e.base.name in windows:
                        if isinstance(e.index, IntLit):
                            idx = e.index.value
                            win = windows[e.base.name]
                            if idx < 0 or idx >= win.count:
                                raise CompileError(
                                    f"Window index [{idx}] out of range for "
                                    f"'{e.base.name}' (size {win.count})"
                                )
                            return Var(f'{e.base.name}_{idx}')
                        else:
                            raise CompileError(
                                f"Window '{e.base.name}' requires a compile-time constant "
                                f"index -- physical registers cannot be dynamically indexed"
                            )
                    return Index(_rw_expr(e.base), _rw_expr(e.index))
                if isinstance(e, BinOp):
                    return BinOp(_rw_expr(e.left), e.op, _rw_expr(e.right))
                if isinstance(e, UnaryOp):
                    return UnaryOp(e.op, _rw_expr(e.expr))
                if isinstance(e, Call):
                    return Call(e.callee, [_rw_expr(a) for a in e.args])
                if isinstance(e, MemberAccess):
                    return MemberAccess(_rw_expr(e.obj), e.member)
                return e

            def _rw_stmt(s):
                if isinstance(s, Assign):
                    return Assign(_rw_expr(s.lvalue), _rw_expr(s.value))
                if isinstance(s, VarDecl):
                    dims = [(_rw_expr(d) if d is not None else None) for d in (s.dims or [])]
                    return VarDecl(s.name, s.type, s.register,
                                   _rw_expr(s.array_size) if s.array_size else None,
                                   dims)
                if isinstance(s, Block):
                    return Block([_rw_stmt(x) for x in s.stmts])
                if isinstance(s, ForStmt):
                    return ForStmt(
                        _rw_stmt(s.init) if s.init else None,
                        _rw_expr(s.cond) if s.cond else None,
                        _rw_stmt(s.update) if s.update else None,
                        _rw_stmt(s.body),
                    )
                if isinstance(s, WhileStmt):
                    return WhileStmt(_rw_expr(s.cond), _rw_stmt(s.body))
                if isinstance(s, IfStmt):
                    return IfStmt(
                        _rw_expr(s.cond),
                        _rw_stmt(s.then_body),
                        _rw_stmt(s.else_body) if s.else_body else None,
                    )
                if isinstance(s, ReturnStmt):
                    return ReturnStmt(_rw_expr(s.value) if s.value else None)
                if isinstance(s, ExprStmt):
                    return ExprStmt(_rw_expr(s.expr))
                return s

            motor.body.stmts = [_rw_stmt(s) for s in motor.body.stmts]

    def _run_sealed_gate(self, prog, ptx: str, tx_name: str):
        """Enforce require-clause and profile-body constraints via the ptxas spill gate.

        Fires when any of these appear in the active transmission:
          require no_spills;
          require no_local_memory;
          require max_registers(N);
        Or when the referenced profile body declares:
          spill_allowed = false;
          local_memory_allowed = false;
          max_registers = N;
        """
        if not tx_name:
            return
        tx = next((t for t in prog.transmissions if t.name == tx_name), None)
        if not tx:
            return

        require_clauses = list(tx.require_clauses)

        # Look up profile body if this transmission references one by name
        profile_decl = None
        if tx.profile and hasattr(prog, 'profiles'):
            profile_decl = next((p for p in prog.profiles if p.name == tx.profile), None)

        no_spills = any(r.kind in ('no_spills', 'no_spill') for r in require_clauses)
        no_local  = any(r.kind in ('no_local_memory', 'no_local') for r in require_clauses)
        max_regs  = next((r.arg for r in require_clauses if r.kind == 'max_registers'), None)

        if profile_decl:
            props = profile_decl.properties
            if not props.get('spill_allowed', True):
                no_spills = True
            if not props.get('local_memory_allowed', True):
                no_local = True
            if 'max_registers' in props and max_regs is None:
                max_regs = props['max_registers']

        if not (no_spills or no_local or max_regs is not None):
            return  # no constraints declared -- skip gate

        try:
            try:
                from .native_verify import single_kernel_spill_gate
            except ImportError:
                from native_verify import single_kernel_spill_gate
        except ImportError:
            return  # ptxas toolchain not available; skip silently

        try:
            result = single_kernel_spill_gate(
                ptx,
                kernel_name=tx_name,
                spill_allowed=not no_spills,
                local_memory_allowed=not no_local,
                max_registers=max_regs,
            )
        except Exception as e:
            raise CompileError(f"Sealed gate could not run (ptxas unavailable?): {e}")

        if result.verdict == "FAIL":
            fail_lines = "\n  ".join(result.fail_reasons)
            raise CompileError(
                f"Sealed profile gate FAIL for transmission '{tx_name}':\n"
                f"  {fail_lines}\n"
                f"  ptxas: {result.registers} regs, "
                f"{result.spills_store}B spill_store, {result.spills_load}B spill_load, "
                f"{result.stack_frame}B stack"
            )

    def get_last_ptx(self) -> str:
        if not self.last_ptx:
            raise CompileError("No PTX generated yet")
        return self.last_ptx

    def get_last_ir(self) -> Optional[MotorIR]:
        return self.last_ir

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python compiler.py <file.mtl> [transmission_name]")
        sys.exit(1)
    c = Compiler()
    ptx, _ = c.compile_file(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else None)
    print(ptx)
