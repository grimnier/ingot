"""
Ingot Recursive Descent Parser - Full Phase 1 Implementation

Parses motor + transmission programs into clean AST.
Enforces basic syntactic rules (no deep semantic yet).
Strictly separated motor (compute) vs transmission (contract + geometry).
"""

from typing import List, Optional
try:
    from .lexer import Token, Lexer, LexerError
    from .ast_nodes import *
except ImportError:
    from lexer import Token, Lexer, LexerError
    from ast_nodes import *

class ParseError(Exception):
    def __init__(self, msg: str, token: Token):
        super().__init__(f"Parse error at {token.line}:{token.column}: {msg}")
        self.token = token

class Parser:
    def __init__(self, tokens: List[Token], filename: str = "<input>"):
        self.tokens = tokens
        self.pos = 0
        self.filename = filename

    def parse(self) -> Program:
        prog = Program()
        while not self._at_end():
            if self._match_keyword("motor"):
                prog.motors.append(self._parse_motor())
            elif self._match_keyword("transmission"):
                prog.transmissions.append(self._parse_transmission())
            elif self._match_keyword("profile"):
                prog.profiles.append(self._parse_profile_decl())
            elif self._match_keyword("packet"):
                prog.packets.append(self._parse_packet_decl())
            elif self._match_keyword("rail"):
                prog.rails.append(self._parse_rail_decl())
            elif self._match_keyword("target_latch"):
                prog.target_latches.append(self._parse_target_latch_decl())
            elif self._match_keyword("route_type"):
                prog.route_types.append(self._parse_route_type_decl())
            elif self._match_keyword("manifold"):
                prog.manifolds.append(self._parse_manifold_decl())
            elif self._match_keyword("attractor"):
                prog.attractors.append(self._parse_attractor_decl())
            else:
                # Host code: skip until next top-level keyword or EOF for Phase 1
                self._skip_host_code()
        return prog

    def _parse_packet_decl(self) -> 'PacketDecl':
        """Parse: packet NAME { (@window or @register field)* }"""
        name = self._consume_identifier()
        self._consume("LBRACE", "expected '{' after packet name")
        fields = []
        while not self._check("RBRACE") and not self._at_end():
            if self._check("WINDOW_RANGE"):
                # @window(Ra..Rb) type name[count];
                range_tok = self._advance()
                parts = range_tok.value.split('..')
                start_reg = parts[0].strip()
                end_reg   = parts[1].strip()
                start_num = int(start_reg[1:])
                end_num   = int(end_reg[1:])
                inferred  = end_num - start_num + 1
                if inferred < 1:
                    raise ParseError(
                        f"Invalid window range {start_reg}..{end_reg} in packet '{name}'",
                        self._current()
                    )
                elem_type = self._parse_type()
                fname     = self._consume_identifier()
                if self._check("LBRACK"):
                    self._advance()
                    sz = self._parse_expr()
                    self._consume("RBRACK", "expected ']' after window size")
                    if isinstance(sz, IntLit) and sz.value != inferred:
                        raise ParseError(
                            f"Window size [{sz.value}] does not match register range "
                            f"{start_reg}..{end_reg} (= {inferred} registers) in packet '{name}'",
                            self._current()
                        )
                if self._check("SEMI"):
                    self._advance()
                fields.append(WindowDecl(name=fname, elem_type=elem_type, count=inferred,
                                         start_reg=start_reg, end_reg=end_reg))
            elif self._check("REGISTER"):
                # @register(Rn) type name;
                reg   = self._advance().value
                typ   = self._parse_type()
                fname = self._consume_identifier()
                if self._check("SEMI"):
                    self._advance()
                fields.append(VarDecl(name=fname, type=typ, register=reg))
            else:
                raise ParseError(
                    "Expected @window or @register field in packet body "
                    f"(got {self._current().type} '{self._current().value}')",
                    self._current()
                )
        self._consume("RBRACE", "expected '}' to close packet body")
        return PacketDecl(name=name, fields=fields)

    def _parse_rail_decl(self) -> 'RailDecl':
        """Parse: rail NAME { packet: PKT; width: N; stride: S; }"""
        name = self._consume_identifier()
        self._consume("LBRACE", "expected '{' after rail name")
        packet_name = ""
        width = 0
        stride = ""
        while not self._check("RBRACE") and not self._at_end():
            key = self._consume_identifier()
            self._consume("COLON", f"expected ':' after rail property '{key}'")
            if key == "packet":
                packet_name = self._consume_identifier()
            elif key == "width":
                tok = self._consume("INTEGER", "expected integer for rail width")
                width = int(tok.value, 0)
            elif key == "stride":
                stride = self._consume_identifier()
            else:
                raise ParseError(f"Unknown rail property '{key}' (expected packet, width, stride)",
                                 self._current())
            self._consume("SEMI", f"expected ';' after rail property '{key}'")
        self._consume("RBRACE", "expected '}' to close rail body")
        if not packet_name:
            raise ParseError(f"Rail '{name}' must declare a packet", self._current())
        if width == 0:
            raise ParseError(f"Rail '{name}' must declare a non-zero width", self._current())
        return RailDecl(name=name, packet_name=packet_name, width=width, stride=stride)

    def _parse_target_latch_decl(self) -> 'TargetLatchDecl':
        """Parse: target_latch NAME { target_bytes: N; compare_at: [s1, s2, ...]; }"""
        name = self._consume_identifier()
        self._consume("LBRACE", "expected '{' after target_latch name")
        target_bytes = 0
        compare_at = []
        while not self._check("RBRACE") and not self._at_end():
            key = self._consume_identifier()
            self._consume("COLON", f"expected ':' after latch property '{key}'")
            if key == "target_bytes":
                tok = self._consume("INTEGER", "expected integer for target_bytes")
                target_bytes = int(tok.value, 0)
                self._consume("SEMI", "expected ';' after target_bytes")
            elif key == "compare_at":
                self._consume("LBRACK", "expected '[' after compare_at:")
                while not self._check("RBRACK") and not self._at_end():
                    stage = self._consume_identifier()
                    compare_at.append(stage)
                    if self._check("COMMA"):
                        self._advance()
                self._consume("RBRACK", "expected ']' to close compare_at list")
                self._consume("SEMI", "expected ';' after compare_at")
            else:
                raise ParseError(
                    f"Unknown target_latch property '{key}' (expected target_bytes, compare_at)",
                    self._current()
                )
        self._consume("RBRACE", "expected '}' to close target_latch body")
        return TargetLatchDecl(name=name, target_bytes=target_bytes, compare_at=compare_at)

    def _parse_route_type_decl(self) -> 'RouteTypeDecl':
        """Parse: route_type NAME { VARIANT_NAME = integer; ... }"""
        name = self._consume_identifier()
        self._consume("LBRACE", "expected '{' after route_type name")
        variants = []
        while not self._check("RBRACE") and not self._at_end():
            vname = self._consume_identifier()
            self._consume("EQ", f"expected '=' after variant name '{vname}'")
            tok = self._consume("INTEGER", f"expected integer value for variant '{vname}'")
            val = int(tok.value, 0)
            self._consume("SEMI", f"expected ';' after variant '{vname}'")
            variants.append(RouteVariant(name=vname, value=val))
        self._consume("RBRACE", "expected '}' to close route_type body")
        return RouteTypeDecl(name=name, variants=variants)

    def _parse_manifold_decl(self) -> 'ManifoldDecl':
        """Parse: manifold NAME { tensor t1: q8_0 @align(16); ... }"""
        name = self._consume_identifier()
        self._consume("LBRACE", "expected '{' after manifold name")
        tensors = []
        while not self._check("RBRACE") and not self._at_end():
            self._consume_keyword("tensor")
            tname = self._consume_identifier()
            self._consume("COLON", "expected ':' after tensor name")
            dtype = self._parse_type()
            align = 16
            if self._check("ANNOTATION") and self._current().value == "@align":
                self._advance()
                self._consume("LPAREN", "expected '(' after @align")
                tok = self._consume("INTEGER", "expected integer in @align")
                align = int(tok.value, 0)
                self._consume("RPAREN", "expected ')' after @align value")
            self._consume("SEMI", "expected ';' after tensor declaration")
            tensors.append(TensorDecl(name=tname, dtype=dtype, alignment=align))
        self._consume("RBRACE", "expected '}' to close manifold body")
        return ManifoldDecl(name=name, tensors=tensors)

    def _parse_attractor_decl(self) -> 'AttractorDecl':
        """Parse: attractor NAME { pin: [t1, t2]; threshold: 0.85; exit_on_hit: true; }"""
        name = self._consume_identifier()
        self._consume("LBRACE", "expected '{' after attractor name")
        target_tokens = []
        threshold = 0.0
        exit_on_hit = True
        while not self._check("RBRACE") and not self._at_end():
            key = self._consume_identifier()
            self._consume("COLON", f"expected ':' after attractor property '{key}'")
            if key == "pin":
                self._consume("LBRACK", "expected '[' after pin:")
                while not self._check("RBRACK") and not self._at_end():
                    tok = self._consume("INTEGER", "expected integer token ID")
                    target_tokens.append(int(tok.value, 0))
                    if self._check("COMMA"):
                        self._advance()
                self._consume("RBRACK", "expected ']' to close pin list")
                self._consume("SEMI", "expected ';' after pin property")
            elif key == "threshold":
                if self._check("FLOAT"):
                    tok = self._advance()
                    threshold = float(tok.value)
                elif self._check("INTEGER"):
                    tok = self._advance()
                    threshold = float(int(tok.value, 0))
                else:
                    raise ParseError("expected float or integer for threshold", self._current())
                self._consume("SEMI", "expected ';' after threshold")
            elif key == "exit_on_hit":
                if self._check_keyword("true"):
                    self._advance()
                    exit_on_hit = True
                elif self._check_keyword("false"):
                    self._advance()
                    exit_on_hit = False
                else:
                    raise ParseError("expected true/false for exit_on_hit", self._current())
                self._consume("SEMI", "expected ';' after exit_on_hit")
            else:
                raise ParseError(
                    f"Unknown attractor property '{key}' (expected pin, threshold, exit_on_hit)",
                    self._current()
                )
        self._consume("RBRACE", "expected '}' to close attractor body")
        return AttractorDecl(name=name, target_tokens=target_tokens, threshold=threshold, exit_on_hit=exit_on_hit)

    def _parse_profile_decl(self) -> 'ProfileDecl':
        name = self._consume_identifier()
        self._consume("LBRACE", "expected '{' after profile name")
        properties = {}
        while not self._check("RBRACE") and not self._at_end():
            key = self._consume_identifier()
            self._consume("EQ", f"expected '=' after profile property '{key}'")
            if self._check("INTEGER"):
                val = int(self._advance().value, 0)
            elif self._check("IDENTIFIER") or self._check("KEYWORD"):
                raw = self._advance().value
                if raw == 'true':
                    val = True
                elif raw == 'false':
                    val = False
                else:
                    raise ParseError(f"Unknown profile value '{raw}' (expected integer, true, or false)", self._current())
            else:
                raise ParseError("Expected profile property value (integer or true/false)", self._current())
            self._consume("SEMI", f"expected ';' after profile property '{key}'")
            properties[key] = val
        self._consume("RBRACE", "expected '}' to close profile body")
        return ProfileDecl(name=name, properties=properties)

    # ---------------- Motor ----------------

    def _parse_motor(self) -> MotorDecl:
        name = self._consume_identifier()
        reg_lock = None
        if self._check("REGISTER"):
            reg_lock = self._advance().value

        self._consume("LBRACE", "expected '{' after motor name")

        # WP1: collect leading direction-typed parameters (inout jacobian<secp256k1> walker;)
        params = []
        while self._is_direction():
            params.append(self._parse_motor_param())

        body = self._parse_block_body(is_motor=True)

        motor = MotorDecl(name=name, params=params, register_lock=reg_lock, body=body)

        # Collect locked registers from var decls for later semantic use
        for stmt in body.stmts:
            if isinstance(stmt, VarDecl) and stmt.register:
                motor.locked_regs[stmt.name] = stmt.register

        self._consume("RBRACE", "expected '}' to close motor")
        return motor

    def _is_direction(self) -> bool:
        return self._check_keyword("in") or self._check_keyword("out") or self._check_keyword("inout")

    def _parse_motor_param(self) -> MotorParam:
        direction = self._advance().value  # in | out | inout
        typ = self._parse_type()
        name = self._consume_identifier()
        self._consume("SEMI", "expected ';' after motor parameter")
        return MotorParam(direction=direction, type=typ, name=name)

    def _parse_block_body(self, is_motor: bool = False) -> Block:
        stmts: List[Stmt] = []
        while not self._check("RBRACE") and not self._at_end():
            if self._check_keyword("motor") or self._check_keyword("transmission"):
                # Should not appear inside motor body
                raise ParseError("Nested motor/transmission not allowed inside block", self._current())
            stmts.append(self._parse_motor_stmt())
        return Block(stmts)

    def _parse_motor_stmt(self) -> Stmt:
        if self._check("LBRACE"):
            self._advance()
            body = self._parse_block_body()
            self._consume("RBRACE", "expected '}' to close block")
            return body

        # Control flow (must come before var decl / expr)
        if self._check_keyword("break"):
            self._advance()
            if self._check("SEMI"): self._advance()
            return BreakStmt()
        if self._check_keyword("continue"):
            self._advance()
            if self._check("SEMI"): self._advance()
            return ContinueStmt()
        if self._check_keyword("__shared__") or self._check_keyword("shared"):
            self._advance()
            elem_type = self._parse_type()
            name = self._consume_identifier()
            dims = self._parse_bracket_dims()
            if not dims:
                raise ParseError("expected '[' after shared array name", self._current())
            arr = dims[0] if len(dims) == 1 and dims[0] is not None else None
            if self._check("SEMI"): self._advance()
            return SharedDeclStmt(elem_type=elem_type, name=name, array_size=arr, dims=dims)
        if self._check_keyword("__syncthreads") or self._check_keyword("syncthreads"):
            self._advance()
            if self._check("LPAREN"):
                self._advance()
                self._consume("RPAREN", "expected ')' after syncthreads")
            if self._check("SEMI"): self._advance()
            return SyncThreadsStmt(barrier_id=0)
        if self._check_keyword("atomicAdd"):
            self._advance()
            self._consume("LPAREN", "expected '(' after atomicAdd")
            ptr = self._parse_expr()
            self._consume("COMMA", "expected ',' after atomicAdd ptr")
            val = self._parse_expr()
            self._consume("RPAREN", "expected ')' after atomicAdd val")
            if self._check("SEMI"): self._advance()
            return AtomicAddStmt(target_ptr=ptr, val_expr=val)
        if self._check_keyword("for"):
            return self._parse_for_stmt()
        if self._check_keyword("while"):
            return self._parse_while_stmt()
        if self._check_keyword("if"):
            return self._parse_if_stmt()
        if self._check_keyword("return"):
            self._advance()
            val = None
            if not self._check("SEMI"):
                val = self._parse_expr()
            if self._check("SEMI"):
                self._advance()
            return ReturnStmt(value=val)

        # use packet_name; -- graft a packet's register layout into this motor
        if self._check_keyword("use"):
            self._advance()
            pkt_name = self._consume_identifier()
            if self._check("SEMI"):
                self._advance()
            return PacketUse(packet_name=pkt_name)

        # Window declaration: @window(Ra..Rb) type name[count];
        if self._check("WINDOW_RANGE"):
            range_tok = self._advance()
            parts = range_tok.value.split('..')
            start_reg = parts[0].strip()
            end_reg = parts[1].strip()
            start_num = int(start_reg[1:])
            end_num = int(end_reg[1:])
            inferred_count = end_num - start_num + 1
            if inferred_count < 1:
                raise ParseError(f"Invalid window range {start_reg}..{end_reg}: start must be <= end", self._current())
            elem_type = self._parse_type()
            win_name = self._consume_identifier()
            # Optional explicit size -- must match register range if given
            if self._check("LBRACK"):
                self._advance()
                sz_expr = self._parse_expr()
                self._consume("RBRACK", "expected ']' after window size")
                if isinstance(sz_expr, IntLit) and sz_expr.value != inferred_count:
                    raise ParseError(
                        f"Window size [{sz_expr.value}] does not match register range "
                        f"{start_reg}..{end_reg} (= {inferred_count} registers)",
                        self._current()
                    )
            if self._check("SEMI"):
                self._advance()
            return WindowDecl(name=win_name, elem_type=elem_type, count=inferred_count,
                              start_reg=start_reg, end_reg=end_reg)

        # var decl with optional @register
        reg = None
        if self._check("REGISTER"):
            reg = self._advance().value

        if self._check_type_or_ident():
            typ = self._parse_type()
            name = self._consume_identifier()
            dims = self._parse_bracket_dims()
            arr = dims[0] if len(dims) == 1 and dims[0] is not None else None

            # Support local var with initializer: float t0 = expr;
            if self._check("EQ"):
                self._advance()
                init_expr = self._parse_expr()
                if self._check("SEMI"):
                    self._advance()
                # Return a tiny block so existing lowering can treat it as decl + assign
                decl = VarDecl(name=name, type=typ, register=reg, array_size=arr, dims=dims)
                assign = Assign(lvalue=Var(name), value=init_expr)
                return Block([decl, assign])

            if self._check("SEMI"):
                self._advance()
            return VarDecl(name=name, type=typ, register=reg, array_size=arr, dims=dims)

        # assignment or expr (handles y[i] = ... and plain assignments)
        expr = self._parse_expr()
        if self._check("EQ"):
            self._advance()
            rhs = self._parse_expr()
            if self._check("SEMI"):
                self._advance()
            return Assign(lvalue=expr, value=rhs)

        if self._check("SEMI"):
            self._advance()
        return ExprStmt(expr=expr)

    def _parse_bracket_dims(self) -> list:
        """Parse a run of array brackets. An empty [] is an unsized dimension.

        `name[12][4]` -> [IntLit(12), IntLit(4)]
        `name[][12][4]` -> [None, IntLit(12), IntLit(4)]
        No bracket -> [].
        """
        dims = []
        while self._check("LBRACK"):
            self._advance()
            if self._check("RBRACK"):
                self._advance()
                dims.append(None)
            else:
                dims.append(self._parse_expr())
                self._consume("RBRACK", "expected ']'")
        return dims

    def _parse_type(self) -> str:
        # WP1: field, jacobian<C>, affine<C>
        if self._check("KEYWORD") and self._current().value == "field":
            self._advance()
            return "field"
        if self._check("KEYWORD") and self._current().value in ("jacobian", "affine"):
            base = self._advance().value
            self._consume("LT", f"expected '<' after {base}")
            param = self._consume_identifier()
            self._consume("GT", f"expected '>' after {base} type parameter")
            return f"{base}<{param}>"

        # Existing primitive types + WP9 quantized neural types
        TYPE_KEYWORDS = {"int", "uint", "float", "double", "i32", "u32", "i64", "u64", "f32", "f64", "float*", "double*",
                         "q8_0", "q6_k", "iq2_xxs", "f16", "bf16"}
        parts = []
        while True:
            if self._check("STAR"):
                parts.append(self._advance().value)
                continue
            if self._check("KEYWORD") and self._current().value in TYPE_KEYWORDS:
                parts.append(self._advance().value)
                continue
            if self._check("IDENTIFIER") and self._current().value in TYPE_KEYWORDS:
                parts.append(self._advance().value)
                continue
            break
        if not parts:
            raise ParseError("Expected type", self._current())
        return " ".join(parts)

    # ---------------- Transmission ----------------

    def _parse_transmission(self) -> TransmissionDecl:
        name = self._consume_identifier()
        self._consume("LBRACE", "expected '{' after transmission name")

        # WP1: optional target, profile, verify, require clauses before grid/block
        target = None
        profile = None
        verify_clauses = []
        require_clauses = []

        while True:
            if self._check_keyword("target"):
                self._advance()
                parts = [self._consume_identifier()]
                while self._check("DOT"):
                    self._advance()
                    parts.append(self._consume_identifier())
                target = ".".join(parts)
                self._consume("SEMI", "expected ';' after target")
            elif self._check_keyword("profile"):
                self._advance()
                profile = self._consume_identifier()
                self._consume("SEMI", "expected ';' after profile")
            elif self._check_keyword("verify"):
                self._advance()
                kind = self._consume_identifier()
                arg = None
                if self._check("LPAREN"):
                    self._advance()
                    arg = self._consume_identifier()
                    self._consume("RPAREN", "expected ')' after verify argument")
                self._consume("SEMI", "expected ';' after verify clause")
                verify_clauses.append(VerifyClause(kind=kind, arg=arg))
            elif self._check_keyword("require"):
                self._advance()
                kind = self._consume_identifier()
                arg = None
                if self._check("LPAREN"):
                    self._advance()
                    # input_invariant takes an expression; max_registers takes an int
                    if kind == "max_registers":
                        tok = self._consume("INTEGER", "expected integer in max_registers")
                        arg = int(tok.value, 0)
                    else:
                        arg = self._parse_expr()
                    self._consume("RPAREN", f"expected ')' after {kind} argument")
                self._consume("SEMI", "expected ';' after require clause")
                require_clauses.append(RequireClause(kind=kind, arg=arg))
            elif self._check_keyword("forbid"):
                # forbid IDENTIFIER; -- records a prohibition contract, not yet enforced at compile time
                self._advance()
                self._consume_identifier()  # consume the forbidden name, discard for now
                self._consume("SEMI", "expected ';' after forbid clause")
            else:
                break

        # grid and block are optional (plain-motor transmissions may omit them)
        grid = []
        block = []
        if self._check_keyword("grid"):
            grid = self._parse_dim3("grid")
        if self._check_keyword("block"):
            block = self._parse_dim3("block")

        stream = None
        if self._match_keyword("stream"):
            stream = self._consume_identifier()
            self._consume("SEMI", "expected ';' after stream")

        contract = self._parse_motor_contract() if self._check_keyword("motor") else None

        self._consume("RBRACE", "expected '}' to close transmission")

        return TransmissionDecl(
            name=name,
            grid=grid,
            block=block,
            stream=stream,
            motor_contract=contract,
            target=target,
            profile=profile,
            verify_clauses=verify_clauses,
            require_clauses=require_clauses,
        )

    def _parse_dim3(self, keyword: str) -> List[Expr]:
        self._consume_keyword(keyword)
        self._consume("LPAREN", f"expected '(' after {keyword}")
        dims = []
        dims.append(self._parse_expr())
        while self._check("COMMA"):
            self._advance()
            dims.append(self._parse_expr())
        self._consume("RPAREN", "expected ')'")
        self._consume("SEMI", f"expected ';' after {keyword} decl")
        return dims

    def _parse_motor_contract(self) -> MotorContract:
        self._consume_keyword("motor")
        motor_name = self._consume_identifier()

        # Support optional @boundary_lock(strict_zero_temps) annotation on the contract
        # (user blueprint: coupling_interface using @boundary_lock for the precision bellhousing)
        boundary_lock = None
        if self._check("ANNOTATION") and self._current().value.startswith("@boundary_lock"):
            ann = self._advance().value
            # Parse (strict_zero_temps) or similar policy
            if self._check("LPAREN"):
                self._advance()
                policy_tok = self._consume("IDENTIFIER", "expected boundary lock policy")
                boundary_lock = policy_tok.value
                self._consume("RPAREN", "expected ')' after @boundary_lock policy")

        self._consume("LBRACE", "expected '{' for motor contract")

        params: List[TxParam] = []
        register_ports: List[RegisterPort] = []
        packet_refs: List[PacketRef] = []

        while not self._check("RBRACE") and not self._at_end():
            # packet packet_name; -- direction-less reference, always inout
            if self._check_keyword("packet"):
                self._advance()
                pkt_name = self._consume_identifier()
                self._consume("SEMI", "expected ';' after packet reference in contract")
                packet_refs.append(PacketRef(packet_name=pkt_name))
                continue

            direction = self._consume_direction()
            # WP1: optional registers(R0..R11) range annotation
            reg_range = None
            if self._check("IDENTIFIER") and self._current().value == "registers":
                self._advance()
                self._consume("LPAREN", "expected '(' after registers")
                start = self._consume("REGISTER", "expected register like R0")
                self._consume("DOTDOT", "expected '..' in register range")
                end = self._consume("REGISTER", "expected register like R11")
                self._consume("RPAREN", "expected ')' after register range")
                reg_range = RegisterRange(start=start.value, end=end.value)
            # Distinguish tx_param vs register_port by peeking for REGISTER token (no range)
            if reg_range is None and self._check("REGISTER"):
                reg = self._advance().value
                typ = self._parse_type()
                pname = self._consume_identifier()
                align = None
                if self._check("ANNOTATION") and self._current().value == "@align":
                    self._advance()
                    self._consume("LPAREN", "")
                    align_tok = self._consume("INTEGER", "expected integer in @align")
                    align = int(align_tok.value)
                    self._consume("RPAREN", "")
                self._consume("SEMI", "expected ';' after register port")
                register_ports.append(RegisterPort(direction=direction, register=reg, type=typ, name=pname))
            else:
                typ = self._parse_type()
                pname = self._consume_identifier()
                align = None
                if self._check("ANNOTATION") and self._current().value == "@align":
                    self._advance()
                    self._consume("LPAREN", "")
                    align_tok = self._consume("INTEGER", "expected integer in @align")
                    align = int(align_tok.value)
                    self._consume("RPAREN", "")
                self._consume("SEMI", "expected ';' after contract param")
                params.append(TxParam(direction=direction, type=typ, name=pname, align=align,
                                      register_range=reg_range))

        self._consume("RBRACE", "expected '}' after motor contract")
        return MotorContract(motor_name=motor_name, params=params, register_ports=register_ports,
                             boundary_lock_policy=boundary_lock, packet_refs=packet_refs)

    # ---------------- Expressions ----------------

    def _parse_expr(self) -> Expr:
        return self._parse_logical_or()

    def _parse_logical_or(self) -> Expr:
        expr = self._parse_logical_and()
        while self._check_op("||"):
            op = self._advance().value
            right = self._parse_logical_and()
            expr = BinOp(expr, op, right)
        return expr

    def _parse_logical_and(self) -> Expr:
        expr = self._parse_bitwise_or()
        while self._check_op("&&"):
            op = self._advance().value
            right = self._parse_bitwise_or()
            expr = BinOp(expr, op, right)
        return expr

    def _parse_bitwise_or(self) -> Expr:
        expr = self._parse_bitwise_xor()
        while self._current().type == "PIPE":
            self._advance()
            right = self._parse_bitwise_xor()
            expr = BinOp(expr, "|", right)
        return expr

    def _parse_bitwise_xor(self) -> Expr:
        expr = self._parse_bitwise_and()
        while self._current().type == "CARET":
            self._advance()
            right = self._parse_bitwise_and()
            expr = BinOp(expr, "^", right)
        return expr

    def _parse_bitwise_and(self) -> Expr:
        expr = self._parse_equality()
        while self._current().type == "AMP":
            self._advance()
            right = self._parse_equality()
            expr = BinOp(expr, "&", right)
        return expr

    def _parse_equality(self) -> Expr:
        expr = self._parse_relational()
        while self._check_op_in(("==", "!=")):
            op = self._advance().value
            right = self._parse_relational()
            expr = BinOp(expr, op, right)
        return expr

    def _parse_relational(self) -> Expr:
        expr = self._parse_shift()
        while (self._check_op_in(("<", "<=", ">", ">=")) or
               self._current().type in ("LT", "GT") or
               (self._current().type == "OPERATOR" and self._current().value in ("<=", ">="))):
            tok = self._advance()
            op = tok.value
            right = self._parse_shift()
            expr = BinOp(expr, op, right)
        return expr

    def _parse_shift(self) -> Expr:
        expr = self._parse_additive()
        while self._check_op_in((">>", "<<")):
            op = self._advance().value
            right = self._parse_additive()
            expr = BinOp(expr, op, right)
        return expr

    def _parse_additive(self) -> Expr:
        expr = self._parse_multiplicative()
        while (self._check_op_in(("+", "-")) or 
               self._current().type in ("PLUS", "MINUS")):
            tok = self._advance()
            op = tok.value
            right = self._parse_multiplicative()
            expr = BinOp(expr, op, right)
        return expr

    def _parse_multiplicative(self) -> Expr:
        expr = self._parse_unary()
        while (self._check_op_in(("*", "/", "%")) or 
               self._current().type in ("STAR", "SLASH", "PERCENT")):
            tok = self._advance()
            op = tok.value
            right = self._parse_unary()
            expr = BinOp(expr, op, right)
        return expr

    def _parse_unary(self) -> Expr:
        # '&' is token AMP, not OPERATOR, so the address-of spelling
        # atomicAdd(&ptr, val) has to be recognized here. Infix '&' stays
        # in the bitwise parser, which only sees AMP after a primary.
        if self._check_op_in(("-", "!", "&", "*")) or self._current().type == "AMP":
            op = self._advance().value
            expr = self._parse_unary()
            return UnaryOp(op, expr)
        if self._current().type == "TILDE":
            self._advance()
            expr = self._parse_unary()
            return UnaryOp("~", expr)
        return self._parse_postfix()

    def _parse_postfix(self) -> Expr:
        expr = self._parse_primary()
        while True:
            if self._check("DOT"):
                self._advance()
                member = self._consume_identifier()
                expr = MemberAccess(obj=expr, member=member)
            elif self._check("LBRACK"):
                self._advance()
                idx = self._parse_expr()
                self._consume("RBRACK", "expected ']'")
                expr = Index(base=expr, index=idx)
            else:
                break
        return expr

    def _parse_primary(self) -> Expr:
        if self._check("BUILTIN"):
            tok = self._advance()
            return Call(callee=f"__builtin_{tok.value.replace('.', '_')}", args=[])

        # Keywords that may appear in expression position:
        #   in/out/inout -- valid variable names inside motor body expressions
        #   field/jacobian/affine -- type keywords that appear in expressions like field.one,
        #                           jacobian(x3, y3, z3)
        # `manifold` is also a top-level declaration keyword. A variable of that
        # name is legal (the declaration parser accepts it), so an index
        # expression manifold[i][j][k] has to see the same name.
        _EXPR_KWS = {'in', 'out', 'inout', 'field', 'jacobian', 'affine', '__shfl_up_sync', '__shfl_down_sync', 'atomicAdd', 'target', 'pin', 'threshold', 'manifold'}
        if self._check("IDENTIFIER") or (self._check("KEYWORD") and self._current().value in _EXPR_KWS):
            name = self._advance().value
            if self._check("LPAREN"):
                self._advance()
                args = []
                if not self._check("RPAREN"):
                    args.append(self._parse_expr())
                    while self._check("COMMA"):
                        self._advance()
                        args.append(self._parse_expr())
                self._consume("RPAREN", "expected ')'")
                return Call(callee=name, args=args)
            return Var(name)

        if self._check("INTEGER"):
            tok = self._advance()
            val = int(tok.value, 0)
            return IntLit(val, tok.value)

        if self._check("FLOAT"):
            tok = self._advance()
            return FloatLit(float(tok.value), tok.value)

        if self._check_keyword("true"):
            self._advance()
            return BoolLit(True)
        if self._check_keyword("false"):
            self._advance()
            return BoolLit(False)

        if self._check("LPAREN"):
            self._advance()
            e = self._parse_expr()
            self._consume("RPAREN", "expected ')'")
            return e

        raise ParseError(f"Unexpected token in expression: {self._current()}", self._current())

    # ---------------- Control Flow (Motor only) ----------------

    def _parse_for_stmt(self) -> ForStmt:
        self._consume_keyword("for")
        self._consume("LPAREN", "expected '(' after for")

        # init: optional (var decl, assignment, expr, or empty)
        init = None
        if not self._check("SEMI"):
            if self._check_type_or_ident():
                reg = None
                if self._check("REGISTER"):
                    reg = self._advance().value
                typ = self._parse_type()
                name = self._consume_identifier()
                dims = self._parse_bracket_dims()
                arr = dims[0] if len(dims) == 1 and dims[0] is not None else None
                if self._check("EQ"):
                    self._advance()
                    value = self._parse_expr()
                    init = Assign(lvalue=Var(name), value=value)
                else:
                    init = VarDecl(name=name, type=typ, register=reg, array_size=arr, dims=dims)
            else:
                e = self._parse_expr()
                if self._check("EQ"):
                    self._advance()
                    rhs = self._parse_expr()
                    init = Assign(lvalue=e, value=rhs)
                else:
                    init = ExprStmt(expr=e)
        self._consume("SEMI", "expected ';' after for init")

        # cond: optional expression
        cond = None
        if not self._check("SEMI"):
            cond = self._parse_expr()
        self._consume("SEMI", "expected ';' after for condition")

        # update: optional (assignment or expr)
        update = None
        if not self._check("RPAREN"):
            e = self._parse_expr()
            if self._check("EQ"):
                self._advance()
                rhs = self._parse_expr()
                update = Assign(lvalue=e, value=rhs)
            else:
                update = ExprStmt(expr=e)
        self._consume("RPAREN", "expected ')' after for update")

        body = self._parse_block_or_single_stmt()
        return ForStmt(init=init, cond=cond, update=update, body=body)

    def _parse_while_stmt(self) -> WhileStmt:
        self._consume_keyword("while")
        self._consume("LPAREN", "expected '(' after while")
        cond = self._parse_expr()
        self._consume("RPAREN", "expected ')' after while condition")
        body = self._parse_block_or_single_stmt()
        return WhileStmt(cond=cond, body=body)

    def _parse_if_stmt(self) -> IfStmt:
        self._consume_keyword("if")
        self._consume("LPAREN", "expected '(' after if")
        cond = self._parse_expr()
        self._consume("RPAREN", "expected ')' after if condition")

        then_body = self._parse_block_or_single_stmt()
        else_body = None
        if self._match_keyword("else"):
            else_body = self._parse_block_or_single_stmt()

        return IfStmt(cond=cond, then_body=then_body, else_body=else_body)

    def _parse_block_or_single_stmt(self) -> Block:
        if self._check("LBRACE"):
            self._advance()
            body = self._parse_block_body()
            self._consume("RBRACE", "expected '}' to close block")
            return body
        # single statement
        stmt = self._parse_motor_stmt()
        return Block([stmt])

    # ---------------- Helpers ----------------

    def _current(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]  # EOF

    def _at_end(self) -> bool:
        return self._current().type == "EOF"

    def _advance(self) -> Token:
        tok = self._current()
        if not self._at_end():
            self.pos += 1
        return tok

    def _check(self, typ: str) -> bool:
        return self._current().type == typ

    def _check_keyword(self, kw: str) -> bool:
        return self._current().type == "KEYWORD" and self._current().value == kw

    def _check_op(self, op: str) -> bool:
        return self._current().type == "OPERATOR" and self._current().value == op

    def _check_op_in(self, ops: tuple) -> bool:
        return self._current().type == "OPERATOR" and self._current().value in ops

    def _check_type_or_ident(self) -> bool:
        if self._current().type in ("KEYWORD", "IDENTIFIER"):
            v = self._current().value
            TYPE_KWS = {"int","uint","float","double","i32","u32","i64","u64","f32","f64",
                        "field","jacobian","affine",
                        "q8_0","q6_k","iq2_xxs","f16","bf16"}
            return v in TYPE_KWS
        return False

    def _consume(self, typ: str, msg: str) -> Token:
        if self._check(typ):
            return self._advance()
        raise ParseError(msg, self._current())

    def _consume_keyword(self, kw: str) -> Token:
        if self._check_keyword(kw):
            return self._advance()
        raise ParseError(f"Expected keyword '{kw}'", self._current())

    def _consume_identifier(self) -> str:
        tok = self._current()
        if tok.type in ("IDENTIFIER", "KEYWORD"):  # allow keywords as idents in some places
            self._advance()
            return tok.value
        raise ParseError("Expected identifier", tok)

    def _consume_direction(self) -> str:
        if self._check_keyword("in") or self._check_keyword("out") or self._check_keyword("inout"):
            return self._advance().value
        raise ParseError("Expected in/out/inout", self._current())

    def _match_keyword(self, kw: str) -> bool:
        if self._check_keyword(kw):
            self._advance()
            return True
        return False

    def _skip_host_code(self):
        # Very simple: skip tokens until we hit a top level keyword or EOF
        depth = 0
        while not self._at_end():
            if self._check("LBRACE"):
                depth += 1
            elif self._check("RBRACE"):
                depth -= 1
            if depth == 0 and (self._check_keyword("motor") or self._check_keyword("transmission")
                           or self._check_keyword("profile") or self._check_keyword("packet")
                           or self._check_keyword("rail") or self._check_keyword("target_latch")
                           or self._check_keyword("route_type") or self._check_keyword("manifold")
                           or self._check_keyword("attractor") or self._at_end()):
                return
            self._advance()

if __name__ == "__main__":
    src = open(__file__.replace("parser.py", "../examples/vector_add.mtl"), encoding="utf-8").read() if False else """
motor vector_add {
    @register(R0) float a;
    @register(R1) float* x;
    @register(R2) float* y;
    for (int i = 0; i < n; i = i + 1) {
        y[i] = a * x[i] + y[i];
    }
}
transmission vadd_launch {
    grid ((n + 255) / 256, 1, 1);
    block (256, 1, 1);
    motor vector_add {
        in  float  a;
        in  float* x @align(16);
        inout float* y @align(16);
        in  int    n;
    }
}
"""
    lexer = Lexer(src)
    toks = lexer.tokenize()
    p = Parser(toks)
    ast = p.parse()
    print("Parsed motors:", [m.name for m in ast.motors])
    print("Parsed transmissions:", [t.name for t in ast.transmissions])
    print("First motor locked_regs:", ast.motors[0].locked_regs if ast.motors else None)
