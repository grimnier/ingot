"""
WP1 parse test: point_add_mixed_plain.mtl

Verifies:
  1. The plain Motor parses without error.
  2. Motor params have correct directions and types.
  3. The Motor body contains the expected field variable declarations.
  4. Both transmissions parse with correct target/profile/verify/require.
  5. Transmission motor contracts have register ranges and parameterized types.
  6. field_interp KATs pass.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lexer import Lexer
from parser import Parser
from ast_nodes import (
    MotorDecl, MotorParam, TransmissionDecl, VerifyClause, RequireClause,
    TxParam, RegisterRange, Block, VarDecl, Assign, MemberAccess, Call, BinOp,
)

PLAIN_MTL = os.path.join(os.path.dirname(__file__), '..', 'examples', 'point_add_mixed_plain.mtl')

def load_ast():
    src = open(PLAIN_MTL, encoding='utf-8').read()
    tokens = Lexer(src).tokenize()
    return Parser(tokens).parse()

def test_motor_params():
    prog = load_ast()
    assert len(prog.motors) == 1, f"Expected 1 motor, got {len(prog.motors)}"
    motor = prog.motors[0]
    assert motor.name == "point_add_mixed_plain"
    assert len(motor.params) == 2, f"Expected 2 motor params, got {len(motor.params)}"

    walker_param = motor.params[0]
    assert walker_param.direction == "inout", f"Expected inout, got {walker_param.direction}"
    assert walker_param.type == "jacobian<secp256k1>", f"Got {walker_param.type}"
    assert walker_param.name == "walker"

    table_param = motor.params[1]
    assert table_param.direction == "in", f"Expected in, got {table_param.direction}"
    assert table_param.type == "affine<secp256k1>", f"Got {table_param.type}"
    assert table_param.name == "table_point"

    print("PASS: motor params (inout jacobian, in affine)")

def test_motor_body_field_decls():
    prog = load_ast()
    motor = prog.motors[0]
    # Body stmts should include field var decls; collect them from nested Block stmts
    field_vars = []
    for stmt in motor.body.stmts:
        if isinstance(stmt, Block):
            for s in stmt.stmts:
                if isinstance(s, VarDecl) and s.type == "field":
                    field_vars.append(s.name)
        elif isinstance(stmt, VarDecl) and stmt.type == "field":
            field_vars.append(stmt.name)

    expected = [
        "z_squared", "u2", "z_cubed", "s2",
        "h", "r",
        "h_squared", "z3", "h_cubed", "xh2", "r_squared",
        "x3", "y3",
    ]
    for name in expected:
        assert name in field_vars, f"Missing field var '{name}'; got {field_vars}"

    print(f"PASS: motor body has {len(field_vars)} field variables")

def test_transmissions():
    prog = load_ast()
    assert len(prog.transmissions) == 2, f"Expected 2 transmissions, got {len(prog.transmissions)}"

    t0, t1 = prog.transmissions

    # --- point_add_sm89 ---
    assert t0.name == "point_add_sm89"
    assert t0.target == "nvidia.sm_89", f"Got {t0.target}"
    assert t0.profile == "secp256k1_mixed_add_compat_v1"

    verify_kinds = [v.kind for v in t0.verify_clauses]
    assert "semantic_equivalence" in verify_kinds, f"Got {verify_kinds}"
    assert "native_text_equal" in verify_kinds
    nte = next(v for v in t0.verify_clauses if v.kind == "native_text_equal")
    assert nte.arg == "point_add_mixed_expanded_v1"

    require_kinds = [r.kind for r in t0.require_clauses]
    assert "no_spills" in require_kinds, f"Got {require_kinds}"
    assert "max_registers" in require_kinds
    mr = next(r for r in t0.require_clauses if r.kind == "max_registers")
    assert mr.arg == 96, f"Got {mr.arg}"

    print("PASS: transmission point_add_sm89 target/profile/verify/require")

    # --- point_add_sm89_z1_fast ---
    assert t1.name == "point_add_sm89_z1_fast"
    assert t1.target == "nvidia.sm_89"
    require_kinds1 = [r.kind for r in t1.require_clauses]
    assert "input_invariant" in require_kinds1, f"Got {require_kinds1}"
    print("PASS: transmission point_add_sm89_z1_fast with input_invariant")

def test_contract_register_ranges():
    prog = load_ast()
    t0 = prog.transmissions[0]
    assert t0.motor_contract is not None
    params = t0.motor_contract.params
    assert len(params) == 2, f"Expected 2 contract params, got {len(params)}"

    walker_p = params[0]
    assert walker_p.direction == "inout"
    assert walker_p.type == "jacobian<secp256k1>"
    assert walker_p.register_range is not None
    assert walker_p.register_range.start == "R0"
    assert walker_p.register_range.end == "R11"

    table_p = params[1]
    assert table_p.direction == "in"
    assert table_p.type == "affine<secp256k1>"
    assert table_p.register_range is not None
    assert table_p.register_range.start == "R12"
    assert table_p.register_range.end == "R19"

    print("PASS: motor contract register ranges R0..R11, R12..R19")

def test_member_access_parses():
    prog = load_ast()
    motor = prog.motors[0]
    # Collect all Assign stmts in the body (field vars use init-assign blocks)
    assigns = []
    for stmt in motor.body.stmts:
        if isinstance(stmt, Block):
            for s in stmt.stmts:
                if isinstance(s, Assign):
                    assigns.append(s)
        elif isinstance(stmt, Assign):
            assigns.append(stmt)

    # Check that walker.z appears as a MemberAccess somewhere in the RHS values
    def has_member_access(expr, obj_name, member_name):
        if isinstance(expr, MemberAccess):
            from ast_nodes import Var
            return (isinstance(expr.obj, Var) and expr.obj.name == obj_name
                    and expr.member == member_name)
        if isinstance(expr, BinOp):
            return has_member_access(expr.left, obj_name, member_name) \
                or has_member_access(expr.right, obj_name, member_name)
        return False

    found_wz = any(has_member_access(a.value, "walker", "z") for a in assigns)
    assert found_wz, "Expected walker.z member access in body"
    print("PASS: member access (walker.z) parsed correctly")

def test_field_interp_kat():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from field_interp import run_kat
    run_kat()

if __name__ == "__main__":
    failures = 0
    for name, fn in [
        ("motor params",         test_motor_params),
        ("motor body field vars",test_motor_body_field_decls),
        ("transmissions",        test_transmissions),
        ("contract reg ranges",  test_contract_register_ranges),
        ("member access",        test_member_access_parses),
        ("field_interp KAT",     test_field_interp_kat),
    ]:
        try:
            fn()
        except AssertionError as e:
            print(f"FAIL [{name}]: {e}")
            failures += 1
        except Exception as e:
            print(f"ERROR [{name}]: {type(e).__name__}: {e}")
            failures += 1

    if failures == 0:
        print(f"\nAll WP1 tests passed.")
    else:
        print(f"\n{failures} test(s) failed.")
        sys.exit(1)
