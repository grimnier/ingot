"""
Affine hop motor parse + KAT test.

Verifies:
  1. affine_hop.mtl parses without error.
  2. Motor params have correct directions and types (inout affine, in affine, in field).
  3. The motor body contains the expected field variable declarations.
  4. field_interp affine_hop KAT passes (G+2G=3G, 2G+3G=5G).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lexer import Lexer
from parser import Parser
from ast_nodes import (
    MotorDecl, MotorParam, Block, VarDecl,
)

HOP_MTL = os.path.join(os.path.dirname(__file__), '..', 'examples', 'affine_hop.mtl')


def load_ast():
    src = open(HOP_MTL, encoding='utf-8').read()
    tokens = Lexer(src).tokenize()
    return Parser(tokens).parse()


def test_motor_params():
    prog = load_ast()
    assert len(prog.motors) == 1, f"Expected 1 motor, got {len(prog.motors)}"
    motor = prog.motors[0]
    assert motor.name == "affine_hop"
    assert len(motor.params) == 3, f"Expected 3 motor params, got {len(motor.params)}"

    p0 = motor.params[0]
    assert p0.direction == "inout", f"p0 direction: {p0.direction}"
    assert p0.type == "affine<secp256k1>", f"p0 type: {p0.type}"
    assert p0.name == "walker"

    p1 = motor.params[1]
    assert p1.direction == "in", f"p1 direction: {p1.direction}"
    assert p1.type == "affine<secp256k1>", f"p1 type: {p1.type}"
    assert p1.name == "jump"

    p2 = motor.params[2]
    assert p2.direction == "in", f"p2 direction: {p2.direction}"
    assert p2.type == "field", f"p2 type: {p2.type}"
    assert p2.name == "my_inv"

    print("PASS: motor params (inout affine<secp256k1> walker, in affine<secp256k1> jump, in field my_inv)")


def test_motor_body_field_decls():
    prog = load_ast()
    motor = prog.motors[0]
    field_vars = []
    for stmt in motor.body.stmts:
        if isinstance(stmt, Block):
            for s in stmt.stmts:
                if isinstance(s, VarDecl) and s.type == "field":
                    field_vars.append(s.name)
        elif isinstance(stmt, VarDecl) and stmt.type == "field":
            field_vars.append(stmt.name)

    expected = ["num", "slope", "slope2", "new_ax", "diff", "new_ay"]
    for name in expected:
        assert name in field_vars, f"Missing field var '{name}'; got {field_vars}"

    print(f"PASS: motor body field vars: {field_vars}")


def test_affine_hop_kat():
    from field_interp import run_affine_hop_kat
    run_affine_hop_kat()


if __name__ == "__main__":
    failures = 0
    for name, fn in [
        ("motor params",          test_motor_params),
        ("motor body field vars", test_motor_body_field_decls),
        ("affine_hop KAT",        test_affine_hop_kat),
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
        print("\nAll affine_hop tests passed.")
    else:
        print(f"\n{failures} test(s) failed.")
        sys.exit(1)
