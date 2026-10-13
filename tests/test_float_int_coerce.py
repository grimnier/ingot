"""Integer operands of a float operation must be converted before the opcode.

`int current_expert; current_expert * 0.001953` used to print

    mul.f32 %f1, %r0, 0.001953;

ptxas rejects that with "Arguments mismatch for instruction 'mul'", which
surfaces at runtime as CUDA error 218. The integer register stays an integer
register. The float opcode is only printed with float operands, after a
cvt.rn.f32.u32. Pure integer kernels do not gain that conversion.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ast_nodes import MotorContract
from compiler import Compiler
from ir import BinOpIR, MotorIR
from ptx_emitter import PTXEmitter

PTXAS = shutil.which('ptxas')

_FLOAT_ARITH = re.compile(
    r'^\s*(mul|add|sub|div|rem|fma|neg|setp)\.(?:[\w.]*)f32\b(.*);'
)
_INT_REG = re.compile(r'%(?:rd\d+|r\d+)\b')
_INT_IMM = re.compile(r'^-?\d+$')


def _compile(src: str) -> str:
    ptx, _ = Compiler().compile_source(src, '<coerce>')
    return ptx


def _float_arith_mismatches(ptx: str):
    bad = []
    for line in ptx.splitlines():
        m = _FLOAT_ARITH.match(line)
        if not m:
            continue
        body = re.sub(r'\[[^\]]*\]', '', m.group(2))
        regs = _INT_REG.findall(body)
        imms = [t.strip() for t in body.split(',') if _INT_IMM.match(t.strip())]
        if regs or imms:
            bad.append(line.strip())
    return bad


def _assemble(ptx: str) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, 'k.ptx')
        out = os.path.join(td, 'k.cubin')
        with open(src, 'w', encoding='utf-8') as f:
            f.write(ptx)
        return subprocess.run(
            [PTXAS, '-arch=sm_89', '-o', out, src],
            capture_output=True, text=True,
        )


SCALE = """
motor scale_expert {
    @register(R1) float* out;
    int current_expert;
    out[0] = current_expert * 0.001953;
}
transmission scale_launch {
    grid (1, 1, 1);
    block (1, 1, 1);
    motor scale_expert { out float* out; }
}
"""


def test_scale_mul_converts_the_int_register():
    ptx = _compile(SCALE)
    assert _float_arith_mismatches(ptx) == []
    assert 'mul.f32 %f1, %r0, 0.001953' not in ptx
    assert 'cvt.rn.f32.u32' in ptx
    mul = next(ln.strip() for ln in ptx.splitlines() if ln.strip().startswith('mul.f32'))
    # Both data operands are float registers. The %r0 is only the cvt source.
    ops = [t.strip() for t in mul.split(',', 1)[1].split(',')]
    assert all(o.startswith('%f') for o in ops)
    cvt_at = ptx.index('cvt.rn.f32.u32')
    mul_at = ptx.index('mul.f32')
    assert cvt_at < mul_at


def test_scale_ir_binop_operands_are_already_f32():
    c = Compiler()
    c.compile_source(SCALE, '<coerce>')
    saw = False
    for op in c.last_ir.ops:
        if isinstance(op, BinOpIR) and op.ptx_type == '.f32' and op.op == '*':
            saw = True
            assert c.last_ir.reg_types[op.left] == '.f32'
            assert c.last_ir.reg_types[op.right] == '.f32'
            assert op.left.startswith('%f') and op.right.startswith('%f')
    assert saw


def _arith(op: str) -> str:
    return f"""
    motor scale_expert {{
        @register(R1) float* out;
        int current_expert;
        out[0] = current_expert {op} 0.001953;
    }}
    transmission scale_launch {{
        grid (1, 1, 1);
        block (1, 1, 1);
        motor scale_expert {{ out float* out; }}
    }}
    """


@pytest.mark.parametrize('op,opcode', [
    ('+', 'add.f32'),
    ('-', 'sub.f32'),
    ('/', 'div.rn.f32'),
])
def test_other_float_ops_convert_too(op, opcode):
    ptx = _compile(_arith(op))
    assert _float_arith_mismatches(ptx) == []
    assert 'cvt.rn.f32.u32' in ptx
    assert any(ln.strip().startswith(opcode) for ln in ptx.splitlines())


def test_fused_fma_uses_the_converted_register():
    src = _arith('*')
    src = src.replace(
        'out[0] = current_expert * 0.001953;',
        'out[0] = current_expert * 0.001953 + 1.0;',
    )
    ptx = _compile(src)
    assert _float_arith_mismatches(ptx) == []
    assert 'cvt.rn.f32.u32' in ptx
    assert any(ln.strip().startswith('fma.rn.f32') for ln in ptx.splitlines())


def test_reversed_operands_and_locked_register():
    src = """
    motor m {
        @register(R0) int current_expert;
        @register(R1) float* out;
        out[0] = 0.001953 * current_expert;
    }
    transmission t {
        grid (1, 1, 1); block (1, 1, 1);
        motor m { out float* out; }
    }
    """
    ptx = _compile(src)
    assert _float_arith_mismatches(ptx) == []
    assert 'cvt.rn.f32.u32 %f1, %R0;' in ptx
    assert not any(
        'mul.f32' in ln and '%R0' in ln for ln in ptx.splitlines()
    )


def test_int_times_int_stays_integer_then_converts_on_float_store():
    src = """
    motor m {
        @register(R1) float* out;
        int a;
        int b;
        float scaled;
        scaled = a * b;
        out[0] = scaled;
    }
    transmission t {
        grid (1, 1, 1); block (1, 1, 1);
        motor m { out float* out; }
    }
    """
    ptx = _compile(src)
    assert any(ln.strip().startswith('mul.lo.u32') for ln in ptx.splitlines())
    assert not any(ln.strip().startswith('mul.f32') for ln in ptx.splitlines())
    assert 'cvt.rn.f32.u32' in ptx


def test_float_product_assigned_to_int_rounds_back():
    src = """
    motor m {
        @register(R1) float* out;
        int current_expert;
        int scaled;
        scaled = current_expert * 0.001953;
        out[0] = scaled;
    }
    transmission t {
        grid (1, 1, 1); block (1, 1, 1);
        motor m { out float* out; }
    }
    """
    ptx = _compile(src)
    assert 'mul.f32' in ptx
    assert 'cvt.rn.f32.u32' in ptx
    assert 'cvt.rni.u32.f32' in ptx
    assert ptx.index('mul.f32') < ptx.index('cvt.rni.u32.f32')


def test_integer_u32_multiply_has_no_float_convert():
    src = """
    motor m {
        @register(R1) u32* out;
        u32 a;
        u32 b;
        out[0] = a * b;
    }
    transmission t {
        grid (1, 1, 1); block (1, 1, 1);
        motor m { out u32* out; }
    }
    """
    ptx = _compile(src)
    assert 'cvt.rn.f32' not in ptx
    assert 'mul.lo.u32' in ptx
    assert 'st.global.u32' in ptx


def test_printer_refuses_the_exact_illegal_string():
    # The IR op is still the mixed form. The printer has to convert anyway.
    ir = MotorIR(name='hand')
    ir.reg_types['%r0'] = '.u32'
    ir.reg_types['%f1'] = '.f32'
    ir.ops = [BinOpIR(dest='%f1', left='%r0', op='*', right='0.001953', ptx_type='.f32')]
    text = '\n'.join(PTXEmitter()._emit_from_ir(ir, MotorContract('hand', []), {}, set()))
    assert 'mul.f32 %f1, %r0, 0.001953;' not in text
    assert 'cvt.rn.f32.u32 %f2, %r0;' in text
    assert 'mul.f32 %f1, %f2, 0.001953;' in text


@pytest.mark.skipif(PTXAS is None, reason='ptxas not installed')
def test_ptxas_accepts_the_scaled_kernel():
    result = _assemble(_compile(SCALE))
    assert result.returncode == 0, result.stderr
