import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from compiler import Compiler


def compile_src(src: str):
    return Compiler().compile_source(src)


def test_lop3_builtin_emits_ptx_lop3():
    src = """
motor m {
    @register(R0) u32 out;
    u32 a;
    u32 b;
    u32 c;
    a = 1;
    b = 2;
    c = 3;
    out = lop3(a, b, c, 0xCA);
}
transmission t { grid(1,1,1); block(256,1,1); motor m {} }
"""
    ptx, _ = compile_src(src)
    assert "lop3.b32" in ptx
    assert ", 202;" in ptx


def test_hash160_u32_emits_lop3_boolean_rounds():
    example = Path(__file__).parent.parent / "examples" / "hash160_u32.mtl"
    ptx, _ = Compiler().compile_file(example)
    assert ptx.count("lop3.b32") >= 60
