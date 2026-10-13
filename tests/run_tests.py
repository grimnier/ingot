"""
Ingot Negative Test Runner (Phase 1)

All files in must_not_compile/*.mtl MUST fail compilation with a clear semantic or parse error.
This proves the contract checker, contamination rules, and register escape detection are working.
"""

import sys
from pathlib import Path

# Robust import whether run as module or script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
try:
    from compiler import Compiler, CompileError
except ImportError:
    # fallback when running directly
    import importlib.util
    spec = importlib.util.spec_from_file_location("compiler", Path(__file__).parent.parent / "src" / "compiler.py")
    compiler_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compiler_mod)
    Compiler = compiler_mod.Compiler
    CompileError = compiler_mod.CompileError

TEST_DIR = Path(__file__).parent / "must_not_compile"

def main():
    failures = 0
    c = Compiler()

    for test_file in sorted(TEST_DIR.glob("*.mtl")):
        print(f"Testing {test_file.name} ... ", end="", flush=True)
        try:
            c.compile_file(test_file)
            print("UNEXPECTED PASS (should have been rejected)")
            failures += 1
        except CompileError as e:
            # Good - it correctly rejected the bad program
            print("CORRECTLY REJECTED")
            # Uncomment for debugging:
            # print("   ", str(e)[:120])
        except Exception as e:
            print(f"CRASHED: {type(e).__name__}: {e}")
            failures += 1

    print(f"\n{failures} failures out of {len(list(TEST_DIR.glob('*.mtl')))} negative tests")
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
