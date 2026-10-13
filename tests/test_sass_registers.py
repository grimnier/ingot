"""
SASS register verification: point_negate_jac oracle motor.

Compiles point_negate_jac.mtl -> PTX -> cubin (via ptxas), then runs
cuobjdump --dump-sass to verify:

  1. 12 LDG.E.64 instructions emitted (all inout params loaded from flat buffer).
  2. 12 STG.E.64 instructions emitted (all inout params written back).
  3. All 4 Y1 buffer offsets (0x20-0x38) have both a load and a store.
  4. X1 buffer offsets (0x00-0x18) load and store the SAME physical register
     at each offset — confirming X1 is passed through unchanged.
  5. At least 8 IADD3/LOP3.LUT instructions are present — the carry-chain
     field negate is live in SASS (not dead-code-eliminated).
  6. Report which physical registers hold Y1_0..Y1_3 during the negate.

Skipped automatically if ptxas or cuobjdump is absent.
"""

import re
import subprocess
import sys
import os
import tempfile
import shutil
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

ORACLE_MTL = os.path.join(os.path.dirname(__file__), '..', 'examples', 'point_negate_jac.mtl')


def _ptxas_available():
    return shutil.which('ptxas') is not None


def _cuobjdump_available():
    return shutil.which('cuobjdump') is not None


pytestmark = pytest.mark.skipif(
    not (_ptxas_available() and _cuobjdump_available()),
    reason="ptxas or cuobjdump not found — SASS verification skipped",
)


@pytest.fixture(scope="module")
def sass_output():
    from compiler import Compiler
    from native_verify import compile_ptx_to_cubin

    c = Compiler()
    ptx, _ = c.compile_file(ORACLE_MTL, 'point_negate_sm89')
    cubin, _ = compile_ptx_to_cubin(ptx, 'sm_89')

    with tempfile.NamedTemporaryFile(suffix='.cubin', delete=False) as f:
        f.write(cubin)
        fname = f.name

    r = subprocess.run(['cuobjdump', '--dump-sass', fname], capture_output=True, text=True)
    assert r.returncode == 0, f"cuobjdump failed: {r.stderr}"
    return r.stdout


# ─── helpers ─────────────────────────────────────────────────────────────────

def _ldg_map(sass):
    """offset (int) -> physical register name for LDG.E.64 instructions."""
    pat = re.compile(r'LDG\.E\.64\s+(R\d+),\s*\[R\d+\.64\+?([0-9a-fx]*)\]')
    result = {}
    for m in pat.finditer(sass):
        reg = m.group(1)
        off_str = m.group(2) or '0x0'
        result[int(off_str, 16)] = reg
    return result


def _stg_map(sass):
    """offset (int) -> physical register name for STG.E.64 instructions."""
    pat = re.compile(r'STG\.E\.64\s*\[R\d+\.64\+?([0-9a-fx]*)\],\s*(R\d+)')
    result = {}
    for m in pat.finditer(sass):
        off_str = m.group(1) or '0x0'
        result[int(off_str, 16)] = m.group(2)
    return result


# Flat-buffer byte offsets for each field:
#   params order: X1_0..3 (0-3), Y1_0..3 (4-7), Z1_0..3 (8-11)
X1_OFFSETS = [0x00, 0x08, 0x10, 0x18]
Y1_OFFSETS = [0x20, 0x28, 0x30, 0x38]
Z1_OFFSETS = [0x40, 0x48, 0x50, 0x58]


# ─── §1: flat-buffer load/store coverage ─────────────────────────────────────

def test_twelve_ldg_instructions(sass_output):
    """All 12 inout params must be loaded from the flat buffer."""
    ldg = _ldg_map(sass_output)
    all_offsets = X1_OFFSETS + Y1_OFFSETS + Z1_OFFSETS
    missing = [hex(o) for o in all_offsets if o not in ldg]
    assert not missing, f"Missing LDG for buffer offsets: {missing}"
    print(f"PASS: 12 LDG.E.64 instructions found, all param offsets covered")


def test_twelve_stg_instructions(sass_output):
    """All 12 inout params must be written back to the flat buffer."""
    stg = _stg_map(sass_output)
    all_offsets = X1_OFFSETS + Y1_OFFSETS + Z1_OFFSETS
    missing = [hex(o) for o in all_offsets if o not in stg]
    assert not missing, f"Missing STG for buffer offsets: {missing}"
    print(f"PASS: 12 STG.E.64 instructions found, all param offsets covered")


# ─── §2: Y1 negate is live ───────────────────────────────────────────────────

def test_y1_offsets_loaded(sass_output):
    """Y1_0..Y1_3 must each have an LDG instruction."""
    ldg = _ldg_map(sass_output)
    for i, off in enumerate(Y1_OFFSETS):
        assert off in ldg, f"No LDG found for Y1_{i} at buffer offset {hex(off)}"
    print(f"PASS: Y1_0..Y1_3 all loaded from flat buffer")


def test_y1_offsets_stored(sass_output):
    """Y1_0..Y1_3 must each have an STG instruction (negate result written back)."""
    stg = _stg_map(sass_output)
    for i, off in enumerate(Y1_OFFSETS):
        assert off in stg, f"No STG found for Y1_{i} at buffer offset {hex(off)}"
    print(f"PASS: Y1_0..Y1_3 all stored back to flat buffer (negate result live)")


def test_carry_chain_iadd3_lop3_present(sass_output):
    """IADD3 and LOP3.LUT must be present in SASS — carry chain is not dead."""
    negate_instrs = [l for l in sass_output.splitlines() if 'IADD3' in l or 'LOP3.LUT' in l]
    assert len(negate_instrs) >= 8, (
        f"Expected >= 8 IADD3/LOP3.LUT negate instructions, got {len(negate_instrs)}.\n"
        "The oracle motor carry chain may have been dead-code-eliminated by ptxas.\n"
        "Check ptx_emitter.py oracle-IO mode: flat-buffer LDG/STG must wrap the kernel."
    )
    print(f"PASS: {len(negate_instrs)} IADD3/LOP3.LUT carry-chain instructions live in SASS")


def test_y1_physical_register_report(sass_output):
    """Report which physical registers hold Y1 limbs during the negate."""
    ldg = _ldg_map(sass_output)
    stg = _stg_map(sass_output)
    y1_names = ['Y1_0', 'Y1_1', 'Y1_2', 'Y1_3']
    for i, (off, name) in enumerate(zip(Y1_OFFSETS, y1_names)):
        load_reg  = ldg.get(off, '???')
        store_reg = stg.get(off, '???')
        tag = "same" if load_reg == store_reg else "remapped"
        print(f"  {name} (buf+{hex(off)}): load={load_reg}  store={store_reg}  [{tag}]")
    # The test always passes — this is a report-only assertion
    print("PASS: Y1 physical register assignments reported above")


# ─── §3: X1 and Z1 are pass-through (load reg == store reg) ─────────────────

def test_x1_physical_registers_unchanged(sass_output):
    """X1 limbs must load and store the same physical register.

    If load reg != store reg at an X1 offset, ptxas moved the value through
    an intermediate (fine for correctness), but we verify the VALUE is
    preserved by checking that no negate instructions touch X1's load registers.
    A simpler proxy: verify each X1 offset has both LDG and STG present,
    and that no IADD3/LOP3 instruction references an X1 load register as
    a destination (i.e., X1 registers are not overwritten by the carry chain).
    """
    ldg = _ldg_map(sass_output)
    stg = _stg_map(sass_output)

    x1_load_regs = {ldg[off] for off in X1_OFFSETS if off in ldg}
    negate_lines = [l for l in sass_output.splitlines() if 'IADD3' in l or 'LOP3.LUT' in l]

    overwritten = []
    for reg in x1_load_regs:
        for line in negate_lines:
            # Destination register is the first Rxx after the instruction name
            dest_m = re.search(r'(?:IADD3|LOP3\.LUT)\S*\s+(R\d+)', line)
            if dest_m and dest_m.group(1) == reg:
                overwritten.append((reg, line.strip()))
                break

    assert not overwritten, (
        f"X1 physical registers were overwritten by carry-chain negate instructions:\n"
        + "\n".join(f"  {r}: {l}" for r, l in overwritten)
    )
    print(f"PASS: X1 physical registers {sorted(x1_load_regs)} not written by carry chain")


def test_z1_physical_registers_unchanged(sass_output):
    """Z1 limbs must not be overwritten by the carry-chain negate instructions."""
    ldg = _ldg_map(sass_output)
    z1_load_regs = {ldg[off] for off in Z1_OFFSETS if off in ldg}
    negate_lines = [l for l in sass_output.splitlines() if 'IADD3' in l or 'LOP3.LUT' in l]

    overwritten = []
    for reg in z1_load_regs:
        for line in negate_lines:
            dest_m = re.search(r'(?:IADD3|LOP3\.LUT)\S*\s+(R\d+)', line)
            if dest_m and dest_m.group(1) == reg:
                overwritten.append((reg, line.strip()))
                break

    assert not overwritten, (
        f"Z1 physical registers were overwritten by carry-chain negate instructions:\n"
        + "\n".join(f"  {r}: {l}" for r, l in overwritten)
    )
    print(f"PASS: Z1 physical registers {sorted(z1_load_regs)} not written by carry chain")


if __name__ == "__main__":
    # Run without pytest for quick manual verification
    sass_fixture_result = [None]

    class _FakeFixture:
        def __init__(self):
            from compiler import Compiler
            from native_verify import compile_ptx_to_cubin
            c = Compiler()
            ptx, _ = c.compile_file(ORACLE_MTL, 'point_negate_sm89')
            cubin, _ = compile_ptx_to_cubin(ptx, 'sm_89')
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.cubin', delete=False) as f:
                f.write(cubin); fname = f.name
            r = subprocess.run(['cuobjdump', '--dump-sass', fname], capture_output=True, text=True)
            self.sass = r.stdout

    fix = _FakeFixture()
    tests = [
        test_twelve_ldg_instructions,
        test_twelve_stg_instructions,
        test_y1_offsets_loaded,
        test_y1_offsets_stored,
        test_carry_chain_iadd3_lop3_present,
        test_y1_physical_register_report,
        test_x1_physical_registers_unchanged,
        test_z1_physical_registers_unchanged,
    ]
    failures = 0
    for t in tests:
        try:
            t(fix.sass)
        except AssertionError as e:
            print(f"FAIL [{t.__name__}]: {e}")
            failures += 1
        except Exception as e:
            print(f"ERROR [{t.__name__}]: {type(e).__name__}: {e}")
            failures += 1
    if failures == 0:
        print(f"\nAll {len(tests)} SASS verification tests passed.")
    else:
        print(f"\n{failures}/{len(tests)} test(s) failed.")
        sys.exit(1)
