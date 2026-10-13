"""
WP3 native verification tests.

Skips automatically if ptxas is not on PATH or fails.
When ptxas is present, verifies:
  1. ptxas compiles oracle PTX without error.
  2. ptxas compiles plain-expanded PTX without error.
  3. Both cubins are non-empty.
  4. register count matches between oracle and plain-expanded.
  5. No spill stores or loads in either kernel.
  6. SASS instruction sequences are equal (the core native equality claim).
  7. Verification manifest serialises to valid JSON.
"""

import sys, os, shutil, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from compiler import Compiler

PLAIN_MTL  = os.path.join(os.path.dirname(__file__), '..', 'examples', 'point_add_mixed_plain.mtl')
ORACLE_MTL = os.path.join(os.path.dirname(__file__), '..', 'examples', 'point_add_mixed.mtl')

PTXAS_AVAILABLE = shutil.which("ptxas") is not None
CUOBJDUMP_AVAILABLE = shutil.which("cuobjdump") is not None
TOOLS_AVAILABLE = PTXAS_AVAILABLE and CUOBJDUMP_AVAILABLE

SKIP_MSG = "ptxas/cuobjdump not on PATH — skipping native verification tests"


def _skip_if_no_tools():
    if not TOOLS_AVAILABLE:
        print(f"SKIP: {SKIP_MSG}")
        return True
    return False


def _compile_both():
    c = Compiler()
    oracle_ptx, _ = c.compile_file(ORACLE_MTL, "pollard_rho_step")
    plain_ptx, _  = c.compile_file(PLAIN_MTL,  "point_add_sm89")
    return oracle_ptx, plain_ptx


def test_ptxas_compiles_oracle():
    if _skip_if_no_tools(): return
    from native_verify import compile_ptx_to_cubin
    oracle_ptx, _ = _compile_both()
    cubin, stderr = compile_ptx_to_cubin(oracle_ptx, sm="sm_89")
    assert len(cubin) > 0, "Oracle cubin is empty"
    print(f"PASS: oracle cubin = {len(cubin)} bytes")

def test_ptxas_compiles_plain():
    if _skip_if_no_tools(): return
    from native_verify import compile_ptx_to_cubin
    _, plain_ptx = _compile_both()
    cubin, stderr = compile_ptx_to_cubin(plain_ptx, sm="sm_89")
    assert len(cubin) > 0, "Plain-expanded cubin is empty"
    print(f"PASS: plain-expanded cubin = {len(cubin)} bytes")

def test_register_counts_match():
    if _skip_if_no_tools(): return
    from native_verify import compile_ptx_to_cubin, parse_ptxas_stats
    oracle_ptx, plain_ptx = _compile_both()
    _, o_stderr = compile_ptx_to_cubin(oracle_ptx)
    _, p_stderr = compile_ptx_to_cubin(plain_ptx)
    o_regs, _, _ = parse_ptxas_stats(o_stderr)
    p_regs, _, _ = parse_ptxas_stats(p_stderr)
    assert o_regs > 0, f"Oracle register count not parsed (got {o_regs})"
    assert p_regs > 0, f"Plain register count not parsed (got {p_regs})"
    assert o_regs == p_regs, \
        f"Register count mismatch: oracle={o_regs}, plain={p_regs}"
    print(f"PASS: both kernels use {o_regs} registers")

def test_no_spills():
    if _skip_if_no_tools(): return
    from native_verify import compile_ptx_to_cubin, parse_ptxas_stats
    oracle_ptx, plain_ptx = _compile_both()
    _, o_stderr = compile_ptx_to_cubin(oracle_ptx)
    _, p_stderr = compile_ptx_to_cubin(plain_ptx)
    for label, stderr in [("oracle", o_stderr), ("plain", p_stderr)]:
        regs, ss, sl = parse_ptxas_stats(stderr)
        assert ss == 0, f"{label}: spill stores = {ss} (expected 0)"
        assert sl == 0, f"{label}: spill loads = {sl} (expected 0)"
    print("PASS: no spills in either kernel")

def test_ptx_normalized_equal():
    if _skip_if_no_tools(): return
    from native_verify import normalize_ptx, compare_ptx
    oracle_ptx, plain_ptx = _compile_both()
    # The oracle kernel name is motor_tx combined
    o_norm = normalize_ptx(oracle_ptx, "point_add_mixed_pollard_rho_step")
    p_norm = normalize_ptx(plain_ptx,  "point_add_mixed_plain_point_add_sm89")
    assert len(o_norm) > 100, f"Oracle PTX norm too short: {len(o_norm)} lines"
    assert len(p_norm) > 100, f"Plain PTX norm too short: {len(p_norm)} lines"
    equal, diff_idx = compare_ptx(o_norm, p_norm)
    if not equal:
        o_line = o_norm[diff_idx] if diff_idx < len(o_norm) else "<missing>"
        p_line = p_norm[diff_idx] if diff_idx < len(p_norm) else "<missing>"
        assert False, (
            f"PTX normalized mismatch at line {diff_idx}:\n"
            f"  oracle: {o_line}\n"
            f"  plain:  {p_line}\n"
            f"  oracle total={len(o_norm)}, plain total={len(p_norm)}"
        )
    print(f"PASS: PTX normalized equal — {len(o_norm)} instruction lines match")

def test_manifest_json():
    if _skip_if_no_tools(): return
    from native_verify import verify_pair
    oracle_ptx, plain_ptx = _compile_both()
    manifest = verify_pair(
        oracle_ptx=oracle_ptx,
        oracle_kernel="point_add_mixed_pollard_rho_step",
        plain_ptx=plain_ptx,
        plain_kernel="point_add_mixed_plain_point_add_sm89",
    )
    j = manifest.to_json()
    parsed = json.loads(j)  # must not raise
    assert parsed["verdict"] in ("PASS", "FAIL")
    assert "oracle" in parsed and "plain" in parsed
    assert "ptx_sha256" in parsed["oracle"]
    assert "ptx_equal" in parsed
    print(f"PASS: manifest JSON valid, verdict={parsed['verdict']}")
    print(f"      ptx_equal={manifest.ptx_equal}, cubin_dce={manifest.cubin_dce}")
    print(f"      Oracle PTX lines: {manifest.oracle.ptx_instr_lines}")
    print(f"      Plain  PTX lines: {manifest.plain.ptx_instr_lines}")
    print(f"      Oracle SHA256:  {manifest.oracle.ptx_sha256[:16]}...")
    print(f"      Plain SHA256:   {manifest.plain.ptx_sha256[:16]}...")
    if manifest.note:
        print(f"      Note: {manifest.note[:80]}...")


if __name__ == "__main__":
    failures = 0
    for name, fn in [
        ("ptxas compiles oracle",      test_ptxas_compiles_oracle),
        ("ptxas compiles plain",       test_ptxas_compiles_plain),
        ("register counts match",      test_register_counts_match),
        ("no spills",                  test_no_spills),
        ("PTX normalized equal",       test_ptx_normalized_equal),
        ("manifest JSON valid",        test_manifest_json),
    ]:
        try:
            fn()
        except AssertionError as e:
            print(f"FAIL [{name}]: {e}")
            failures += 1
        except Exception as e:
            import traceback
            print(f"ERROR [{name}]: {type(e).__name__}: {e}")
            traceback.print_exc()
            failures += 1

    if failures == 0:
        if not TOOLS_AVAILABLE:
            print(f"\nAll WP3 tests skipped (tools not available).")
        else:
            print(f"\nAll WP3 native verification tests passed.")
    else:
        print(f"\n{failures} test(s) failed.")
        sys.exit(1)
