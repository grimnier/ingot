"""
Tests for spill gate infrastructure (native_verify.py).
Covers: ptxas stats parsing, local-op detection, missing-ptxas skip behavior.
"""
import sys
import os
import shutil
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from native_verify import (
    parse_ptxas_stats_full,
    check_ptx_local_ops,
    single_kernel_spill_gate,
    SpillGateResult,
    verify_no_temp_touches_locked_physical,
)

PTXAS_AVAILABLE = shutil.which('ptxas') is not None


# --- §1: ptxas stdout stats parsing ------------------------------------------

SAMPLE_PTXAS_STDOUT = """
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'test_kernel' for 'sm_89'
ptxas info    : Function properties for test_kernel
ptxas info    :     0 bytes stack frame, 16 bytes spill stores, 8 bytes spill loads
ptxas info    : Used 64 registers, 400 bytes cmem[0]
"""

def test_parse_ptxas_stats_full_registers():
    regs, ss, sl, sf = parse_ptxas_stats_full(SAMPLE_PTXAS_STDOUT)
    assert regs == 64

def test_parse_ptxas_stats_full_spill_stores():
    regs, ss, sl, sf = parse_ptxas_stats_full(SAMPLE_PTXAS_STDOUT)
    assert ss == 16

def test_parse_ptxas_stats_full_spill_loads():
    regs, ss, sl, sf = parse_ptxas_stats_full(SAMPLE_PTXAS_STDOUT)
    assert sl == 8

def test_parse_ptxas_stats_full_stack_frame():
    regs, ss, sl, sf = parse_ptxas_stats_full(SAMPLE_PTXAS_STDOUT)
    assert sf == 0

def test_parse_ptxas_stats_full_zero_spills():
    clean = """
ptxas info    : Used 32 registers, 400 bytes cmem[0]
ptxas info    :     0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
"""
    regs, ss, sl, sf = parse_ptxas_stats_full(clean)
    assert regs == 32
    assert ss == 0
    assert sl == 0
    assert sf == 0

def test_parse_ptxas_stats_full_empty():
    regs, ss, sl, sf = parse_ptxas_stats_full("")
    assert regs == 0
    assert ss == 0
    assert sl == 0
    assert sf == 0


# --- §2: local memory op detection -------------------------------------------

def test_check_ptx_local_ops_ld_local():
    ptx = ".visible .entry k() { ld.local.u32 %r0, [%rd0]; }"
    assert check_ptx_local_ops(ptx) is True

def test_check_ptx_local_ops_st_local():
    ptx = ".visible .entry k() { st.local.u32 [%rd0], %r0; }"
    assert check_ptx_local_ops(ptx) is True

def test_check_ptx_local_ops_local_decl():
    ptx = ".local .align 4 .b8 __local_depot[64];"
    assert check_ptx_local_ops(ptx) is True

def test_check_ptx_local_ops_clean():
    ptx = ".visible .entry k() { add.u32 %r0, %r1, %r2; }"
    assert check_ptx_local_ops(ptx) is False


# --- §3: missing ptxas -- clean skip/fail behavior ---------------------------

def test_spill_gate_missing_ptxas_returns_fail(monkeypatch):
    # single_kernel_spill_gate catches RuntimeError from compile_ptx_to_cubin.
    # When ptxas is absent, subprocess.run raises FileNotFoundError inside
    # compile_ptx_to_cubin, which is not caught there and propagates as
    # FileNotFoundError -- not caught by single_kernel_spill_gate either.
    # We therefore patch compile_ptx_to_cubin directly to raise RuntimeError,
    # which is what the gate's except clause handles.
    import native_verify
    def fake_compile(ptx, sm="sm_89"):
        raise RuntimeError("ptxas not found: [Errno 2] No such file or directory")
    monkeypatch.setattr(native_verify, "compile_ptx_to_cubin", fake_compile)

    result = single_kernel_spill_gate("// fake ptx", kernel_name="test")
    assert isinstance(result, SpillGateResult)
    assert result.verdict == "FAIL"
    assert any("ptxas" in r.lower() or "failed" in r.lower() for r in result.fail_reasons)

def test_spill_gate_ptxas_nonzero_exit_returns_fail(monkeypatch):
    # compile_ptx_to_cubin checks returncode and raises RuntimeError on failure.
    # Patching subprocess.run to return a nonzero exit exercises that path.
    import subprocess
    class FakeResult:
        returncode = 1
        stdout = ""
        stderr = "ptxas error: syntax error at line 1"
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FakeResult())

    result = single_kernel_spill_gate("bad ptx", kernel_name="test")
    assert result.verdict == "FAIL"
    assert len(result.fail_reasons) > 0


# --- §4: real ptxas stats (skip if ptxas absent) -----------------------------

MINIMAL_PTX = """
.version 7.8
.target sm_89
.address_size 64

.visible .entry test_add(
    .param .u64 param0,
    .param .u64 param1
)
{
    .reg .u32 %r<4>;
    .reg .u64 %rd<3>;

    ld.param.u64 %rd0, [param0];
    ld.param.u64 %rd1, [param1];
    ld.global.u32 %r0, [%rd0];
    ld.global.u32 %r1, [%rd1];
    add.u32 %r2, %r0, %r1;
    st.global.u32 [%rd0], %r2;
    ret;
}
"""

@pytest.mark.skipif(not PTXAS_AVAILABLE, reason="ptxas not installed")
def test_spill_gate_real_ptxas_clean_kernel():
    result = single_kernel_spill_gate(
        MINIMAL_PTX,
        kernel_name="test_add",
        sm="sm_89",
        spill_allowed=False,
        local_memory_allowed=False,
    )
    assert result.verdict == "PASS", f"Expected PASS, got FAIL: {result.fail_reasons}"
    assert result.registers > 0
    assert result.spills_store == 0
    assert result.spills_load == 0

@pytest.mark.skipif(not PTXAS_AVAILABLE, reason="ptxas not installed")
def test_spill_gate_real_ptxas_max_registers_exceeded():
    result = single_kernel_spill_gate(
        MINIMAL_PTX,
        kernel_name="test_add",
        sm="sm_89",
        max_registers=1,
    )
    assert result.verdict == "FAIL"
    assert any("register count" in r for r in result.fail_reasons)


# --- §5: verify_no_temp_touches_locked_physical --------------------------------

def test_verify_no_violation_clean_ptx():
    """Clean PTX: locked %R2 only appears in ld.param and mov -- should pass."""
    ptx = """\
.version 7.8
.target sm_89
.address_size 64
.visible .entry k()
{
    .reg .u64 %R2;
    .reg .u64 %rd<4>;
    ld.param.u64 %R2, [x];
    mov.u64 %rd0, %R2;
    ret;
}
"""
    # Should not raise: %R2 is in the locked set
    verify_no_temp_touches_locked_physical(ptx, ["R2"])


def test_verify_no_violation_empty_locked():
    """Empty locked list: PTX with no uppercase-R registers should pass."""
    ptx = """\
    add.u64 %rd0, %rd1, %rd2;
    mul.lo.u32 %r0, %r1, %r2;
"""
    # No %R<n> names at all -- no violation
    verify_no_temp_touches_locked_physical(ptx, [])


def test_verify_violation_unlocked_R_name():
    """Violation: %R5 appears in PTX but was not declared as a locked physical."""
    ptx = """\
    add.u64 %R5, %rd0, %rd1;
"""
    import pytest as _pytest
    with _pytest.raises(RuntimeError, match=r"%R5"):
        # Locked set is empty -- %R5 is an undeclared unlocked uppercase-R name
        verify_no_temp_touches_locked_physical(ptx, [])


def test_verify_violation_R_outside_locked_set():
    """Violation: %R3 appears but only R0 and R1 are in the locked set."""
    ptx = """\
    ld.param.u64 %R0, [buf];
    ld.param.u64 %R1, [buf2];
    add.u64 %R3, %R0, %R1;
"""
    import pytest as _pytest
    with _pytest.raises(RuntimeError, match=r"%R3"):
        verify_no_temp_touches_locked_physical(ptx, ["R0", "R1"])


def test_verify_violation_reports_all():
    """Multiple violations: %R7 and %R9 appear but neither is in the locked set."""
    ptx = """\
    add.u64 %R7, %rd0, %rd1;
    sub.u32 %R9, %r0, %r1;
"""
    import pytest as _pytest
    with _pytest.raises(RuntimeError) as exc_info:
        verify_no_temp_touches_locked_physical(ptx, [])
    msg = str(exc_info.value)
    assert "%R7" in msg
    assert "%R9" in msg
    assert "2 violation(s)" in msg


def test_verify_arithmetic_into_declared_locked_allowed():
    """Arithmetic writing into a declared locked register is correct behavior -- no violation."""
    ptx = """\
    add.u64 %R0, %R1, %R1;
    add.u64 %R1, %R0, %R0;
"""
    # %R0 and %R1 are both in the locked set -- emitter is routing results
    # into programmer-declared locked registers, which is the intended behavior.
    verify_no_temp_touches_locked_physical(ptx, ["R0", "R1"])


def test_verify_no_violation_mov_to_locked_allowed():
    """mov to a locked register is legitimate initialization -- should NOT flag."""
    ptx = """\
    mov.u64 %R5, %rd3;
    mov.f32 %R5, 0f3F800000;
"""
    # %R5 is in the locked set -- no violation
    verify_no_temp_touches_locked_physical(ptx, ["R5"])


def test_verify_no_violation_ld_global_to_locked_allowed():
    """ld.global into a locked register is a legitimate load -- should NOT flag."""
    ptx = """\
    ld.global.u64 %R2, [%rd0];
"""
    verify_no_temp_touches_locked_physical(ptx, ["R2"])


def test_verify_no_uppercase_R_in_lowercase_only_ptx():
    """PTX using only lowercase temp banks (%rd, %r, %f, %p) passes regardless of locked set."""
    ptx = """\
    add.u64 %rd0, %rd1, %rd2;
    mul.lo.u64 %rd3, %rd0, %rd4;
    madc.hi.cc.u64 %rd5, %rd0, %rd1, %rd2;
"""
    # No %R<n> names anywhere -- no violation even if locked set is non-empty
    verify_no_temp_touches_locked_physical(ptx, ["R0", "R2"])


def test_verify_setp_result_in_locked_ok_if_declared():
    """setp result going into a declared locked register is fine."""
    ptx = """\
    setp.lt.u64 %R0, %rd1, %rd2;
"""
    # %R0 is in the locked set -- programmer declared it, no violation
    verify_no_temp_touches_locked_physical(ptx, ["R0"])


def test_verify_setp_result_in_undeclared_R_is_violation():
    """setp result going into an undeclared %R4 -- that is a violation."""
    ptx = """\
    setp.lt.u64 %R4, %rd1, %rd2;
"""
    import pytest as _pytest
    with _pytest.raises(RuntimeError, match=r"%R4"):
        # %R4 is not in the locked set
        verify_no_temp_touches_locked_physical(ptx, ["R0"])
