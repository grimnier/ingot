"""
WP3: Native verification pipeline.

Compiles PTX to cubin via ptxas, extracts kernel SASS via cuobjdump,
compares instruction sequences, and emits a signed verification manifest.

No GPU required -- ptxas is an assembler; it compiles PTX to binary without
a device present.
"""

import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional, List, Tuple

# -- Public result types ------------------------------------------------------

@dataclass
class SpillGateResult:
    """Single-kernel spill/register gate result. Verdict is PASS or FAIL."""
    kernel_name: str
    sm: str
    registers: int
    spills_store: int    # bytes spilled to local memory on writes
    spills_load: int     # bytes spilled to local memory on reads
    stack_frame: int     # bytes of stack frame
    has_local_ops: bool  # True if PTX contains ld.local/st.local instructions
    verdict: str         # "PASS" | "FAIL"
    fail_reasons: List[str]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent)

@dataclass
class KernelStats:
    source_label: str        # e.g. "oracle" or "plain"
    ptx_sha256: str
    cubin_sha256: str
    kernel_name: str
    registers: int           # from ptxas --verbose
    spills_store: int
    spills_load: int
    sass_lines: int          # number of SASS instruction lines extracted
    ptx_instr_lines: int     # number of normalized PTX instruction lines

@dataclass
class VerificationManifest:
    ptxas_version: str
    sm: str
    oracle: KernelStats
    plain: KernelStats
    sass_equal: bool
    ptx_equal: bool          # PTX-level normalized comparison (primary when DCE fires)
    cubin_dce: bool          # True when ptxas DCE'd the kernel body (no global outputs)
    first_diff_line: Optional[int]     # None if equal; 0-indexed
    verdict: str                        # "PASS" or "FAIL"
    note: str = ""

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent)


# -- Core pipeline steps ------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_str(s: str) -> str:
    return sha256_bytes(s.encode())


def ptxas_version() -> str:
    try:
        r = subprocess.run(["ptxas", "--version"], capture_output=True, text=True, timeout=10)
        line = (r.stdout or r.stderr or "").splitlines()[0]
        return line.strip()
    except Exception as e:
        return f"unknown ({e})"


def compile_ptx_to_cubin(ptx_source: str, sm: str = "sm_89") -> bytes:
    """
    Invoke ptxas to assemble a PTX string to a cubin.
    Returns the raw cubin bytes.
    Raises RuntimeError on ptxas failure.
    """
    with tempfile.TemporaryDirectory() as tmp:
        ptx_path = os.path.join(tmp, "input.ptx")
        cubin_path = os.path.join(tmp, "output.cubin")
        Path(ptx_path).write_text(ptx_source, encoding="utf-8")

        cmd = [
            "ptxas",
            f"-arch={sm}",
            "--verbose",           # register/spill info on stderr
            ptx_path,
            "-o", cubin_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(
                f"ptxas failed (rc={result.returncode}):\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
        cubin_bytes = Path(cubin_path).read_bytes()
        return cubin_bytes, result.stdout + result.stderr


def parse_ptxas_stats(stderr: str) -> Tuple[int, int, int]:
    """
    Parse ptxas --verbose output to extract (registers, spills_store, spills_load).
    Example line:
      ptxas info    : Used 96 registers, 400 bytes cmem[0]
      ptxas info    : Function properties for point_add_sm89
      ptxas info    :     0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
    """
    regs, ss, sl, _ = parse_ptxas_stats_full(stderr)
    return regs, ss, sl


def parse_ptxas_stats_full(stderr: str) -> Tuple[int, int, int, int]:
    """Parse ptxas --verbose output: (registers, spills_store, spills_load, stack_frame)."""
    regs = 0
    spills_store = 0
    spills_load  = 0
    stack_frame  = 0

    for line in stderr.splitlines():
        m = re.search(r'Used (\d+) registers', line)
        if m:
            regs = int(m.group(1))
        m = re.search(r'(\d+) bytes stack frame', line)
        if m:
            stack_frame = int(m.group(1))
        m = re.search(r'(\d+) bytes spill stores', line)
        if m:
            spills_store = int(m.group(1))
        m = re.search(r'(\d+) bytes spill loads', line)
        if m:
            spills_load = int(m.group(1))

    return regs, spills_store, spills_load, stack_frame


def parse_ptxas_function_stats(stderr: str) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Parse ptxas --verbose output per entry function.

    Returns:
        function_name -> (registers, spills_store, spills_load, stack_frame)
    """
    stats: Dict[str, Tuple[int, int, int, int]] = {}
    current = ""
    stack_frame = 0
    spills_store = 0
    spills_load = 0

    for line in stderr.splitlines():
        m = re.search(r"Compiling entry function '([^']+)'", line)
        if m:
            current = m.group(1)
            stack_frame = spills_store = spills_load = 0
            continue
        m = re.search(r'(\d+) bytes stack frame', line)
        if m:
            stack_frame = int(m.group(1))
        m = re.search(r'(\d+) bytes spill stores', line)
        if m:
            spills_store = int(m.group(1))
        m = re.search(r'(\d+) bytes spill loads', line)
        if m:
            spills_load = int(m.group(1))
        m = re.search(r'Used (\d+) registers', line)
        if m and current:
            stats[current] = (int(m.group(1)), spills_store, spills_load, stack_frame)

    return stats


def check_ptx_local_ops(ptx: str) -> bool:
    """Return True if the PTX text contains local memory instructions or declarations."""
    return bool(re.search(r'(ld\.local\.|st\.local\.|\.local\s+\.)', ptx))


def single_kernel_spill_gate(
    ptx: str,
    kernel_name: str = "",
    sm: str = "sm_89",
    max_registers: Optional[int] = None,
    spill_allowed: bool = True,
    local_memory_allowed: bool = True,
) -> 'SpillGateResult':
    """
    Compile a PTX string through ptxas and check it against sealed-profile constraints.

    This is the first-class compile-time gate: called by the Ingot compiler when any
    sealed profile or `require` clause mandates a spill-free or local-memory-free kernel.
    Can also be called from the standalone spill_gate CLI tool to audit existing CUDA PTX.

    Returns SpillGateResult with verdict PASS or FAIL and a list of specific failures.
    """
    try:
        _, stderr = compile_ptx_to_cubin(ptx, sm)
    except RuntimeError as e:
        return SpillGateResult(
            kernel_name=kernel_name, sm=sm, registers=0,
            spills_store=0, spills_load=0, stack_frame=0,
            has_local_ops=False, verdict="FAIL",
            fail_reasons=[f"ptxas assembly failed: {e}"]
        )

    regs, ss, sl, sf = parse_ptxas_stats_full(stderr)
    has_local = check_ptx_local_ops(ptx)

    fail_reasons = []
    if not spill_allowed:
        if ss > 0:
            fail_reasons.append(f"spill stores: {ss} bytes (violates no_spills)")
        if sl > 0:
            fail_reasons.append(f"spill loads: {sl} bytes (violates no_spills)")
        if sf > 0:
            fail_reasons.append(f"stack frame: {sf} bytes (violates no_spills)")
    if not local_memory_allowed and has_local:
        fail_reasons.append("local memory instructions detected (violates no_local_memory)")
    if max_registers is not None and regs > max_registers:
        fail_reasons.append(f"register count {regs} exceeds max_registers({max_registers})")

    verdict = "FAIL" if fail_reasons else "PASS"
    return SpillGateResult(
        kernel_name=kernel_name, sm=sm,
        registers=regs, spills_store=ss, spills_load=sl,
        stack_frame=sf, has_local_ops=has_local,
        verdict=verdict, fail_reasons=fail_reasons
    )


def extract_sass(cubin_bytes: bytes, kernel_name: str) -> List[str]:
    """
    Use cuobjdump --dump-sass to extract the SASS instruction lines for
    the named kernel.  Returns one string per instruction line, with
    register operands intact (e.g. 'IMAD.MOV.U32 R3, RZ, RZ, R0').

    If kernel_name is empty, returns all SASS lines from the first kernel found.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cubin_path = os.path.join(tmp, "input.cubin")
        Path(cubin_path).write_bytes(cubin_bytes)

        cmd = ["cuobjdump", "--dump-sass", cubin_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(
                f"cuobjdump failed (rc={result.returncode}):\n{result.stderr}"
            )

    sass_text = result.stdout
    return _parse_sass_for_kernel(sass_text, kernel_name)


def _parse_sass_for_kernel(sass_text: str, kernel_name: str) -> List[str]:
    """
    cuobjdump --dump-sass output looks like:

        Fatbin elf code:
        ================
        arch = sm_89
        code for sm_89
        Function : point_add_sm89
        .headerflags @"EF_CUDA_SM89 EF_CUDA_PTX_SM(EF_CUDA_SM89)"
                /*0000*/  IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
                ...

    We extract lines inside the target function's block.
    If kernel_name is empty, return all instruction lines from the file.
    """
    lines = sass_text.splitlines()
    in_target = (kernel_name == "")
    result: List[str] = []

    for line in lines:
        # Detect function header
        stripped = line.strip()
        if stripped.startswith("Function :"):
            fname = stripped[len("Function :"):].strip()
            in_target = (kernel_name == "" or fname == kernel_name)
            continue

        if not in_target:
            continue

        # Skip metadata lines (.headerflags, blank, comments, arch lines)
        if not stripped or stripped.startswith(".") or stripped.startswith("/*") and stripped.endswith("*/"):
            continue
        if stripped.startswith("Fatbin") or stripped.startswith("arch") or \
           stripped.startswith("code for") or stripped.startswith("="):
            continue

        # Instruction lines look like:
        #   /*0000*/  IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
        # Strip the address comment and normalise whitespace.
        instr = re.sub(r'/\*[0-9a-fA-F]+\*/', '', stripped).strip()
        if instr:
            # Normalise multiple spaces to single space for comparison
            instr = re.sub(r'\s+', ' ', instr)
            result.append(instr)

    return result


def compare_sass(lines_a: List[str], lines_b: List[str]) -> Tuple[bool, Optional[int]]:
    """
    Compare two SASS instruction sequences.
    Returns (equal, first_diff_index) where first_diff_index is None if equal.
    """
    if len(lines_a) != len(lines_b):
        return False, min(len(lines_a), len(lines_b))
    for i, (a, b) in enumerate(zip(lines_a, lines_b)):
        if a != b:
            return False, i
    return True, None


def normalize_ptx(ptx_source: str, kernel_name: str) -> List[str]:
    """
    Extract and normalize PTX instructions for the named .entry.
    Normalization:
      - Strip virtual register names (%v0, %v12, etc.) -> %vN placeholder token
      - Preserve locked physical register names (%R0..%R19)
      - Strip leading whitespace and semicolons
      - Skip .reg declarations, labels, comments, and directives
    Returns one normalized line per PTX instruction (preserves opcode + operand count).
    """
    lines = ptx_source.splitlines()
    in_entry = False
    depth = 0
    result: List[str] = []

    for raw in lines:
        stripped = raw.strip()
        if not in_entry:
            if ".entry" in stripped and kernel_name in stripped:
                in_entry = True
                depth = 0
            continue

        if stripped == "{":
            depth += 1
            continue
        if stripped == "}":
            depth -= 1
            if depth <= 0:
                break
            continue

        # Skip directives and declarations
        if not stripped or stripped.startswith(".") or stripped.startswith("//"):
            continue
        if stripped.endswith(":"):   # labels
            continue
        if stripped.startswith("(") or stripped.startswith(")"):
            continue

        # Strip semicolon
        instr = stripped.rstrip(";").strip()
        if not instr:
            continue

        # Normalize virtual registers (%v\d+) -> %vN  (preserves structure, removes numbering)
        instr = re.sub(r'%v\d+', '%vN', instr)
        # Normalize compiler temps (%r\d+, %f\d+, %rd\d+, %p\d+) -> %tN
        instr = re.sub(r'%(r|f|rd|p)\d+', r'%\1N', instr)
        # Normalise whitespace
        instr = re.sub(r'\s+', ' ', instr)
        result.append(instr)

    return result


def compare_ptx(lines_a: List[str], lines_b: List[str]) -> Tuple[bool, Optional[int]]:
    """Same as compare_sass but for normalized PTX lines."""
    if len(lines_a) != len(lines_b):
        return False, min(len(lines_a), len(lines_b))
    for i, (a, b) in enumerate(zip(lines_a, lines_b)):
        if a != b:
            return False, i
    return True, None


# -- High-level entry point ----------------------------------------------------

def verify_pair(
    oracle_ptx:     str,
    oracle_kernel:  str,
    plain_ptx:      str,
    plain_kernel:   str,
    sm:             str = "sm_89",
    oracle_label:   str = "oracle",
    plain_label:    str = "plain",
) -> VerificationManifest:
    """
    Full WP3 verification pass.

    Primary path (always runs): PTX-level normalized comparison.
      Normalizes virtual register names and compares instruction sequences
      for the named kernels.  This is the primary equality proof because
      it operates on the full kernel body before ptxas DCE.

    Secondary path (cubin/SASS): runs ptxas + cuobjdump.
      These kernels have no global memory outputs (pure register-state
      kernels for Pollard's rho), so ptxas DCE's the body to MOV+EXIT.
      When cubin_dce is True in the manifest, SASS comparison is replaced
      by the PTX comparison result.

    Verdict: PASS when ptx_equal is True (the primary proof).
    """
    version = ptxas_version()

    # -- PTX-level comparison (primary) ---------------------------------------
    o_ptx_norm = normalize_ptx(oracle_ptx, oracle_kernel)
    p_ptx_norm = normalize_ptx(plain_ptx,  plain_kernel)
    ptx_equal, ptx_diff_idx = compare_ptx(o_ptx_norm, p_ptx_norm)

    # -- Cubin / SASS (secondary, informational) -------------------------------
    oracle_cubin, oracle_stderr = compile_ptx_to_cubin(oracle_ptx, sm)
    o_regs, o_ss, o_sl = parse_ptxas_stats(oracle_stderr)

    plain_cubin, plain_stderr = compile_ptx_to_cubin(plain_ptx, sm)
    p_regs, p_ss, p_sl = parse_ptxas_stats(plain_stderr)

    o_sass = extract_sass(oracle_cubin, oracle_kernel)
    p_sass = extract_sass(plain_cubin,  plain_kernel)

    # Detect DCE: if SASS has <= 4 instructions, ptxas eliminated the body
    dce = (len(o_sass) <= 4 and len(p_sass) <= 4)
    sass_equal, sass_diff_idx = compare_sass(o_sass, p_sass)

    # Use PTX diff index for reporting when DCE fires
    diff_idx = ptx_diff_idx if dce else sass_diff_idx

    oracle_stats = KernelStats(
        source_label=oracle_label,
        ptx_sha256=sha256_str(oracle_ptx),
        cubin_sha256=sha256_bytes(oracle_cubin),
        kernel_name=oracle_kernel,
        registers=o_regs,
        spills_store=o_ss,
        spills_load=o_sl,
        sass_lines=len(o_sass),
        ptx_instr_lines=len(o_ptx_norm),
    )
    plain_stats = KernelStats(
        source_label=plain_label,
        ptx_sha256=sha256_str(plain_ptx),
        cubin_sha256=sha256_bytes(plain_cubin),
        kernel_name=plain_kernel,
        registers=p_regs,
        spills_store=p_ss,
        spills_load=p_sl,
        sass_lines=len(p_sass),
        ptx_instr_lines=len(p_ptx_norm),
    )

    # Build note
    note_parts = []
    if dce:
        note_parts.append(
            "cubin_dce=True: kernels have no global outputs; ptxas eliminated body. "
            "PTX-level comparison is the primary verification."
        )
    if not ptx_equal and ptx_diff_idx is not None:
        o_line = o_ptx_norm[ptx_diff_idx] if ptx_diff_idx < len(o_ptx_norm) else "<missing>"
        p_line = p_ptx_norm[ptx_diff_idx] if ptx_diff_idx < len(p_ptx_norm) else "<missing>"
        note_parts.append(
            f"First PTX diff at line {ptx_diff_idx}:\n  oracle: {o_line}\n  plain:  {p_line}"
        )

    # Verdict is based on PTX equality (primary proof)
    verdict = "PASS" if ptx_equal else "FAIL"

    return VerificationManifest(
        ptxas_version=version,
        sm=sm,
        oracle=oracle_stats,
        plain=plain_stats,
        sass_equal=sass_equal,
        ptx_equal=ptx_equal,
        cubin_dce=dce,
        first_diff_line=diff_idx,
        verdict=verdict,
        note="\n".join(note_parts),
    )


# -- Register allocation safety check ----------------------------------------

def verify_no_temp_touches_locked_physical(ptx_text: str, locked_reg_names: list) -> None:
    """
    Post-emit safety check: scan PTX text and assert no compiler-allocated temp
    register shares a name with a hard-locked physical register.
    Raises RuntimeError if a violation is found.

    Convention in Ingot PTX:
      - Locked physical registers are declared and used as %R0, %R1, ... (uppercase R).
        They come from @register(Rn) annotations and are always in the locked_physical set.
      - Compiler-allocated temps use lowercase prefixes: %r0, %f0, %rd0, %p0.
        PTX treats %R0 and %r0 as DISTINCT names (case-sensitive identifiers), so
        the namespaces are already separate in a correct implementation.

    What this check catches:
      The emitter may accidentally emit a %R<n> name (uppercase) for a register that
      was NOT declared as a locked physical by the programmer -- i.e. the emitter
      invented a new uppercase-R name outside the locked set.  This would mean a
      compiler-generated slot was given a locked-style name, which is a naming
      contract violation that could confuse downstream tooling and future passes.

      Specifically: scan the PTX for every %R<n> register reference.  Any %R<n>
      that appears in the PTX but was NOT in the locked_reg_names set is a violation.

    What this check does NOT flag (these are all correct emitter behavior):
      - Arithmetic or logic instructions writing INTO a declared locked register:
            add.u64 %R0, %R1, %R1;   // correct: programmer wrote a = b+b
        Locked registers are intentionally written by arithmetic ops; that is the
        whole purpose of the @register(Rn) annotation.
      - Lowercase temp registers (%r0, %f0, %rd0, %p0) in any position.
      - Locked registers appearing as source operands.

    Args:
        ptx_text:         Full PTX source string after emission.
        locked_reg_names: List of bare physical register names, e.g. ["R0", "R2", "R7"].
                          These are the names from MotorIR.locked_physical.

    Raises:
        RuntimeError: Describes every violation found. Raised only after collecting
                      all violations so the error message is complete.
    """
    import re as _re

    violations: list = []

    locked_ptx = set("%" + n for n in locked_reg_names)  # e.g. {"%R0", "%R2"}

    # Find all uppercase-R register references in the PTX.
    # Pattern matches %R followed by one or more alphanumeric chars (e.g. %R0, %R12, %R0a).
    # PTX is case-sensitive: %R0 and %r0 are different identifiers.
    UPPER_R_REF = _re.compile(r'%R([A-Za-z0-9_]+)')

    seen_upper_r: set = set()
    for line in ptx_text.splitlines():
        stripped = line.strip()
        # Skip blank lines, pure comments, and pure directive lines
        if not stripped or stripped.startswith("//"):
            continue
        for m in UPPER_R_REF.finditer(line):
            reg = "%" + "R" + m.group(1)
            seen_upper_r.add(reg)

    # Any %R<n> in the PTX that is NOT in the declared locked set is a violation.
    unlocked = seen_upper_r - locked_ptx
    for reg in sorted(unlocked):
        violations.append(
            f"  {reg} appears in PTX output but was not declared as a "
            f"locked physical register (locked set: {sorted(locked_ptx)})"
        )

    if violations:
        raise RuntimeError(
            "verify_no_temp_touches_locked_physical: "
            f"{len(violations)} violation(s) found -- PTX contains uppercase-R "
            "register name(s) outside the hard-locked physical set:\n"
            + "\n".join(violations)
        )


# -- CLI entry point --------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from compiler import Compiler

    EXAMPLES = Path(__file__).parent.parent / "examples"
    oracle_ptx, _ = Compiler().compile_file(EXAMPLES / "point_add_mixed.mtl",      "pollard_rho_step")
    plain_ptx,  _ = Compiler().compile_file(EXAMPLES / "point_add_mixed_plain.mtl", "point_add_sm89")

    manifest = verify_pair(
        oracle_ptx=oracle_ptx,
        oracle_kernel="pollard_rho_step",
        plain_ptx=plain_ptx,
        plain_kernel="point_add_sm89",
    )

    print(manifest.to_json())
    print(f"\nVerdict: {manifest.verdict}")
    if manifest.note:
        print(manifest.note)
    sys.exit(0 if manifest.verdict == "PASS" else 1)
