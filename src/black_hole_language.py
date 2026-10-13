"""Ingot black-hole language harness.

This module keeps the contract honest:

1. The Ingot black-hole program must compile and pass no-spill PTX gates.
2. The real CUDA black-hole worker must recover known public-key fixtures.

The current real worker is the X-only public-key verifier from the work directory
workspace. HASH160 lowering is represented by the Ingot pipeline skeleton and
can be swapped in when the generated backend is ready.
"""

from __future__ import annotations

import json
import subprocess
import sys
import csv
import io
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import Iterable

from compiler import Compiler
from native_verify import compile_ptx_to_cubin, parse_ptxas_function_stats


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MOTOR_WORK = Path(os.environ.get("INGOT_MOTOR_WORK", "work"))
DEFAULT_MOTOR_PROBE = Path(os.environ.get("INGOT_MOTOR_PROBE", "probe"))
DEFAULT_TARGETS = DEFAULT_MOTOR_WORK / "cookie_cutter_targets.json"
DEFAULT_X_WORKER = DEFAULT_MOTOR_WORK / "black_hole_server.exe"
DEFAULT_HASH160_WORKER = DEFAULT_MOTOR_WORK / "black_hole_hash160_server.exe"
DEFAULT_BRIDGEV3_WORKER = DEFAULT_MOTOR_PROBE / "cuda_bottom_end" / "bottom_end_cuda_mock_bridgev3.exe"
DEFAULT_BRIDGEV3_P71_ROWS = (
    DEFAULT_MOTOR_PROBE
    / "bottom_end_reports"
    / "bottom_end_bridge_packet_v3_p71_t3_z64_e24_rows.csv"
)
COORDINATE_HOT_PATH_SPEC = ROOT / "examples" / "black_hole_coordinate_hot_path.mtl"
BRIDGEV3_LABEL_HOT_PATH_SPEC = ROOT / "examples" / "black_hole_bridgev3_label_hot_path.mtl"
DRIFT_IDENTITY_HOT_PATH_SPEC = ROOT / "examples" / "black_hole_drift_identity_hot_path.mtl"
PIPELINE_SPEC = ROOT / "examples" / "black_hole_hash160_pipeline.mtl"
PACKET_SPEC = ROOT / "examples" / "black_hole_packet.mtl"
COORDINATE_HOT_PATH_REGS = tuple(f"%R{i}" for i in range(26))
BRIDGEV3_LABEL_HOT_PATH_REGS = tuple(f"%R{i}" for i in range(15))
DRIFT_IDENTITY_HOT_PATH_REGS = tuple(f"%R{i}" for i in range(19))


@dataclass(frozen=True)
class IngotBlackHoleGate:
    spec: Path
    entries: dict[str, tuple[int, int, int, int]]

    @property
    def entry_count(self) -> int:
        return len(self.entries)


@dataclass(frozen=True)
class PublicKeyFixtureResult:
    puzzle: int
    scalar: int
    low: int
    count: int
    hit_offset: int
    seconds: float
    coordinates_per_sec: float

    @property
    def recovered_scalar(self) -> int:
        return self.low + self.hit_offset


@dataclass(frozen=True)
class Hash160FixtureResult(PublicKeyFixtureResult):
    batch_inv: int
    verifier: str = "hash160"


@dataclass(frozen=True)
class BridgeV3LabelResult:
    rows: int
    rounds: int
    candidates_per_packet: int
    seconds: float
    packet_labels_per_sec: float
    true_virtual_integers_per_sec: float
    routed_packets_per_sec: float
    dyadic_mask_expansions_per_sec: float
    z_side_rail_additions_per_sec: float
    packet_q_contracts_per_sec: float
    candidate_compare_contracts_per_sec: float
    hash_verifier: str
    full_qx_contract: str
    curve_identity: str
    acc: int
    hot_path_gate: IngotBlackHoleGate


@dataclass(frozen=True)
class KnownKLandingFixture:
    puzzle: int
    range_low: int
    range_high: int
    known_k: int
    source: str


@dataclass(frozen=True)
class KnownKLandingResult:
    puzzle: int
    range_low: int
    range_high: int
    known_k: int
    offset: int
    packet_bits: int
    packet_span: int
    packet_cell: int
    local_offset: int
    landed_k: int
    source: str
    hot_path_gate: IngotBlackHoleGate | None

    @property
    def match(self) -> bool:
        return self.landed_k == self.known_k


KNOWN_K_LANDING_FIXTURES = {
    5: KnownKLandingFixture(
        puzzle=5,
        range_low=0x10,
        range_high=0x1F,
        known_k=0x15,
        source="bottom_end_additive_power_tail_p5_p30_t3_z64_e24_best.csv",
    ),
    65: KnownKLandingFixture(
        puzzle=65,
        range_low=0x10000000000000000,
        range_high=0x1FFFFFFFFFFFFFFFF,
        known_k=0x1A838B13505B26867,
        source="solved_p30plus_shape_k_oracle.csv",
    ),
    70: KnownKLandingFixture(
        puzzle=70,
        range_low=0x200000000000000000,
        range_high=0x3FFFFFFFFFFFFFFFFF,
        known_k=0x349B84B6431A6C4EF1,
        source="solved_p30plus_shape_k_oracle.csv",
    ),
}


@lru_cache(maxsize=None)
def _cached_gate_ingot_spec(spec_text: str, max_registers: int) -> IngotBlackHoleGate:
    return gate_ingot_spec(Path(spec_text), max_registers=max_registers)


@lru_cache(maxsize=None)
def _cached_gate_coordinate_hot_path(spec_text: str) -> IngotBlackHoleGate:
    return gate_coordinate_hot_path(Path(spec_text))


@lru_cache(maxsize=None)
def _cached_gate_bridgev3_label_hot_path(spec_text: str) -> IngotBlackHoleGate:
    return gate_bridgev3_label_hot_path(Path(spec_text))


@lru_cache(maxsize=None)
def _cached_gate_drift_identity_hot_path(spec_text: str) -> IngotBlackHoleGate:
    return gate_drift_identity_hot_path(Path(spec_text))


class BlackHoleCoordinateWorker:
    """Fast coordinate CUDA worker, launched only after the Ingot hot-path gate."""

    def __init__(
        self,
        executable: Path = DEFAULT_X_WORKER,
        hot_path_spec: Path = COORDINATE_HOT_PATH_SPEC,
        require_ingot_hot_path: bool = True,
    ) -> None:
        self.executable = executable
        self.hot_path_gate = (
            _cached_gate_coordinate_hot_path(str(hot_path_spec))
            if require_ingot_hot_path
            else None
        )
        popen_kwargs: dict = dict(
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(executable.parent),
            bufsize=1,
        )
        if sys.platform == "win32":
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = subprocess.SW_HIDE
            popen_kwargs["startupinfo"] = startup
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        self.process = subprocess.Popen([str(executable), "--server"], **popen_kwargs)
        ready = self._read()
        if not ready.get("ready"):
            raise RuntimeError(f"black-hole worker did not become ready: {ready}")

    def _read(self) -> dict:
        if self.process.stdout is None:
            raise RuntimeError("black-hole worker stdout is unavailable")
        line = self.process.stdout.readline()
        if not line:
            error = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"black-hole worker stopped unexpectedly: {error}")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def scan(self, public_key_x: str, low: int, count: int) -> dict:
        if self.process.stdin is None:
            raise RuntimeError("black-hole worker stdin is unavailable")
        self.process.stdin.write(f"{public_key_x} 0x{low:x} {count}\n")
        self.process.stdin.flush()
        return self._read()

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            if self.process.stdin is not None:
                self.process.stdin.write("QUIT\n")
                self.process.stdin.flush()
            self.process.wait(timeout=2)
        except Exception:
            self.process.kill()
            self.process.wait(timeout=2)

    def __enter__(self) -> "BlackHoleXWorker":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


BlackHoleXWorker = BlackHoleCoordinateWorker


class BlackHoleHash160Worker(BlackHoleCoordinateWorker):
    """Persistent wrapper around the address/HASH160 CUDA black-hole server."""

    def __init__(
        self,
        executable: Path = DEFAULT_HASH160_WORKER,
        require_ingot_hot_path: bool = False,
    ) -> None:
        super().__init__(
            executable,
            hot_path_spec=PIPELINE_SPEC,
            require_ingot_hot_path=require_ingot_hot_path,
        )

    def scan_hash160(self, target_hash160: str, low: int, count: int) -> dict:
        return self.scan(target_hash160, low, count)

    def __enter__(self) -> "BlackHoleHash160Worker":
        return self


def load_targets(path: Path = DEFAULT_TARGETS) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def gate_ingot_spec(spec: Path = PIPELINE_SPEC, max_registers: int = 96) -> IngotBlackHoleGate:
    ptx, _ = Compiler().compile_all_file(spec)
    _, ptxas_stats = compile_ptx_to_cubin(ptx, sm="sm_89")
    entries = parse_ptxas_function_stats(ptxas_stats)
    if not entries:
        raise AssertionError(f"{spec.name}: no ptxas entry stats parsed")
    for name, (registers, spill_stores, spill_loads, stack_frame) in entries.items():
        if registers <= 0:
            raise AssertionError(f"{name}: no register count parsed")
        if registers > max_registers:
            raise AssertionError(f"{name}: register count {registers} exceeds {max_registers}")
        if spill_stores or spill_loads or stack_frame:
            raise AssertionError(
                f"{name}: stack/spill violation stack={stack_frame} "
                f"stores={spill_stores} loads={spill_loads}"
            )
    return IngotBlackHoleGate(spec=spec, entries=entries)


def gate_coordinate_hot_path(spec: Path = COORDINATE_HOT_PATH_SPEC) -> IngotBlackHoleGate:
    ptx, _ = Compiler().compile_all_file(spec)
    missing = [reg for reg in COORDINATE_HOT_PATH_REGS if reg not in ptx]
    if missing:
        raise AssertionError(f"{spec.name}: missing locked hot-path registers {missing}")
    required_ops = (
        "mul.lo.u64 %R16",
        "mul.lo.u64 %R20",
        "xor.b64",
        "or.b64 %R24",
    )
    missing_ops = [op for op in required_ops if op not in ptx]
    if missing_ops:
        raise AssertionError(f"{spec.name}: missing coordinate hot-path ops {missing_ops}")

    _, ptxas_stats = compile_ptx_to_cubin(ptx, sm="sm_89")
    entries = parse_ptxas_function_stats(ptxas_stats)
    if not entries:
        raise AssertionError(f"{spec.name}: no ptxas entry stats parsed")
    for name, (registers, spill_stores, spill_loads, stack_frame) in entries.items():
        if spill_stores or spill_loads or stack_frame:
            raise AssertionError(
                f"{name}: coordinate hot-path stack/spill violation stack={stack_frame} "
                f"stores={spill_stores} loads={spill_loads}"
            )
    return IngotBlackHoleGate(spec=spec, entries=entries)


def gate_bridgev3_label_hot_path(spec: Path = BRIDGEV3_LABEL_HOT_PATH_SPEC) -> IngotBlackHoleGate:
    ptx, _ = Compiler().compile_all_file(spec)
    missing = [reg for reg in BRIDGEV3_LABEL_HOT_PATH_REGS if reg not in ptx]
    if missing:
        raise AssertionError(f"{spec.name}: missing bridgev3 locked registers {missing}")
    required_ops = (
        "mad.lo.u64 %v0, %R3, %R4, %R0",
        "add.u64 %R7, %v0, %R5",
        "and.b64 %R10, %R1",
        "mov.u64 %R12, %R6",
        "mov.u64 %R13, %R6",
        "mul.lo.u64 %v1, %R1",
        "shl.b64",
    )
    missing_ops = [op for op in required_ops if op not in ptx]
    if missing_ops:
        raise AssertionError(f"{spec.name}: missing bridgev3 label ops {missing_ops}")

    _, ptxas_stats = compile_ptx_to_cubin(ptx, sm="sm_89")
    entries = parse_ptxas_function_stats(ptxas_stats)
    if not entries:
        raise AssertionError(f"{spec.name}: no ptxas entry stats parsed")
    for name, (_registers, spill_stores, spill_loads, stack_frame) in entries.items():
        if spill_stores or spill_loads or stack_frame:
            raise AssertionError(
                f"{name}: bridgev3 label stack/spill violation stack={stack_frame} "
                f"stores={spill_stores} loads={spill_loads}"
            )
    return IngotBlackHoleGate(spec=spec, entries=entries)


def gate_drift_identity_hot_path(spec: Path = DRIFT_IDENTITY_HOT_PATH_SPEC) -> IngotBlackHoleGate:
    ptx, _ = Compiler().compile_all_file(spec)
    missing = [reg for reg in DRIFT_IDENTITY_HOT_PATH_REGS if reg not in ptx]
    if missing:
        raise AssertionError(f"{spec.name}: missing drift identity registers {missing}")
    forbidden = ("GX", "GY", "jac64", "point_mul", "raw G", " G;")
    leaked = [token for token in forbidden if token in ptx]
    if leaked:
        raise AssertionError(f"{spec.name}: raw curve generator leaked into drift rail {leaked}")
    required_ops = (
        "mov.u64 %R9, %R0",
        "and.b64 %R10, %R1",
        "shl.b64",
        "or.b64 %R12",
        "mad.lo.u64",
        "xor.b64 %R14",
        "cvt.u64.u32 %R17, 3",
    )
    missing_ops = [op for op in required_ops if op not in ptx]
    if missing_ops:
        raise AssertionError(f"{spec.name}: missing drift identity ops {missing_ops}")

    _, ptxas_stats = compile_ptx_to_cubin(ptx, sm="sm_89")
    entries = parse_ptxas_function_stats(ptxas_stats)
    if not entries:
        raise AssertionError(f"{spec.name}: no ptxas entry stats parsed")
    for name, (_registers, spill_stores, spill_loads, stack_frame) in entries.items():
        if spill_stores or spill_loads or stack_frame:
            raise AssertionError(
                f"{name}: drift identity stack/spill violation stack={stack_frame} "
                f"stores={spill_stores} loads={spill_loads}"
            )
    return IngotBlackHoleGate(spec=spec, entries=entries)


def run_bridgev3_label_hot_path(
    rounds: int = 1_048_576,
    rows_path: Path = DEFAULT_BRIDGEV3_P71_ROWS,
    candidates_per_packet: int = 4,
    executable: Path = DEFAULT_BRIDGEV3_WORKER,
    require_ingot_hot_path: bool = True,
    drift_identity: bool = False,
) -> BridgeV3LabelResult:
    gate = (
        _cached_gate_drift_identity_hot_path(str(DRIFT_IDENTITY_HOT_PATH_SPEC))
        if drift_identity
        else _cached_gate_bridgev3_label_hot_path(str(BRIDGEV3_LABEL_HOT_PATH_SPEC))
        if require_ingot_hot_path
        else IngotBlackHoleGate(spec=BRIDGEV3_LABEL_HOT_PATH_SPEC, entries={})
    )
    completed = subprocess.run(
        [
            str(executable),
            str(rounds),
            "bridgev3",
            str(rows_path),
            str(candidates_per_packet),
        ],
        cwd=str(executable.parent),
        text=True,
        capture_output=True,
        check=True,
    )
    reader = csv.DictReader(io.StringIO(completed.stdout))
    row = next(reader)
    if row["mode"] != "bridge_packet_v3_qrail_cuda":
        raise AssertionError(f"unexpected bridgev3 mode: {row['mode']}")
    if row["hash_verifier"] != "not_run":
        raise AssertionError(f"bridgev3 label lane unexpectedly ran hash verifier: {row['hash_verifier']}")
    return BridgeV3LabelResult(
        rows=int(row["rows"]),
        rounds=int(row["rounds"]),
        candidates_per_packet=int(row["candidates_per_packet"]),
        seconds=float(row["seconds"]),
        packet_labels_per_sec=float(row["packet_labels_per_sec"]),
        true_virtual_integers_per_sec=float(row["true_virtual_integers_per_sec"]),
        routed_packets_per_sec=float(row["routed_packets_per_sec"]),
        dyadic_mask_expansions_per_sec=float(row["dyadic_mask_expansions_per_sec"]),
        z_side_rail_additions_per_sec=float(row["z_side_rail_additions_per_sec"]),
        packet_q_contracts_per_sec=float(row["packet_q_contracts_per_sec"]),
        candidate_compare_contracts_per_sec=float(row["candidate_compare_contracts_per_sec"]),
        hash_verifier=row["hash_verifier"],
        full_qx_contract=row["full_qx_contract"],
        curve_identity=row["curve_identity"],
        acc=int(row["acc"]),
        hot_path_gate=gate,
    )


def known_public_key_targets(puzzles: Iterable[int], targets_path: Path = DEFAULT_TARGETS) -> list[dict]:
    wanted = set(puzzles)
    rows = []
    for row in load_targets(targets_path):
        if row.get("puzzle") in wanted and row.get("known_scalar_hex") and row.get("public_key_x"):
            rows.append(row)
    missing = wanted - {row["puzzle"] for row in rows}
    if missing:
        raise ValueError(f"missing known public-key fixtures: {sorted(missing)}")
    return rows


def known_k_landing_fixtures(puzzles: Iterable[int] = (70, 65, 5)) -> list[KnownKLandingFixture]:
    wanted = list(puzzles)
    missing = [puzzle for puzzle in wanted if puzzle not in KNOWN_K_LANDING_FIXTURES]
    if missing:
        raise ValueError(f"missing known-k landing fixtures: {missing}")
    return [KNOWN_K_LANDING_FIXTURES[puzzle] for puzzle in wanted]


def compute_known_k_landing(
    fixture: KnownKLandingFixture,
    packet_bits: int = 30,
    hot_path_gate: IngotBlackHoleGate | None = None,
) -> KnownKLandingResult:
    if packet_bits < 0:
        raise ValueError("packet_bits must be non-negative")
    if not (fixture.range_low <= fixture.known_k <= fixture.range_high):
        raise AssertionError(
            f"P{fixture.puzzle}: known k {fixture.known_k:#x} is outside "
            f"[{fixture.range_low:#x}, {fixture.range_high:#x}]"
        )
    packet_span = 1 << packet_bits
    offset = fixture.known_k - fixture.range_low
    packet_cell, local_offset = divmod(offset, packet_span)
    landed_k = fixture.range_low + packet_cell * packet_span + local_offset
    return KnownKLandingResult(
        puzzle=fixture.puzzle,
        range_low=fixture.range_low,
        range_high=fixture.range_high,
        known_k=fixture.known_k,
        offset=offset,
        packet_bits=packet_bits,
        packet_span=packet_span,
        packet_cell=packet_cell,
        local_offset=local_offset,
        landed_k=landed_k,
        source=fixture.source,
        hot_path_gate=hot_path_gate,
    )


def run_known_k_landings(
    puzzles: Iterable[int] = (70, 65, 5),
    packet_bits: int = 30,
    require_ingot_hot_path: bool = True,
) -> list[KnownKLandingResult]:
    gate = (
        _cached_gate_drift_identity_hot_path(str(DRIFT_IDENTITY_HOT_PATH_SPEC))
        if require_ingot_hot_path
        else None
    )
    results = [
        compute_known_k_landing(fixture, packet_bits=packet_bits, hot_path_gate=gate)
        for fixture in known_k_landing_fixtures(puzzles)
    ]
    missed = [result.puzzle for result in results if not result.match]
    if missed:
        raise AssertionError(f"known-k landing mismatch for puzzles: {missed}")
    return results


def run_public_key_fixture(
    row: dict,
    worker: BlackHoleXWorker,
    radius: int = 2048,
) -> PublicKeyFixtureResult:
    scalar = int(row["known_scalar_hex"], 16)
    low = max(int(row["range_low_hex"], 16), scalar - radius)
    high = min(int(row["range_high_hex"], 16), scalar + radius - 1)
    count = high - low + 1
    result = worker.scan(row["public_key_x"], low, count)
    if not result["match"]:
        raise AssertionError(f"P{row['puzzle']}: no hit in bracket [{low:#x}, {high:#x}]")
    hit_offset = int(result["hit_offset"])
    recovered = low + hit_offset
    if recovered != scalar:
        raise AssertionError(f"P{row['puzzle']}: recovered {recovered:#x}, expected {scalar:#x}")
    return PublicKeyFixtureResult(
        puzzle=int(row["puzzle"]),
        scalar=scalar,
        low=low,
        count=count,
        hit_offset=hit_offset,
        seconds=float(result["seconds"]),
        coordinates_per_sec=float(result["coordinates_per_sec"]),
    )


def run_hash160_fixture(
    row: dict,
    worker: BlackHoleHash160Worker,
    radius: int = 2048,
) -> Hash160FixtureResult:
    scalar = int(row["known_scalar_hex"], 16)
    low = max(int(row["range_low_hex"], 16), scalar - radius)
    high = min(int(row["range_high_hex"], 16), scalar + radius - 1)
    count = high - low + 1
    result = worker.scan_hash160(row["hash160"], low, count)
    if result.get("verifier") != "hash160":
        raise AssertionError(f"P{row['puzzle']}: wrong verifier rail {result.get('verifier')!r}")
    if not result["match"]:
        raise AssertionError(f"P{row['puzzle']}: no HASH160 hit in bracket [{low:#x}, {high:#x}]")
    hit_offset = int(result["hit_offset"])
    recovered = low + hit_offset
    if recovered != scalar:
        raise AssertionError(f"P{row['puzzle']}: HASH160 recovered {recovered:#x}, expected {scalar:#x}")
    return Hash160FixtureResult(
        puzzle=int(row["puzzle"]),
        scalar=scalar,
        low=low,
        count=count,
        hit_offset=hit_offset,
        seconds=float(result["seconds"]),
        coordinates_per_sec=float(result["coordinates_per_sec"]),
        batch_inv=int(result.get("batch_inv", 0)),
    )


def run_public_key_fixtures(
    puzzles: Iterable[int] = (10, 15, 20, 40),
    worker_path: Path = DEFAULT_X_WORKER,
    targets_path: Path = DEFAULT_TARGETS,
    radius: int = 2048,
) -> list[PublicKeyFixtureResult]:
    rows = known_public_key_targets(puzzles, targets_path)
    with BlackHoleXWorker(worker_path) as worker:
        return [run_public_key_fixture(row, worker, radius=radius) for row in rows]


def run_hash160_fixtures(
    puzzles: Iterable[int] = (10, 15, 20, 40),
    worker_path: Path = DEFAULT_HASH160_WORKER,
    targets_path: Path = DEFAULT_TARGETS,
    radius: int = 2048,
) -> list[Hash160FixtureResult]:
    rows = known_public_key_targets(puzzles, targets_path)
    with BlackHoleHash160Worker(worker_path) as worker:
        return [run_hash160_fixture(row, worker, radius=radius) for row in rows]
