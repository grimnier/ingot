import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from black_hole_language import (
    BRIDGEV3_LABEL_HOT_PATH_SPEC,
    BlackHoleHash160Worker,
    BlackHoleXWorker,
    COORDINATE_HOT_PATH_SPEC,
    DEFAULT_BRIDGEV3_P71_ROWS,
    DEFAULT_BRIDGEV3_WORKER,
    DEFAULT_HASH160_WORKER,
    DEFAULT_X_WORKER,
    DRIFT_IDENTITY_HOT_PATH_SPEC,
    known_public_key_targets,
    PACKET_SPEC,
    PIPELINE_SPEC,
    gate_bridgev3_label_hot_path,
    gate_coordinate_hot_path,
    gate_drift_identity_hot_path,
    gate_ingot_spec,
    run_bridgev3_label_hot_path,
    run_hash160_fixtures,
    run_known_k_landings,
    run_public_key_fixtures,
)


pytestmark = pytest.mark.skipif(
    shutil.which("ptxas") is None
    or not DEFAULT_X_WORKER.exists()
    or not DEFAULT_HASH160_WORKER.exists(),
    reason="ptxas or real black-hole CUDA workers not available",
)


def test_ingot_black_hole_specs_are_no_spill():
    coordinate = gate_coordinate_hot_path(COORDINATE_HOT_PATH_SPEC)
    bridgev3 = gate_bridgev3_label_hot_path(BRIDGEV3_LABEL_HOT_PATH_SPEC)
    drift = gate_drift_identity_hot_path(DRIFT_IDENTITY_HOT_PATH_SPEC)
    pipeline = gate_ingot_spec(PIPELINE_SPEC)
    packet = gate_ingot_spec(PACKET_SPEC)

    assert coordinate.entry_count == 1
    assert bridgev3.entry_count == 1
    assert drift.entry_count == 1
    assert pipeline.entry_count == 5
    assert packet.entry_count == 5


def test_real_public_key_fixtures_p10_p15_p20_p40():
    results = run_public_key_fixtures((10, 15, 20, 40), radius=2048)

    assert [r.puzzle for r in results] == [10, 15, 20, 40]
    for result in results:
        assert result.recovered_scalar == result.scalar
        assert 0 <= result.hit_offset < result.count
        assert result.seconds > 0


def test_real_hash160_fixtures_p10_p15_p20_p40():
    results = run_hash160_fixtures((10, 15, 20, 40), radius=2048)

    assert [r.puzzle for r in results] == [10, 15, 20, 40]
    for result in results:
        assert result.verifier == "hash160"
        assert result.recovered_scalar == result.scalar
        assert 0 <= result.hit_offset < result.count
        assert result.batch_inv > 0
        assert result.seconds > 0


def test_real_public_key_p40_target_container():
    # This is the full billion-container shape: P40's target lives inside this
    # container at offset 776,549,334.
    target = known_public_key_targets((40,))[0]
    low = 0xE980000000
    count = 1 << 30

    with BlackHoleXWorker(DEFAULT_X_WORKER) as worker:
        assert worker.hot_path_gate is not None
        assert worker.hot_path_gate.spec == COORDINATE_HOT_PATH_SPEC
        result = worker.scan(target["public_key_x"], low, count)

    assert result["match"] is True
    assert low + int(result["hit_offset"]) == 0xE9AE4933D6
    assert int(result["hit_offset"]) == 776_549_334
    assert float(result["seconds"]) > 0


def test_real_hash160_p40_target_container():
    target = known_public_key_targets((40,))[0]
    low = 0xE980000000
    count = 1 << 30

    with BlackHoleHash160Worker(DEFAULT_HASH160_WORKER) as worker:
        result = worker.scan_hash160(target["hash160"], low, count)

    assert result["verifier"] == "hash160"
    assert result["match"] is True
    assert low + int(result["hit_offset"]) == 0xE9AE4933D6
    assert int(result["hit_offset"]) == 776_549_334
    assert float(result["seconds"]) > 0


@pytest.mark.skipif(
    not DEFAULT_BRIDGEV3_WORKER.exists() or not DEFAULT_BRIDGEV3_P71_ROWS.exists(),
    reason="bridgev3 CUDA label worker or P71 rows not available",
)
def test_real_bridgev3_p71_label_hot_path():
    result = run_bridgev3_label_hot_path(drift_identity=True)

    assert result.hot_path_gate.spec == DRIFT_IDENTITY_HOT_PATH_SPEC
    assert result.rows == 223
    assert result.rounds == 1_048_576
    assert result.candidates_per_packet == 4
    assert result.hash_verifier == "not_run"
    assert result.full_qx_contract == "full_32_byte_qx_equality"
    assert result.packet_labels_per_sec > 1.0e9
    assert result.candidate_compare_contracts_per_sec > result.packet_labels_per_sec


def test_known_k_landings_p70_p65_p5_billion_geometry():
    results = run_known_k_landings((70, 65, 5), packet_bits=30)

    assert [result.puzzle for result in results] == [70, 65, 5]
    assert all(result.match for result in results)
    assert all(result.packet_span == 1 << 30 for result in results)
    assert all(result.hot_path_gate.spec == DRIFT_IDENTITY_HOT_PATH_SPEC for result in results)

    by_puzzle = {result.puzzle: result for result in results}
    assert by_puzzle[70].known_k == 0x349B84B6431A6C4EF1
    assert by_puzzle[70].packet_cell == 354_034_047_244
    assert by_puzzle[70].local_offset == 443_305_713
    assert by_puzzle[65].known_k == 0x1A838B13505B26867
    assert by_puzzle[65].packet_cell == 11_289_150_676
    assert by_puzzle[65].local_offset == 95_578_215
    assert by_puzzle[5].known_k == 0x15
    assert by_puzzle[5].packet_cell == 0
    assert by_puzzle[5].local_offset == 5


def test_known_k_landings_match_legacy_byte_packet_oracle_rows():
    results = run_known_k_landings((70, 65, 5), packet_bits=8)

    assert all(result.match for result in results)
    assert all(result.packet_span == 256 for result in results)

    by_puzzle = {result.puzzle: result for result in results}
    assert by_puzzle[70].packet_cell == 1_484_926_420_493_429_838
    assert by_puzzle[70].local_offset == 241
    assert by_puzzle[65].packet_cell == 47_350_129_837_322_856
    assert by_puzzle[65].local_offset == 103
    assert by_puzzle[5].packet_cell == 0
    assert by_puzzle[5].local_offset == 5
