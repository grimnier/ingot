import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from compiler import Compiler
from native_verify import compile_ptx_to_cubin, parse_ptxas_function_stats


ROOT = Path(__file__).parent.parent
PIPELINE = ROOT / "examples" / "black_hole_hash160_pipeline.mtl"


pytestmark = pytest.mark.skipif(
    shutil.which("ptxas") is None,
    reason="ptxas not found; black-hole pipeline spill gate skipped",
)


def test_black_hole_hash160_pipeline_is_five_no_spill_entries():
    ptx, prog = Compiler().compile_all_file(PIPELINE)
    _, ptxas_stats = compile_ptx_to_cubin(ptx, sm="sm_89")
    per_entry = parse_ptxas_function_stats(ptxas_stats)

    expected = {
        "seed_walker_seed_walker_tx",
        "sha256_bh_sha256_bh_tx",
        "bswap_bridge_bswap_bridge_tx",
        "ripemd160_bh_ripemd160_bh_tx",
        "match_collector_match_collector_tx",
    }
    assert {t.name for t in prog.transmissions} == {
        "seed_walker_tx",
        "sha256_bh_tx",
        "bswap_bridge_tx",
        "ripemd160_bh_tx",
        "match_collector_tx",
    }
    assert set(per_entry) == expected

    for name, (registers, spill_stores, spill_loads, stack_frame) in per_entry.items():
        assert registers > 0, f"{name}: no register count parsed"
        assert registers <= 96, f"{name}: register count {registers} exceeds profile"
        assert spill_stores == 0, f"{name}: spill stores {spill_stores}"
        assert spill_loads == 0, f"{name}: spill loads {spill_loads}"
        assert stack_frame == 0, f"{name}: stack frame {stack_frame}"
