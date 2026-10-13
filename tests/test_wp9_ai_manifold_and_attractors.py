"""
WP9: AI Model Layer -- Manifolds, Attractors, and Quantized Neural Types.
========================================================================

Tests:
  1. manifold parses to ManifoldDecl with correct tensors, types, and alignments
  2. attractor parses to AttractorDecl with pinned target tokens, threshold, exit flag
  3. Quantized pointer types (q8_0*, q6_k*, iq2_xxs*, f16*, bf16*) parse cleanly
  4. `use AttractorName;` injects NAME_target and NAME_threshold constants into motor body
  5. `use ManifoldName;` records the manifold in motor.used_manifolds
  6. Full teleological motor with attractor pinning compiles to valid sm_89 PTX
  7. Quantized tensor parameters lower to 64-bit addresses (.u64) in PTX
"""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lexer import Lexer
from parser import Parser, ParseError
from ast_nodes import (Program, ManifoldDecl, TensorDecl, AttractorDecl,
                       VarDecl, Assign)
from compiler import Compiler, CompileError

# ── helpers ───────────────────────────────────────────────────────────────────

def parse_src(src: str) -> Program:
    return Parser(Lexer(src).tokenize()).parse()

def compile_src(src: str):
    return Compiler().compile_source(src)

SIMPLE_TX = "transmission t { grid(1,1,1); block(256,1,1); motor m {} }"

# ── 1. manifold parses correctly ──────────────────────────────────────────────

def test_manifold_parses_to_manifold_decl():
    src = """
manifold DeepSeekLayerManifold {
    tensor token_embd   : q8_0   @align(16);
    tensor attn_norm    : f32    @align(16);
    tensor ffn_down     : iq2_xxs @align(16);
    tensor output_unemb : q6_k   @align(16);
}
motor m {}
""" + SIMPLE_TX
    prog = parse_src(src)
    assert len(prog.manifolds) == 1
    m = prog.manifolds[0]
    assert isinstance(m, ManifoldDecl)
    assert m.name == "DeepSeekLayerManifold"
    assert len(m.tensors) == 4
    
    t0 = m.tensors[0]
    assert t0.name == "token_embd"
    assert t0.dtype == "q8_0"
    assert t0.alignment == 16

    t2 = m.tensors[2]
    assert t2.name == "ffn_down"
    assert t2.dtype == "iq2_xxs"
    assert t2.alignment == 16

# ── 2. attractor parses correctly ─────────────────────────────────────────────

def test_attractor_parses_correctly():
    src = """
attractor BinaryChoiceAttractor {
    pin: [129280, 15, 100];
    threshold: 0.85;
    exit_on_hit: true;
}
motor m {}
""" + SIMPLE_TX
    prog = parse_src(src)
    assert len(prog.attractors) == 1
    a = prog.attractors[0]
    assert isinstance(a, AttractorDecl)
    assert a.name == "BinaryChoiceAttractor"
    assert a.target_tokens == [129280, 15, 100]
    assert abs(a.threshold - 0.85) < 1e-5
    assert a.exit_on_hit is True

# ── 3. quantized types in motor contracts ─────────────────────────────────────

def test_quantized_types_in_motor_contract():
    src = """
motor score_candidates {
    @register(R0) float* x_final;
    @register(R1) q6_k*  u_cands;
    @register(R2) float* out_logits;
    int gid;
    gid = blockIdx.x * blockDim.x + threadIdx.x;
    out_logits[gid] = x_final[gid];
}

transmission score_tx {
    grid (1, 1, 1);
    block (256, 1, 1);
    motor score_candidates {
        in    float* x_final    @align(16);
        in    q6_k*  u_cands    @align(16);
        inout float* out_logits @align(16);
    }
}
"""
    ptx, prog = compile_src(src)
    assert ".target sm_89" in ptx
    assert "score_candidates" in ptx
    # Check that pointers to quantized types are emitted as 64-bit addresses
    assert ".reg .u64 %R0;" in ptx
    assert ".reg .u64 %R1;" in ptx
    assert ".reg .u64 %R2;" in ptx

# ── 4. use Attractor injects constants ────────────────────────────────────────

def test_use_attractor_injects_constants():
    src = """
attractor TargetTokenAttractor {
    pin: [42, 100];
    threshold: 0.75;
}

motor check_attractor {
    use TargetTokenAttractor;
    @register(R0) float* logits;
    u64 target_val;
    target_val = TargetTokenAttractor_target;
    logits[0] = TargetTokenAttractor_threshold;
    logits[1] = target_val;
}

transmission check_tx {
    grid (1, 1, 1);
    block (256, 1, 1);
    motor check_attractor {
        inout float* logits @align(16);
    }
}
"""
    ptx, prog = compile_src(src)
    motor = prog.motors[0]
    assert "TargetTokenAttractor" in motor.used_attractors
    assert ".target sm_89" in ptx
    # Verify the constant injection happened (42 injected as target)
    assert "42" in ptx

# ── 5. use Manifold records in motor ──────────────────────────────────────────

def test_use_manifold_records_in_motor():
    src = """
manifold LayerWeights {
    tensor ffn_gate : iq2_xxs @align(16);
    tensor ffn_down : iq2_xxs @align(16);
}

motor eval_ffn {
    use LayerWeights;
    @register(R0) float* x;
    x[0] = x[0] + 1.0;
}

transmission ffn_tx {
    grid (1, 1, 1);
    block (256, 1, 1);
    motor eval_ffn {
        inout float* x @align(16);
    }
}
"""
    ptx, prog = compile_src(src)
    motor = prog.motors[0]
    assert "LayerWeights" in motor.used_manifolds
    assert ".target sm_89" in ptx

# ── 6. Full end-to-end teleological neural motor ──────────────────────────────

def test_full_teleological_neural_motor():
    src = """
attractor BinaryYesNo {
    pin: [15, 16]; // 'y', 'n'
    threshold: 0.90;
}

manifold DeepSeekWeights {
    tensor unemb_weight : q6_k @align(16);
}

motor teleological_step {
    use BinaryYesNo;
    use DeepSeekWeights;

    @register(R0) float* hidden_state;
    @register(R1) q6_k*  unemb_matrix;
    @register(R2) float* out_logits;

    u64 target_tok;
    float thresh;
    target_tok = BinaryYesNo_target;
    thresh = BinaryYesNo_threshold;

    int gid;
    gid = blockIdx.x * blockDim.x + threadIdx.x;
    out_logits[gid] = hidden_state[gid] * thresh;
}

transmission teleological_launch {
    grid (1, 1, 1);
    block (256, 1, 1);
    motor teleological_step {
        in    float* hidden_state @align(16);
        in    q6_k*  unemb_matrix @align(16);
        inout float* out_logits   @align(16);
    }
}
"""
    ptx, prog = compile_src(src)
    assert ".target sm_89" in ptx
    assert "teleological_step" in ptx
    assert "teleological_launch" in ptx
    assert ".reg .u64 %R0;" in ptx
    assert ".reg .u64 %R1;" in ptx
    assert ".reg .u64 %R2;" in ptx
