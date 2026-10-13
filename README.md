# Ingot GPU Programming Language (v1.0)

> **Ingot**: A Register-Locked, Zero-Spill Domain-Specific GPU Programming Language Compiler for High-Performance Cryptography & Parallel Compute.

---

## Highlights & Innovations

- **Motor vs. Transmission Architecture**: Physical separation of pure GPU compute (`motor`) from launch geometry & memory contracts (`transmission`).
- **Physical Register Locking (`@register(Rn)`)**: Direct, explicit allocation of physical GPU registers ($R_0 \dots R_{31}$) for zero local VRAM spilling.
- **Carry-Chain PTX Intrinsics**: Native 64-bit carry-chain arithmetic (`add_cc64`, `mad_lo_cc64`, `madc_hi_cc64`).
- **Zero-Spill Modular Field Math**: Pre-built sealed profiles for secp256k1 256-bit modular field multiplication and point addition.
- **Terminal User Interface**: Built-in CLI driver (`ingotc`) with ASCII register allocation heatmaps (`--reg-map`) and source line error carets.

---

## Quick Start & Installation

### Option 1: Install via `pip`
```bash
git clone https://github.com/ingot-lang/ingot.git
cd ingot
pip install -e .
```

### Option 2: Run directly with Python
```bash
python src/ingotc.py examples/point_add_mixed.mtl -o point_add_mixed.ptx --target sm_86 --verify
```

---

## Command Line Usage (`ingotc`)

```bash
# Compile source file and verify register locks
ingotc examples/fe_mul.mtl --verify

# Render ASCII physical register allocation heatmap
ingotc examples/fe_mul.mtl --reg-map

# Emit PTX assembly targeting NVIDIA Ampere (sm_86) or Ada Lovelace (sm_89)
ingotc examples/point_add_mixed.mtl -o point_add_mixed.ptx --target sm_86
```

### Register Allocation Heatmap Output (`--reg-map`)

```
========================================================================
  INGOT COMPILER v1.0 — GPU Domain-Specific Language Toolchain            
========================================================================
  Source File           : examples/fe_mul.mtl
  Target Architecture   : sm_80
  [INGOT COMPILED OK]   : Parsed 1 Motor(s), 1 Transmission(s)
  ├─ Motor 'fe_mul': 16 locked registers (0 spills [REGISTER LOCKED])

--- REGISTER ALLOCATION HEATMAP: 'fe_mul' ---
  [ R0 ] [ R1 ] [ R2 ] [ R3 ] [ R4 ] [ R5 ] [ R6 ] [ R7 ] 
  [ R8 ] [ R9 ] [R10 ] [R11 ] [R12 ] [R13 ] [R14 ] [R15 ] 
  [R16 ] [R17 ] [R18 ] [R19 ] [R20 ] [R21 ] [R22 ] [R23 ] 
  [R24 ] [R25 ] [R26 ] [R27 ] [R28 ] [R29 ] [R30 ] [R31 ] 
  Legend: [R_n] = Claimed Physical Register | [  ] = Free Register Slot
```

---

## Language Syntax Example (`.mtl`)

```ingot
// 256-Bit Modular Addition Motor
motor field256_add {
    @register(R0) u64 a0; @register(R1) u64 a1; @register(R2) u64 a2; @register(R3) u64 a3;
    @register(R4) u64 b0; @register(R5) u64 b1; @register(R6) u64 b2; @register(R7) u64 b3;

    u64 r0; u64 r1; u64 r2; u64 r3;
    r0 = add_cc64(a0, b0);
    r1 = addc64(a1, b1);
    r2 = addc64(a2, b2);
    r3 = addc64(a3, b3);

    a0 = r0; a1 = r1; a2 = r2; a3 = r3;
}

transmission field256_add_tx {
    grid(1024, 1, 1);
    block(128, 1, 1);
    motor field256_add;
}
```

---

## Verification & Testing

Run the complete compiler validation test suite and real GPU kernel execution:

```bash
python validate_all.py
```

---

## Directory Structure

```
ingot/
├── docs/                      # Formal Language Spec & Tutorials
│   ├── LANGUAGE_SPEC.md       # Ingot Language Reference Manual
│   └── TUTORIAL.md            # Developer Quick-Start Guide
├── stdlib/                    # Ingot Standard Library Modules
│   ├── field256.mtl           # 256-Bit Modular Arithmetic
│   ├── secp256k1.mtl          # Elliptic Curve Point Addition
│   └── hash160.mtl            # SHA-256 + RIPEMD-160 Digest
├── src/                       # Compiler Pipeline Source
│   ├── ingotc.py              # Main Compiler Driver CLI
│   ├── lexer.py               # Tokenizer
│   ├── parser.py              # AST Parser
│   ├── semantic.py            # Contract Checker
│   ├── ir.py                  # Intermediate Representation
│   └── ptx_emitter.py         # PTX Code Generator
├── examples/                  # 44 Verified .mtl Kernel Examples
├── validate_all.py            # Full Automated Test Suite
└── LICENSE                    # Open-Source MIT License
```

---

## License

Distributed under the [MIT License](LICENSE).
