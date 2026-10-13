# Ingot GPU Language Specification (v1.0)

## Overview
Ingot is a domain-specific, register-locked GPU programming language engineered for high-performance cryptographic arithmetic, warp synchronization, and low-latency parallel computing.

---

## 1. Syntax & Core Declarations

### Motor Blocks (`motor`)
A `motor` defines a GPU compute block executing on CUDA cores. Physical registers are locked to specific variables using `@register(Rn)`:

```ingot
motor field_mul {
    @register(R0) u64 a0; @register(R1) u64 a1;
    @register(R2) u64 b0; @register(R3) u64 b1;
    @register(R4) u64 t0; @register(R5) u64 t1;

    t0 = mul_lo64(a0, b0);
    t1 = mad_lo_cc64(a0, b1, 0);
    a0 = t0; a1 = t1;
}
```

### Transmission Blocks (`transmission`)
A `transmission` defines the execution contract, launch geometry, and grid/block parameters:

```ingot
transmission field_mul_tx {
    grid(1024, 1, 1);
    block(128, 1, 1);
    motor field_mul;
}
```

---

## 2. Types & Primitive Intrinsics

### Primitive Data Types
* **`u64`**: 64-bit unsigned integer (registers `%R0...%Rn`).
* **`u32`**: 32-bit unsigned integer (registers `%r0...%rn`).
* **`f32` / `f64`**: Single and double precision floating point.

### Builtin Carry-Chain Assembly Intrinsics
* **`add_cc64(a, b)`**: 64-bit addition setting PTX carry flag (`add.cc.u64`).
* **`addc64(a, b)`**: 64-bit addition with carry in (`addc.u64`).
* **`mad_lo_cc64(a, b, c)`**: 64-bit multiply-accumulate low bits setting carry (`mad.lo.cc.u64`).
* **`madc_hi64(a, b, c)`**: 64-bit multiply-accumulate high bits with carry (`madc.hi.u64`).

---

## 3. Parallel GPU Extensions

* **`__shared__` Memory**: Declares thread-block shared memory arrays (`__shared__ u64 buf[256];`, or `shared float tile[32][8];`). The bare keyword `shared` is the same declaration. The array is row-major; every dimension is a constant. The emitter allocates `.shared .b8` by element width. The address is `mov.u32` of the symbol, a shared-space offset, then `ld.shared` / `st.shared`.
* **`__syncthreads()`**: Synchronizes execution across all threads in a thread block (`bar.sync 0;`). `syncthreads()` is the same statement.
* **Multi-dimensional indexes**: A pointer declared `float* manifold[][12][4]` is indexed `manifold[expert][row][col]`. Only the leading dimension may be unsized. The compiler emits the row-major stride product (`expert * 48 + row * 4 + col` for that shape) and the existing element-byte scale. A one-dimensional `p[i]` with no declared shape is unchanged.
* **Warp Shuffles**: `__shfl_up_sync(mask, val, delta)` and `__shfl_down_sync(mask, val, delta)` for intra-warp data transfer.
* **Atomics**: `atomicAdd(ptr, val)` and `atomicAdd(&ptr, val)` add into the addressed element. The element type selects the opcode: `atom.global.add.f32` for `float*`, `atom.global.add.u32` for `u32*`, `atom.global.add.u64` for `u64*`. A `__shared__` or `shared` array uses `atom.shared` on the same `.u32` shared-space offset as `ld.shared` / `st.shared`. The value is converted to the element type before the atomic. `atomicCAS` is not in the language. A plain `int*` keeps the historical `.f32` element type.

---

## 4. Compiler Commands (`ingotc`)

```bash
# Compile source file and verify register locks
python src/ingotc.py examples/fe_mul.mtl -o fe_mul.ptx --target sm_86 --verify

# Display register allocation heatmap
python src/ingotc.py examples/fe_mul.mtl --reg-map
```
