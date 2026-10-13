"""
Host/device marshalling codegen for Ingot transmissions.
Generates C++ host-side kernel launch stubs from a compiled Ingot program.
"""
from pathlib import Path
from typing import Optional
import sys

# Import AST nodes (will be available when loaded via the compiler)
try:
    from ast_nodes import Program, TransmissionDecl, MotorDecl
except ImportError:
    pass


def generate_host_stub(program, ptx_text: str, out_path: Optional[Path] = None) -> str:
    """
    Generate a C++ host stub (.cuh) for calling the compiled Ingot kernel.

    Returns the generated C++ source as a string.
    If out_path is provided, also writes it to disk.
    """
    lines = []
    lines.append("// Auto-generated Ingot host stub — do not edit manually")
    lines.append("#pragma once")
    lines.append("#include <cuda_runtime.h>")
    lines.append("#include <stdint.h>")
    lines.append("")

    for tx in getattr(program, 'transmissions', []):
        kernel_name = _kernel_name(tx)
        params = getattr(tx, 'params', []) or []
        motor_params = []
        if tx.motor_contract:
            motor_params = getattr(tx.motor_contract, 'params', []) or []

        # Use motor_contract params if tx has no direct params
        effective_params = params if params else motor_params

        # extern "C" declaration
        c_params = _params_to_c(effective_params)
        lines.append(f'extern "C" __global__ void {kernel_name}({c_params});')
        lines.append("")

        # Grid/block from transmission
        grid = _dim_to_str(getattr(tx, 'grid', None))
        block = _dim_to_str(getattr(tx, 'block', None))

        # C++ wrapper function
        fn_params = _params_to_c_fn(effective_params)
        lines.append(f"inline void launch_{tx.name}({fn_params}) {{")
        lines.append(f"    dim3 grid{grid};")
        lines.append(f"    dim3 block{block};")
        lines.append(f"    {kernel_name}<<<grid, block>>>({_param_names(effective_params)});")
        lines.append("    cudaDeviceSynchronize();")
        lines.append("}")
        lines.append("")

    result = "\n".join(lines)
    if out_path:
        out_path.write_text(result, encoding="utf-8")
    return result


def _kernel_name(tx) -> str:
    """Derive the PTX entry-point name for a TransmissionDecl.

    The PTX emitter names kernels as <motor_name>_sm89 when a motor contract
    is present, falling back to <tx_name>_kernel for geometry-only transmissions.
    """
    if tx.motor_contract and getattr(tx.motor_contract, 'motor_name', None):
        return f"{tx.motor_contract.motor_name}_sm89"
    return f"{tx.name}_kernel"


def _expr_to_c(e) -> str:
    """Render an MTLang AST expression node as a C expression string."""
    if e is None:
        return "0"
    # IntLit / FloatLit
    if hasattr(e, 'raw') and hasattr(e, 'value') and not hasattr(e, 'name'):
        return str(e.raw)
    # Var
    if hasattr(e, 'name') and not hasattr(e, 'left'):
        return str(e.name)
    # BinOp
    if hasattr(e, 'left') and hasattr(e, 'op') and hasattr(e, 'right'):
        return f"({_expr_to_c(e.left)} {e.op} {_expr_to_c(e.right)})"
    # UnaryOp
    if hasattr(e, 'op') and hasattr(e, 'expr') and not hasattr(e, 'left'):
        return f"({e.op}{_expr_to_c(e.expr)})"
    # Fallback: plain integer or string
    if hasattr(e, 'value'):
        return str(e.value)
    return str(e)


def _dim_to_str(dim) -> str:
    """Convert a grid/block dim (list of Expr or None) to a dim3 constructor string."""
    if dim is None or (isinstance(dim, list) and len(dim) == 0):
        return "(1, 1, 1)"
    if isinstance(dim, list):
        x = _expr_to_c(dim[0]) if len(dim) > 0 else "1"
        y = _expr_to_c(dim[1]) if len(dim) > 1 else "1"
        z = _expr_to_c(dim[2]) if len(dim) > 2 else "1"
        return f"({x}, {y}, {z})"
    # Scalar / already a string
    if hasattr(dim, 'x'):
        return f"({dim.x}, {dim.y or 1}, {dim.z or 1})"
    return f"({dim})"


def _type_to_c(t) -> str:
    """Map an MTLang type string to its closest C/CUDA equivalent."""
    if t is None:
        return "void*"
    ts = str(t).lower().strip()
    is_ptr = "*" in ts or ts.endswith("[]")
    # Strip pointer sigils for matching
    base = ts.replace("*", "").replace("[]", "").strip()
    if base in ("float", "f32"):
        return "float*" if is_ptr else "float"
    if base in ("double", "f64"):
        return "double*" if is_ptr else "double"
    if base in ("u64", "uint64", "uint64_t"):
        return "uint64_t*" if is_ptr else "uint64_t"
    if base in ("u32", "uint32", "uint32_t"):
        return "uint32_t*" if is_ptr else "uint32_t"
    if base in ("u16", "uint16", "uint16_t"):
        return "uint16_t*" if is_ptr else "uint16_t"
    if base in ("u8", "uint8", "uint8_t", "byte"):
        return "uint8_t*" if is_ptr else "uint8_t"
    if base in ("i64", "int64", "int64_t"):
        return "int64_t*" if is_ptr else "int64_t"
    if base in ("i32", "int32", "int32_t", "int"):
        return "int32_t*" if is_ptr else "int32_t"
    # Unknown / geometric types (field, affine<...>, jacobian<...>) -> void*
    return "void*"


def _params_to_c(params) -> str:
    """Render a list of TxParam nodes as a C parameter list string."""
    if not params:
        return ""
    parts = []
    for p in params:
        raw_type = getattr(p, 'type', None) or getattr(p, 'typ', None)
        typ = _type_to_c(raw_type)
        name = getattr(p, 'name', 'param')
        parts.append(f"{typ} {name}")
    return ", ".join(parts)


def _params_to_c_fn(params) -> str:
    return _params_to_c(params)


def _param_names(params) -> str:
    if not params:
        return ""
    return ", ".join(getattr(p, 'name', 'param') for p in params)


if __name__ == "__main__":
    # CLI: python codegen_host.py file.mtl  ->  file.cuh
    sys.path.insert(0, str(Path(__file__).parent))
    from compiler import Compiler
    if len(sys.argv) < 2:
        print("Usage: codegen_host.py <file.mtl>")
        sys.exit(1)
    src = Path(sys.argv[1])
    c = Compiler()
    ptx, prog = c.compile_file(src)
    out = src.with_suffix(".cuh")
    stub = generate_host_stub(prog, ptx, out)
    print(f"Written: {out}")
    print(stub)
