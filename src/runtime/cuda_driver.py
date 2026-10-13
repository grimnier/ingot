"""
Ingot CUDA Driver Runtime (ctypes, pure, no pycuda)

Loads PTX produced by the Ingot compiler, allocates memory, launches kernels,
and retrieves results using only the NVIDIA CUDA Driver API via ctypes.

This is the "transmission" side in the physical metaphor: it moves data and
configures the machine exactly according to the contract declared in the source.

Works on Windows (nvcuda.dll) and Linux (libcuda.so.1).
"""

import ctypes
import os
import platform
from ctypes import c_void_p, c_uint, c_int, c_ulonglong, c_float, POINTER, byref
from typing import Any, Dict, List, Optional, Tuple

# ---- Load CUDA Driver ----

def _load_cuda_driver():
    system = platform.system()
    if system == "Windows":
        try:
            return ctypes.windll.LoadLibrary("nvcuda.dll")
        except Exception as e:
            raise RuntimeError("Failed to load nvcuda.dll - is NVIDIA driver installed?") from e
    else:
        for name in ("libcuda.so.1", "libcuda.so"):
            try:
                return ctypes.CDLL(name)
            except:
                continue
        raise RuntimeError("Failed to load libcuda.so.1")

CUDA = _load_cuda_driver()

# ---- CUDA Driver API Bindings (minimal but sufficient) ----

CUresult = c_int
CUdevice = c_int
CUcontext = c_void_p
CUmodule = c_void_p
CUfunction = c_void_p
CUdeviceptr = c_ulonglong
CUstream = c_void_p

cudaSuccess = 0

def CHECK(res):
    if res != cudaSuccess:
        raise RuntimeError(f"CUDA Driver error: {res}")

# Core functions we need
CUDA.cuInit.argtypes = [c_uint]
CUDA.cuInit.restype = CUresult

CUDA.cuDeviceGet.argtypes = [POINTER(CUdevice), c_int]
CUDA.cuDeviceGet.restype = CUresult

CUDA.cuCtxCreate_v2.argtypes = [POINTER(CUcontext), c_uint, CUdevice]
CUDA.cuCtxCreate_v2.restype = CUresult

CUDA.cuModuleLoadData.argtypes = [POINTER(CUmodule), ctypes.c_char_p]
CUDA.cuModuleLoadData.restype = CUresult

CUDA.cuModuleLoadDataEx.argtypes = [
    POINTER(CUmodule), ctypes.c_char_p,
    c_uint,            # numOptions
    POINTER(c_int),    # options array (CUjit_option enum = int)
    POINTER(c_void_p), # optionValues array
]
CUDA.cuModuleLoadDataEx.restype = CUresult

CU_JIT_MAX_REGISTERS = 0  # CUjit_option: max registers per thread

CUDA.cuModuleGetFunction.argtypes = [POINTER(CUfunction), CUmodule, ctypes.c_char_p]
CUDA.cuModuleGetFunction.restype = CUresult

CUDA.cuMemAlloc_v2.argtypes = [POINTER(CUdeviceptr), c_ulonglong]
CUDA.cuMemAlloc_v2.restype = CUresult

CUDA.cuMemcpyHtoD_v2.argtypes = [CUdeviceptr, ctypes.c_void_p, c_ulonglong]
CUDA.cuMemcpyHtoD_v2.restype = CUresult

CUDA.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, CUdeviceptr, c_ulonglong]
CUDA.cuMemcpyDtoH_v2.restype = CUresult

CUDA.cuMemFree_v2.argtypes = [CUdeviceptr]
CUDA.cuMemFree_v2.restype = CUresult

CUDA.cuLaunchKernel.argtypes = [
    CUfunction,
    c_uint, c_uint, c_uint,   # grid
    c_uint, c_uint, c_uint,   # block
    c_uint,                   # shared
    CUstream,
    POINTER(c_void_p),        # kernel params (array of pointers)
    POINTER(c_void_p),        # extra
]
CUDA.cuLaunchKernel.restype = CUresult

CUDA.cuCtxSynchronize.argtypes = []
CUDA.cuCtxSynchronize.restype = CUresult

CUDA.cuModuleUnload.argtypes = [CUmodule]
CUDA.cuModuleUnload.restype = CUresult

CUDA.cuCtxDestroy_v2.argtypes = [CUcontext]
CUDA.cuCtxDestroy_v2.restype = CUresult

class CUDADriverRuntime:
    def __init__(self):
        CHECK(CUDA.cuInit(0))
        dev = CUdevice()
        CHECK(CUDA.cuDeviceGet(byref(dev), 0))
        ctx = CUcontext()
        CHECK(CUDA.cuCtxCreate_v2(byref(ctx), 0, dev))
        self.ctx = ctx
        self.module: Optional[CUmodule] = None
        self.func: Optional[CUfunction] = None

    # Automatic Transmission Fluid Routing (P10) note:
    # Future: The transmission contract will drive automatic host<->device
    # memory movement. The driver will generate the necessary cuMemcpy
    # calls based on the declared in/out/inout parameters and geometry,
    # so the programmer never writes explicit cudaMemcpy again.

    def load_ptx(self, ptx_source: str, kernel_name: str, max_registers: int = 0):
        if isinstance(ptx_source, str):
            ptx_bytes = ptx_source.encode("utf-8")
        else:
            ptx_bytes = ptx_source
        mod = CUmodule()
        if max_registers > 0:
            opts = (c_int * 1)(CU_JIT_MAX_REGISTERS)
            vals = (c_void_p * 1)(max_registers)
            CHECK(CUDA.cuModuleLoadDataEx(byref(mod), ptx_bytes, 1, opts, vals))
        else:
            CHECK(CUDA.cuModuleLoadData(byref(mod), ptx_bytes))
        self.module = mod

        fn = CUfunction()
        CHECK(CUDA.cuModuleGetFunction(byref(fn), mod, kernel_name.encode("utf-8")))
        self.func = fn

    def destroy(self):
        """Release this runtime's module and CUDA context.

        Each CUDADriverRuntime() creates a fresh context (cuCtxCreate) and each
        load_ptx loads a module (cuModuleLoadData); without unloading/destroying
        them, repeated launches (e.g. an army wash firing hundreds of kernels)
        leak context+module memory until the driver returns CUDA_ERROR_OUT_OF_MEMORY
        (error 2). Call this when the runtime is done. Best-effort; never raises.
        """
        try:
            if getattr(self, "module", None) is not None:
                CUDA.cuModuleUnload(self.module)
                self.module = None
                self.func = None
        except Exception:
            pass
        try:
            if getattr(self, "ctx", None) is not None:
                CUDA.cuCtxDestroy_v2(self.ctx)
                self.ctx = None
        except Exception:
            pass

    def launch_simple_vector(self, x: List[float], y: List[float]) -> List[float]:
        """Robust launcher for the current realistic vector_add dev kernel (2x float*)."""
        n = len(x)
        assert len(y) == n

        d_x = CUdeviceptr()
        d_y = CUdeviceptr()
        CHECK(CUDA.cuMemAlloc_v2(byref(d_x), n * 4))
        CHECK(CUDA.cuMemAlloc_v2(byref(d_y), n * 4))

        x_arr = (c_float * n)(*x)
        y_arr = (c_float * n)(*y)
        CHECK(CUDA.cuMemcpyHtoD_v2(d_x, x_arr, n * 4))
        CHECK(CUDA.cuMemcpyHtoD_v2(d_y, y_arr, n * 4))

        params = [
            ctypes.cast(ctypes.byref(d_x), c_void_p),
            ctypes.cast(ctypes.byref(d_y), c_void_p),
        ]
        params_array = (c_void_p * len(params))(*params)

        # Ultra-safe 1x1 launch for dev kernel validation
        CHECK(CUDA.cuLaunchKernel(
            self.func,
            1, 1, 1,
            1, 1, 1,
            0, None,
            params_array, None
        ))
        CHECK(CUDA.cuCtxSynchronize())

        out = (c_float * n)()
        CHECK(CUDA.cuMemcpyDtoH_v2(out, d_y, n * 4))

        CUDA.cuMemFree_v2(d_x)
        CUDA.cuMemFree_v2(d_y)

        return list(out)

    def launch_vector_add(self, a: float, x: List[float], y: List[float]) -> List[float]:
        """Launch the compiled vector_add kernel (float a, float* x, float* y, int n)."""
        n = len(x)
        assert len(y) == n

        d_x = CUdeviceptr()
        d_y = CUdeviceptr()
        CHECK(CUDA.cuMemAlloc_v2(byref(d_x), n * 4))
        CHECK(CUDA.cuMemAlloc_v2(byref(d_y), n * 4))

        x_arr = (c_float * n)(*x)
        y_arr = (c_float * n)(*y)
        CHECK(CUDA.cuMemcpyHtoD_v2(d_x, x_arr, n * 4))
        CHECK(CUDA.cuMemcpyHtoD_v2(d_y, y_arr, n * 4))

        # Param order matches PTX signature: float a, float* x, float* y, int n
        a_val = c_float(a)
        n_val = c_uint(n)
        params = [
            ctypes.cast(ctypes.byref(a_val), c_void_p),
            ctypes.cast(ctypes.byref(d_x),  c_void_p),
            ctypes.cast(ctypes.byref(d_y),  c_void_p),
            ctypes.cast(ctypes.byref(n_val), c_void_p),
        ]
        params_array = (c_void_p * len(params))(*params)

        grid_x = max(1, (n + 255) // 256)
        CHECK(CUDA.cuLaunchKernel(
            self.func,
            grid_x, 1, 1,
            256, 1, 1,
            0, None,
            params_array, None
        ))
        CHECK(CUDA.cuCtxSynchronize())

        out = (c_float * n)()
        CHECK(CUDA.cuMemcpyDtoH_v2(out, d_y, n * 4))

        CUDA.cuMemFree_v2(d_x)
        CUDA.cuMemFree_v2(d_y)

        return list(out)

    def close(self):
        # Context is leaked for simplicity in this prototype; real code would destroy
        pass

    # ==================================================================
    # New generalized launch path (added for FINISH handoff priorities)
    # ==================================================================
    def launch_with_host_buffers(
        self,
        ptx_code: str,
        kernel_name: str,
        grid_dim: Tuple[int, int, int],
        block_dim: Tuple[int, int, int],
        host_buffers: List[Any],
        scalars: List[Any] = None,
    ) -> Dict[str, Any]:
        """
        General-purpose launcher used by the new IngotLauncher.
        Handles:
          - host buffers (numpy or bytearray-like) -> device alloc + HtoD + launch + DtoH
          - scalar args (int/float) passed by value
        Returns timing + basic success info.
        """
        scalars = scalars or []
        import time
        import numpy as np

        t0 = time.perf_counter()

        # Load module + kernel
        module = CUmodule()
        CHECK(CUDA.cuModuleLoadData(byref(module), ptx_code.encode("utf-8")))
        func = CUfunction()
        CHECK(CUDA.cuModuleGetFunction(byref(func), module, kernel_name.encode("utf-8")))

        # Prepare device pointers and param list
        d_ptrs = []
        params = []

        # Track byte sizes separately so DtoH uses the same size as HtoD.
        d_sizes = []

        # Buffers first
        for buf in host_buffers:
            if isinstance(buf, np.ndarray):
                data = buf.tobytes()
                nbytes = buf.nbytes
            elif hasattr(buf, '_type_'):
                # ctypes array: use sizeof for byte count, not len (which counts elements)
                nbytes = ctypes.sizeof(buf)
                tmp = (ctypes.c_char * nbytes)()
                ctypes.memmove(tmp, buf, nbytes)
                data = bytes(tmp)
            else:
                data = bytes(buf)
                nbytes = len(data)

            dptr = CUdeviceptr()
            CHECK(CUDA.cuMemAlloc_v2(byref(dptr), nbytes))
            CHECK(CUDA.cuMemcpyHtoD_v2(dptr, data, nbytes))
            d_ptrs.append(dptr)
            d_sizes.append(nbytes)
            params.append(ctypes.cast(ctypes.byref(dptr), c_void_p))

        # Scalars: match size to PTX .param declaration
        # .param .u32 expects 4 bytes -- c_ulonglong (8B) caused CUDA_ERROR_INVALID_PTX (218)
        # Use c_uint for integers that fit in 32 bits, c_ulonglong only for 64-bit values
        for s in scalars:
            if isinstance(s, int):
                val = c_uint(s) if s <= 0xFFFFFFFF else c_ulonglong(s)
            else:
                val = c_float(float(s))
            params.append(ctypes.cast(ctypes.byref(val), c_void_p))

        params_array = (c_void_p * len(params))(*params)

        gx, gy, gz = grid_dim
        bx, by, bz = block_dim

        CHECK(CUDA.cuLaunchKernel(
            func,
            gx, gy, gz,
            bx, by, bz,
            0, None,
            params_array, None
        ))
        CHECK(CUDA.cuCtxSynchronize())

        # Copy results back -- use the same byte counts recorded during HtoD.
        for buf, dptr, nbytes in zip(host_buffers, d_ptrs, d_sizes):
            if isinstance(buf, np.ndarray):
                out = (ctypes.c_byte * nbytes)()
                CHECK(CUDA.cuMemcpyDtoH_v2(out, dptr, nbytes))
                buf[:] = np.frombuffer(bytes(out), dtype=buf.dtype).reshape(buf.shape)
            elif hasattr(buf, '_type_'):
                # ctypes array: copy directly into the buffer's memory
                CHECK(CUDA.cuMemcpyDtoH_v2(ctypes.c_void_p(ctypes.addressof(buf)), dptr, nbytes))
            else:
                out = (ctypes.c_byte * nbytes)()
                CHECK(CUDA.cuMemcpyDtoH_v2(out, dptr, nbytes))
                if hasattr(buf, '__setitem__'):
                    for j in range(nbytes):
                        buf[j] = out[j]

            CUDA.cuMemFree_v2(dptr)

        elapsed = time.perf_counter() - t0

        return {
            "success": True,
            "elapsed_ms": elapsed * 1000,
            "grid": grid_dim,
            "block": block_dim,
            "num_buffers": len(host_buffers),
            "num_scalars": len(scalars),
        }
