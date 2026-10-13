# Super simple runner for your first Ingot program
# This lets you write code in minimal.mtl and actually see a result.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from compiler import Compiler
from runtime.launcher import IngotLauncher
import ctypes

def main():
    print("=== Ingot Minimal Runner ===\n")

    mtl_file = Path(__file__).parent / "minimal.mtl"
    source = mtl_file.read_text()

    # Compile
    c = Compiler()
    ptx, prog = c.compile_file(str(mtl_file))
    print("Compiled successfully!")
    print(f"Motors: { [m.name for m in prog.motors] }")
    print(f"Transmissions: { [t.name for t in prog.transmissions] }\n")

    # Prepare tiny input/output buffers
    input_buf = (ctypes.c_float * 1)(42.0)   # input value
    output_buf = (ctypes.c_float * 1)(0.0)   # where the answer will go

    # Launch using the proper generalized launcher (best path right now)
    launcher = IngotLauncher(verbose=True)
    # Try launching. This may require numpy in some driver paths.
    try:
        result = launcher.launch_buffers(
            ptx,
            kernel_name="minimal_minimal_launch",
            host_buffers=[input_buf, output_buf],
            source_for_contract=source,
            strict_explicit=True
        )
        success = True
    except Exception as e:
        print(f"Launch attempt note: {e}")
        print("The PTX compiled and is valid. Actual GPU execution can need extra setup.")
        success = False

    if success:
        print("\n=== Result ===")
        print(f"Input was:  {input_buf[0]}")
        print(f"Output is:  {output_buf[0]}")
        print(f"Expected:   {input_buf[0] + 1.0}")

if __name__ == "__main__":
    main()
