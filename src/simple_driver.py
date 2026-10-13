#!/usr/bin/env python3
"""
Ingot Simple Driver (Phase 1 Prototype)

This is a throw-away driver used to exercise the early components:
lexer + parser_stub + semantic analyzer.

Usage:
    python simple_driver.py path/to/program.ingot

It will eventually evolve into the real `ingotc` compiler driver.
"""

import sys
from pathlib import Path

# Very crude tokenization for the prototype driver
# In a real implementation we would use the proper Lexer class
def crude_tokenize(source: str):
    tokens = []
    for line in source.splitlines():
        for tok in line.replace(';', ' ; ').replace('{', ' { ').replace('}', ' } ').split():
            if tok.strip():
                tokens.append(tok)
    return tokens

def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_driver.py <file.ingot>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    source = path.read_text()
    tokens = crude_tokenize(source)

    print("=== Tokens (crude) ===")
    print(tokens[:50], "..." if len(tokens) > 50 else "")

    # Very rough structure detection for demo purposes
    has_motor = "motor" in tokens
    has_transmission = "transmission" in tokens

    print("\n=== Very Early Semantic Checks ===")
    if has_motor and has_transmission:
        print("Found both 'motor' and 'transmission' declarations.")
        print("In a real compiler we would now run full contract checking.")
    else:
        print("Warning: Did not find both motor and transmission in this file (demo only).")

    print("\nPrototype driver finished successfully (no real compilation yet).")

if __name__ == "__main__":
    main()
