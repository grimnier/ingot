# Ingot High-Level IR (HIR) - Initial Design

## Goals
- Represent the program after parsing and basic semantic checking.
- Preserve the Motor / Transmission separation.
- Make register annotations explicit and first-class.

## Node Kinds

### Program
Root node containing a list of top-level declarations.

### Motor
- name: string
- register_hints: map<variable_name, register>
- body: list of statements (still fairly high level at this stage)

### Transmission
- name: string
- grid: dim3
- block: dim3
- stream: optional
- motor_contract: reference to a Motor + its parameter directions/types/alignments

### RegisterLockedVariable
Special variable node that carries a hard register binding.

## Lowering Notes

From HIR we will lower to a Mid-level IR (MIR) that is closer to PTX:
- Explicit load/store operations with address spaces.
- Phi nodes or explicit register moves for control flow.
- Hard register constraints attached to values.

Register locking should survive all the way to the final PTX emission.
