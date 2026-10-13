# Ingot Grammar Specification (Working Draft)

**Temporary Project Name**: Ingot (Motor/Transmission Language) — final name TBD. This project is completely independent of any agent or orchestration systems.
**Status**: Phase 1 Draft

## Lexical Structure

### Keywords
`motor`, `transmission`, `in`, `out`, `inout`, `grid`, `block`, `stream`, `for`, `if`, `else`, `return`, `true`, `false`

### Annotations
`@register(...)`, `@align(...)`, `@shared`, `@texture`, `@const`

### Register Binding
`@register(R0)` through `@register(R255)` or symbolic names in the future.

### Identifiers
Standard C-style identifiers, starting with letter or underscore.

### Literals
- Integer literals (decimal, hex with 0x)
- Floating point literals
- Boolean literals

## Syntax

### Top Level

```
program
    : top_level_decl*
    ;

top_level_decl
    : motor_decl
    | transmission_decl
    | host_function
    ;
```

### Motor Declaration

```
motor_decl
    : 'motor' IDENTIFIER (register_annotation)? '{' motor_statement* '}'
    ;

register_annotation
    : '@register' '(' REGISTER ')' 
    ;

REGISTER
    : 'R' [0-9]+
    ;

motor_statement
    : variable_decl
    | shared_decl
    | syncthreads_stmt
    | atomic_add_stmt
    | assignment
    | for_loop
    | if_stmt
    | return_stmt
    | expr ';'
    ;

variable_decl
    : (register_annotation)? type IDENTIFIER bracket_dim* ';'
    ;

bracket_dim
    : '[' expr ']'
    | '[' ']'          // unsized leading dimension of a pointer: T* name[][12][4]
    ;

shared_decl
    : ('__shared__' | 'shared') type IDENTIFIER bracket_dim+ ';'
    ;

syncthreads_stmt
    : ('__syncthreads' | 'syncthreads') '('? ')'? ';'
    ;

atomic_add_stmt
    : 'atomicAdd' '(' expr ',' expr ')' ';'
    ;
```

### Transmission Declaration

```
transmission_decl
    : 'transmission' IDENTIFIER '{' 
        grid_decl
        block_decl
        (stream_decl)?
        motor_binding
      '}'
    ;

grid_decl
    : 'grid' '(' dim3 ')' ';'
    ;

block_decl
    : 'block' '(' dim3 ')' ';'
    ;

stream_decl
    : 'stream' IDENTIFIER ';'
    ;

dim3
    : expr (',' expr (',' expr)? )?
    ;

motor_binding
    : 'motor' IDENTIFIER '{' transmission_param* '}'
    ;

transmission_param
    : direction type IDENTIFIER (align_annotation)? ';'
    ;

direction
    : 'in' | 'out' | 'inout'
    ;

align_annotation
    : '@align' '(' INTEGER ')'
    ;
```

### Host Code (Phase 1)
Host code is currently allowed as normal C-style functions outside of motor/transmission blocks. Future phases will introduce stricter separation.

## Register Semantics

- Variables annotated with `@register(Rn)` are **hard-bound** to that physical register.
- The compiler backend **must not** spill these variables to local memory.
- Taking the address of a register-bound variable is only allowed in very restricted ways (future rule).

This is one of the core differentiators from CUDA C++.

## Notes

This grammar is intentionally minimal for Phase 1. We will expand it as we implement the parser and discover needs.