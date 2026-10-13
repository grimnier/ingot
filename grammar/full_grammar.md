# Ingot Full Grammar Reference (Phase 1 Draft)

## Lexical Rules

### Keywords
`motor`, `transmission`, `in`, `out`, `inout`, `grid`, `block`, `stream`, `for`, `if`, `else`, `return`

### Annotations
`@register(Rn)`, `@align(N)`, `@shared`, `@texture`, `@const`

### Identifiers
`[a-zA-Z_][a-zA-Z0-9_]*`

### Registers
`R[0-9]+`

## Syntax Grammar (EBNF)

```
program
    : top_level_decl* EOF
    ;

top_level_decl
    : motor_decl
    | transmission_decl
    | host_function
    ;

motor_decl
    : 'motor' ID (register_annotation)? '{' motor_stmt* '}'
    ;

register_annotation
    : '@register' '(' REGISTER ')'
    ;

motor_stmt
    : var_decl
    | assignment
    | for_stmt
    | if_stmt
    | return_stmt
    | expr ';'
    ;

var_decl
    : (register_annotation)? type ID array_suffix? ';'
    ;

transmission_decl
    : 'transmission' ID '{' 
        grid_decl
        block_decl
        (stream_decl)?
        motor_contract
      '}'
    ;

motor_contract
    : 'motor' ID '{' tx_param* '}'
    ;

tx_param
    : direction type ID (align_annotation)? ';'
    ;

direction
    : 'in' | 'out' | 'inout'
    ;

align_annotation
    : '@align' '(' INTEGER ')'
    ;
```

## Notes on Current State

This is still a working draft. The grammar will be refined as we implement the parser and discover edge cases.

Register annotations are currently only allowed on variable declarations inside motors and on parameters inside transmission motor contracts.
