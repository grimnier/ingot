"""
Ingot Lexer - Production Prototype

Strict tokenization for motor/transmission language.
- No Hermes, no agent references.
- Clean error reporting with line/column.
- Supports full Phase 1 syntax including hex, floats, compound operators, block comments.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Token:
    type: str
    value: str
    line: int
    column: int

    def __repr__(self):
        return f"Token({self.type}, {self.value!r}, {self.line}:{self.column})"

class LexerError(Exception):
    def __init__(self, msg: str, line: int, column: int):
        super().__init__(f"Lexer error at {line}:{column}: {msg}")
        self.line = line
        self.column = column

class Lexer:
    KEYWORDS = {
        'motor', 'transmission', 'in', 'out', 'inout',
        'grid', 'block', 'stream', 'for', 'while', 'if', 'else', 'return',
        'break', 'continue',
        'int', 'uint', 'float', 'double', 'i32', 'u32', 'f32', 'f64',
        'true', 'false',
        # WP1: semantic type and profile keywords
        'field', 'jacobian', 'affine',
        'target', 'profile', 'verify', 'require', 'forbid',
        # Register windows (physical register arrays with compile-time indices)
        'window',
        # Packet declarations and references
        'packet', 'use',
        # Rail, target_latch, route_type (WP8)
        'rail', 'target_latch', 'route_type',
        # Parallel GPU extensions
        '__shared__', '__syncthreads', 'shared', 'syncthreads',
        '__shfl_up_sync', '__shfl_down_sync', 'atomicAdd',
        # WP9: AI Model Layer (Manifolds, Attractors, Quantized Neural Types)
        'manifold', 'attractor', 'tensor', 'pin', 'threshold',
        'q8_0', 'q6_k', 'iq2_xxs', 'f16', 'bf16',
    }

    def __init__(self, source: str, filename: str = "<input>"):
        self.source = source
        self.filename = filename
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []

    def tokenize(self) -> List[Token]:
        self.tokens = []
        self.pos = 0
        self.line = 1
        self.column = 1
        n = len(self.source)

        while self.pos < n:
            ch = self.source[self.pos]

            # Whitespace
            if ch in ' \t\r':
                self.column += 1
                self.pos += 1
                continue
            if ch == '\n':
                self.line += 1
                self.column = 1
                self.pos += 1
                continue

            # Block comment /* ... */
            if ch == '/' and self.pos + 1 < n and self.source[self.pos+1] == '*':
                start_line, start_col = self.line, self.column
                self.pos += 2
                self.column += 2
                closed = False
                while self.pos < n:
                    if self.source[self.pos] == '*' and self.pos+1 < n and self.source[self.pos+1] == '/':
                        self.pos += 2
                        self.column += 2
                        closed = True
                        break
                    if self.source[self.pos] == '\n':
                        self.line += 1
                        self.column = 1
                    else:
                        self.column += 1
                    self.pos += 1
                if not closed:
                    raise LexerError("Unterminated block comment", start_line, start_col)
                continue

            # Line comment //
            if ch == '/' and self.pos + 1 < n and self.source[self.pos+1] == '/':
                self.pos += 2
                self.column += 2
                while self.pos < n and self.source[self.pos] != '\n':
                    self.pos += 1
                    self.column += 1
                continue

            # Window annotation @window(Ra..Rb) -- register-windowed array declaration
            if self.source.startswith('@window', self.pos):
                m = re.match(r'@window\s*\(\s*(R\d+)\s*\.\.\s*(R\d+)\s*\)', self.source[self.pos:])
                if m:
                    val = m.group(0)
                    start_reg = m.group(1)
                    end_reg = m.group(2)
                    self.tokens.append(Token('WINDOW_RANGE', f'{start_reg}..{end_reg}', self.line, self.column))
                    self.pos += len(val)
                    self.column += len(val)
                    continue
                else:
                    raise LexerError("Malformed @window annotation -- expected @window(Ra..Rb)", self.line, self.column)

            # Register annotation @register(Rn)
            if self.source.startswith('@register', self.pos):
                m = re.match(r'@register\s*\(\s*(R\d+)\s*\)', self.source[self.pos:])
                if m:
                    val = m.group(0)
                    reg = m.group(1)
                    self.tokens.append(Token('REGISTER', reg, self.line, self.column))
                    self.pos += len(val)
                    self.column += len(val)
                    continue
                else:
                    raise LexerError("Malformed @register annotation", self.line, self.column)

            # Bare REGISTER token for explicit handoff splines (R0, R1, ... inside motor_contract)
            # This enables the precision bellhousing / splined shaft syntax: "in R2 u64 addr;"
            if ch == 'R' and self.pos + 1 < n and self.source[self.pos+1].isdigit():
                m = re.match(r'R(\d+)\b', self.source[self.pos:])
                if m:
                    reg = 'R' + m.group(1)
                    self.tokens.append(Token('REGISTER', reg, self.line, self.column))
                    self.pos += len(reg)
                    self.column += len(reg)
                    continue

            # Other annotations @align etc.
            if ch == '@':
                m = re.match(r'@[a-zA-Z_][a-zA-Z0-9_]*', self.source[self.pos:])
                if m:
                    val = m.group(0)
                    self.tokens.append(Token('ANNOTATION', val, self.line, self.column))
                    self.pos += len(val)
                    self.column += len(val)
                    continue

            # Identifiers / keywords
            if ch.isalpha() or ch == '_':
                m = re.match(r'[a-zA-Z_][a-zA-Z0-9_]*', self.source[self.pos:])
                if not m:
                    raise LexerError("Unexpected character in identifier", self.line, self.column)
                val = m.group(0)
                after = self.pos + len(val)
                # threadIdx.x / blockIdx.x builtins
                if val in ('threadIdx', 'blockIdx', 'blockDim', 'gridDim') and after < n and self.source[after] == '.':
                    if after + 1 < n and self.source[after + 1] in ('x', 'y', 'z'):
                        dim = self.source[after + 1]
                        self.tokens.append(Token('BUILTIN', f'{val}.{dim}', self.line, self.column))
                        self.column += len(val) + 2
                        self.pos = after + 2
                        continue
                kind = 'KEYWORD' if val in self.KEYWORDS else 'IDENTIFIER'
                self.tokens.append(Token(kind, val, self.line, self.column))
                self.pos += len(val)
                self.column += len(val)
                continue

            # Hex numbers 0x...
            if ch == '0' and self.pos + 1 < n and self.source[self.pos+1] in 'xX':
                m = re.match(r'0[xX][0-9a-fA-F]+', self.source[self.pos:])
                if not m:
                    raise LexerError("Invalid hex number", self.line, self.column)
                val = m.group(0)
                self.tokens.append(Token('INTEGER', val, self.line, self.column))
                self.pos += len(val)
                self.column += len(val)
                continue

            # Numbers (int or float)
            if ch.isdigit():
                m = re.match(r'\d+(\.\d+)?([eE][+-]?\d+)?', self.source[self.pos:])
                if not m:
                    raise LexerError("Invalid number", self.line, self.column)
                val = m.group(0)
                if '.' in val or 'e' in val.lower():
                    self.tokens.append(Token('FLOAT', val, self.line, self.column))
                else:
                    self.tokens.append(Token('INTEGER', val, self.line, self.column))
                self.pos += len(val)
                self.column += len(val)
                continue

            # Two-char operators (must check before single-char)
            two = self.source[self.pos:self.pos+2]
            if two in ('==', '!=', '<=', '>=', '&&', '||', '>>', '<<', '..'):
                tok_type = 'DOTDOT' if two == '..' else 'OPERATOR'
                self.tokens.append(Token(tok_type, two, self.line, self.column))
                self.pos += 2
                self.column += 2
                continue

            # Single char tokens / operators
            single_map = {
                '(': 'LPAREN', ')': 'RPAREN',
                '{': 'LBRACE', '}': 'RBRACE',
                '[': 'LBRACK', ']': 'RBRACK',
                ';': 'SEMI', ',': 'COMMA', ':': 'COLON',
                '+': 'PLUS', '-': 'MINUS', '*': 'STAR', '/': 'SLASH', '%': 'PERCENT',
                '=': 'EQ', '<': 'LT', '>': 'GT', '!': 'BANG',
                '&': 'AMP', '|': 'PIPE', '^': 'CARET', '~': 'TILDE',
                '.': 'DOT',
            }
            if ch in single_map:
                self.tokens.append(Token(single_map[ch], ch, self.line, self.column))
                self.pos += 1
                self.column += 1
                continue

            raise LexerError(f"Unexpected character {ch!r}", self.line, self.column)

        self.tokens.append(Token('EOF', '', self.line, self.column))
        return self.tokens


if __name__ == "__main__":
    sample = '''// test
motor vector_add {
    @register(R0) float a;
    @register(R1) float* x;
    @register(R2) float* y;
    for (int i = 0; i < n; i = i + 1) {
        y[i] = a * x[i] + y[i];
    }
}

transmission launch {
    grid ( (n+255)/256 , 1, 1 );
    block (256, 1, 1);
    motor vector_add {
        in  float  a;
        in  float* x @align(16);
        inout float* y @align(16);
        in  int n;
    }
}
'''
    lexer = Lexer(sample)
    for t in lexer.tokenize():
        print(t)
