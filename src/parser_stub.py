"""
Ingot Parser Stub (Phase 1 Prototype)

This is a minimal skeleton whose only job right now is to recognize the top-level
structure of a program (motor and transmission declarations) and perform very
basic semantic checks.

It is intentionally simple. The real parser will be written once the grammar
is locked.
"""

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class MotorDecl:
    name: str
    register_locked: bool = False

@dataclass
class TransmissionDecl:
    name: str
    motor_name: str
    has_grid_block: bool = False

class ParseError(Exception):
    pass

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def parse_program(self) -> List:
        decls = []
        while not self._at_end():
            if self._match("motor"):
                decls.append(self._parse_motor())
            elif self._match("transmission"):
                decls.append(self._parse_transmission())
            else:
                self._advance()  # skip host code for now
        return decls

    def _parse_motor(self) -> MotorDecl:
        name = self._consume_identifier()
        register_locked = self._match_register_annotation()
        # Skip body for Phase 1 stub
        self._skip_until_matching_brace()
        return MotorDecl(name, register_locked)

    def _parse_transmission(self) -> TransmissionDecl:
        name = self._consume_identifier()
        self._skip_until_matching_brace()  # very crude
        # In a real parser we would extract the motor reference here
        return TransmissionDecl(name, motor_name="unknown")

    # --- Extremely minimal helpers ---
    def _match(self, keyword: str) -> bool:
        if self._current() == keyword:
            self._advance()
            return True
        return False

    def _match_register_annotation(self) -> bool:
        if self._current() and self._current().startswith("@register"):
            self._advance()
            return True
        return False

    def _consume_identifier(self) -> str:
        tok = self._current()
        if tok and tok[0].isalpha():
            self._advance()
            return tok
        raise ParseError("Expected identifier")

    def _skip_until_matching_brace(self):
        depth = 0
        while not self._at_end():
            tok = self._current()
            if tok == "{":
                depth += 1
            elif tok == "}":
                depth -= 1
                if depth == 0:
                    self._advance()
                    return
            self._advance()

    def _current(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _advance(self):
        if not self._at_end():
            self.pos += 1

    def _at_end(self):
        return self.pos >= len(self.tokens)
