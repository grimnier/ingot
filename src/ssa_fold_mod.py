"""
ssa_fold_mod -- prove the user's insight: rename each write to its own version (a,b,c...), then the
SAME safe copy-prop folds ALL the parking copies (not just the 7 the name-based version dared touch).

  ssa_rename_and_fold(lines):
    1. SSA-rename the generic %rd temps (each definition -> a fresh %vN). Locked %R / %r / %f / %p
       are left alone.  (straight-line kernel -> linear renaming is exact.)
    2. Copy-prop fold: in SSA every `mov %vA, %x` is single-def, so substituting %vA->%x is always
       value-safe. Fold them all.
    3. Declare the new %v bank up front.
"""
import re

_RD = re.compile(r"%rd\d+")


def _parse(ln):
    s = ln.strip()
    if not s or s.startswith(("//", ".", "{", "}")) or s.endswith(":"):
        return None
    m = re.match(r"([\w.]+)\s+(.*);\s*$", s)
    return (m.group(1), [a.strip() for a in m.group(2).split(",")]) if m else None


def _wreg(opc, args):
    if not args or opc[:2] in ("st", "br", "re", "at") or opc.startswith("@"):
        return None
    m = re.match(r"^(%\w+)$", args[0])
    return m.group(1) if m else None


def _ssa(lines):
    cur, cnt = {}, [0]
    def fresh():
        nm = f"%v{cnt[0]}"; cnt[0] += 1; return nm
    out = []
    for ln in lines:
        p = _parse(ln)
        if not p:
            out.append(ln); continue
        opc, args = p
        w = _wreg(opc, args)
        indent = ln[:len(ln) - len(ln.lstrip())]
        sub = lambda a: _RD.sub(lambda m: cur.get(m.group(0), m.group(0)), a)
        if w and w.startswith("%rd"):
            new_rest = [sub(a) for a in args[1:]]   # reads use the CURRENT version
            nw = fresh(); cur[w] = nw               # the write starts a NEW version
            out.append(f"{indent}{opc} {', '.join([nw] + new_rest)};")
        else:
            out.append(f"{indent}{opc} {', '.join(sub(a) for a in args)};")
    return out, cnt[0]


def _fold(lines):
    # PTX special registers that are live hardware reads -- they must NOT be
    # copy-propagated.  Substituting %globaltimer or %clock at every use site
    # would cause the timer to be re-read at each use rather than being captured
    # once at the clock64()/clock32() call point, which violates the intrinsic's
    # semantics (the user expects a frozen start value).
    _NO_FOLD_SRCS = frozenset({
        "%globaltimer", "%globaltimer_lo", "%globaltimer_hi",
        "%clock", "%clock64",
        "%laneid", "%warpid", "%smid", "%nsmid",
        "%tid.x", "%tid.y", "%tid.z",
        "%ctaid.x", "%ctaid.y", "%ctaid.z",
        "%ntid.x", "%ntid.y", "%ntid.z",
        "%nctaid.x", "%nctaid.y", "%nctaid.z",
    })
    changed = True
    while changed:
        changed = False
        for i, ln in enumerate(lines):
            p = _parse(ln)
            if not p:
                continue
            opc, args = p
            if (opc.startswith("mov") and len(args) == 2
                    and re.match(r"^%v\d+$", args[0]) and re.match(r"^%\w+$", args[1])):
                a, b = args
                if b in _NO_FOLD_SRCS:
                    continue  # never propagate live hardware register reads
                word = re.compile(r"(?<![\w%])" + re.escape(a) + r"(?![\w])")
                lines = [(word.sub(b, l) if j != i else l) for j, l in enumerate(lines)]
                lines = lines[:i] + lines[i+1:]
                changed = True
                break
    return lines


def ssa_rename_and_fold(lines):
    renamed, nv = _ssa(lines)
    folded = _fold(renamed)
    decl = [f"    .reg .u64 %v<{nv}>;"] if nv else []
    return decl + folded
