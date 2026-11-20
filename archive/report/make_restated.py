"""Generate report/restated/: verbatim copies of the statements of the theory notes, for the main body of the report.

Usage: python report/make_restated.py        (run it before compiling report/main.tex or report/synthesis.tex)

For every labelled theorem-like environment (theorem, proposition, lemma, corollary, definition, assumption, remark,
example) of the five notes theory/{B3,B2,B1,A1,A3}_*.tex, the script writes
  * report/restated/<label with ':' replaced by '-'>.tex, containing
        \\begin{restated}{<Type>}{<label>}{<appendix>}{<1 if the note proves it, else 0>}{<name>}
        <statement>
        \\end{restated}
    where <statement> is the body of the environment with its \\label commands removed and its displayed equations
    made unnumbered, each \\label{X} of a displayed equation becoming \\tag{\\ref*{X}}: the restatement shows the
    equation numbers of the appendix and defines no new label. The fourth argument is 0 if the note does not prove the
    result (definitions, assumptions, remarks, results quoted from the literature), 1 if a proof environment follows
    the statement, and prf:<label> if the proof is elsewhere and starts with \\phantomsection\\label{prf:<label>}
    (the pointer printed after the restatement then gives the page of the proof). If the statement ends with a display,
    \\postdisplaypenalty=10000 is set before it, so that the pointer cannot be separated from the display;
  * one line \\restatedmeta{<label>}{<Type>}{<appendix>}{<name>} per environment in report/restated/names.tex, from which
    \\resref{<label>} prints "<Type> <number> (<name>)" (report/restate_defs.tex).
The body of the report restates a result with \\restate{<label>}. The notes stay the only source of the statements:
edit the note, never a generated file. The script checks that every label used with \\restate or \\resref in
report/sections/ exists in the notes and exits with status 1 otherwise.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTES = [("B3", "B3_prop59_tightness"), ("B2", "B2_effective_curvature"), ("B1", "B1_preconditioned_saga"),
         ("A1", "A1_lower_bound"), ("A3", "A3_lkatyusha_lyapunov")]
ENVS = ["theorem", "proposition", "lemma", "corollary", "definition", "assumption", "remark", "example"]
PROVABLE = {"theorem", "proposition", "lemma", "corollary"}
MATH = ["equation", "align", "gather", "multline", "flalign", "alignat", "eqnarray"]
BS = "\\"
OUT = os.path.join(ROOT, "report", "restated")


def optional_argument(text):
    """(argument, rest) for an optional argument [..] at the start of text (balanced brackets), else ("", text)."""
    t = text.lstrip()
    if not t.startswith("["):
        return "", text
    depth = 0
    for j, ch in enumerate(t):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return t[1:j], t[j + 1:]
    raise ValueError("unbalanced optional argument: " + t[:80])


def strip_comments_for_search(text):
    """Text with TeX comments blanked (same length), so that searches ignore commented-out code."""
    return re.sub(r"(?<!\\)%[^\n]*", lambda m: " " * len(m.group(0)), text)


def convert_math(body):
    """Make displayed equations unnumbered; their labels become \\tag{\\ref*{label}}."""
    pat = re.compile(r"\\begin\{(" + "|".join(MATH) + r")\}")
    out, pos = [], 0
    while True:
        m = pat.search(body, pos)
        if not m:
            out.append(body[pos:])
            break
        env = m.group(1)
        end = body.find(BS + "end{" + env + "}", m.end())
        if end < 0:
            raise ValueError("unterminated " + env)
        inner = body[m.end():end]
        inner = re.sub(r"\\label\{([^}]*)\}", lambda k: BS + "tag{" + BS + "ref*{" + k.group(1) + "}}", inner)
        out.append(body[pos:m.start()] + BS + "begin{" + env + "*}" + inner + BS + "end{" + env + "*}")
        pos = end + len(BS + "end{" + env + "}")
    return "".join(out)


def protect_last_display(body):
    """If the statement ends with a displayed equation, forbid a page break between it and the pointer that follows."""
    tail = body.rstrip()
    ends = re.search(r"(\\end\{(?:" + "|".join(MATH) + r")\*\}|\\\])\s*$", tail)
    if not ends:
        return body
    starts = list(re.finditer(r"\\begin\{(?:" + "|".join(MATH) + r")\*\}|\\\[", body))
    if not starts:
        return body
    k = starts[-1].start()
    return body[:k] + BS + "postdisplaypenalty=10000 " + body[k:]


def statements():
    """Yield (appendix, note, env, label, name, body, proved) for every labelled theorem-like environment."""
    pat = re.compile(r"\\begin\{(" + "|".join(ENVS) + r")\}")
    for app, note in NOTES:
        src = open(os.path.join(ROOT, "theory", note + ".tex"), encoding="utf-8").read()
        search = strip_comments_for_search(src)
        proofs_of = set(re.findall(r"\\begin\{proof\}\[[^\]]*?\\ref\{([^}]*)\}", search))
        proof_labels = set(re.findall(r"\\label\{(prf:[^}]*)\}", search))
        for m in pat.finditer(search):
            env = m.group(1)
            end = search.find(BS + "end{" + env + "}", m.end())
            name, rest = optional_argument(src[m.end():end])
            lab = re.match(r"\s*\\label\{([^}]*)\}", rest)
            if not lab:
                continue
            label = lab.group(1)
            if label.count(":") != 1:
                sys.exit(f"label {label!r} of {note}.tex must contain exactly one colon")
            body = rest[lab.end():]
            body = convert_math(body)
            body = re.sub(r"\\label\{[^}]*\}", "", body)
            body = protect_last_display(body)
            after = search[end + len(BS + "end{" + env + "}"):]
            adjacent = re.match(r"\s*\\begin\{proof\}", after) is not None
            if env not in PROVABLE or not (adjacent or label in proofs_of):
                proved = "0"
            elif not adjacent and "prf:" + label in proof_labels:
                proved = "prf:" + label
            else:
                proved = "1"
            yield app, note, env, label, name.strip(), body.strip("\n"), proved


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        if f.endswith(".tex"):
            os.remove(os.path.join(OUT, f))
    names, known = [], set()
    for app, note, env, label, name, body, proved in statements():
        known.add(label)
        typ = env.capitalize()
        head = (f"% Generated by report/make_restated.py from theory/{note}.tex: do not edit (edit the note and rerun).\n"
                f"{BS}begin{{restated}}{{{typ}}}{{{label}}}{{{app}}}{{{proved}}}{{{name}}}\n")
        with open(os.path.join(OUT, label.replace(":", "-") + ".tex"), "w", encoding="utf-8", newline="\n") as f:
            f.write(head + body + "\n" + BS + "end{restated}\n")
        names.append(f"{BS}restatedmeta{{{label}}}{{{typ}}}{{{app}}}{{{name}}}")
    with open(os.path.join(OUT, "names.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write("% Generated by report/make_restated.py: type, appendix and name of every labelled result of the notes.\n")
        f.write("\n".join(names) + "\n")
    used = set()
    paths = [os.path.join(dp, fn) for dp, _, fns in os.walk(os.path.join(ROOT, "report", "sections")) for fn in fns]
    paths += [os.path.join(ROOT, "report", fn) for fn in os.listdir(os.path.join(ROOT, "report"))]
    for path in paths:
        if path.endswith(".tex"):
            text = strip_comments_for_search(open(path, encoding="utf-8").read())
            used |= set(re.findall(r"\\(?:restate|resref\*?)\{([^}]*)\}", text))
    used.discard("label")
    missing = sorted(used - known)
    print(f"{len(names)} statements written to report/restated/; {len(used)} labels used in the report")
    if missing:
        print("labels used with \\restate or \\resref but absent from the notes: " + ", ".join(missing))
        sys.exit(1)


if __name__ == "__main__":
    main()
