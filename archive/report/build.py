"""Build the two PDFs of the report, in the order they require.

Usage: python report/build.py            (from any directory; needs Python, pdflatex and bibtex)
  1. python report/make_restated.py: statements of the appendix results restated in the main body (report/restated/);
  2. report/main.tex      -> report/main.pdf       (pdflatex, bibtex, pdflatex, pdflatex);
  3. report/synthesis.tex -> report/synthesis.pdf  (same; its cross-references are read from main.aux, hence step 2 first).
For each document the script prints the number of pages, the LaTeX errors, the undefined references and the overfull
boxes wider than 5pt, and it exits with status 1 if a document has an error or an undefined reference.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    return subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, encoding="utf-8", errors="replace")


def build(doc):
    for cmd in (["pdflatex", "-interaction=nonstopmode", doc + ".tex"], ["bibtex", doc],
                ["pdflatex", "-interaction=nonstopmode", doc + ".tex"], ["pdflatex", "-interaction=nonstopmode", doc + ".tex"]):
        run(cmd)
    log = open(os.path.join(HERE, doc + ".log"), encoding="utf-8", errors="replace").read()
    errors = re.findall(r"^! .*", log, flags=re.M)
    undefined = sorted(set(re.findall(r"(?:Reference|Citation) `([^']+)' on page \d+ undefined", log))
                       | set(re.findall(r"Unknown result `([^']+)'", log)))
    overfull = [float(x) for x in re.findall(r"Overfull \\hbox \((\d+\.?\d*)pt too wide\)", log)]
    pages = re.search(r"Output written on .*?\((\d+) pages", log)
    print(f"{doc}.pdf: {pages.group(1) if pages else '?'} pages; {len(errors)} errors; {len(undefined)} undefined "
          f"references; {sum(o > 5 for o in overfull)} overfull boxes wider than 5pt")
    for e in errors[:10]:
        print("   ", e)
    if undefined:
        print("    undefined:", ", ".join(undefined[:20]))
    return not errors and not undefined


def main():
    gen = subprocess.run([sys.executable, os.path.join(HERE, "make_restated.py")], capture_output=True, text=True)
    print(gen.stdout.strip() or gen.stderr.strip())
    if gen.returncode:
        sys.exit(1)
    ok = build("main")
    ok = build("synthesis") and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
