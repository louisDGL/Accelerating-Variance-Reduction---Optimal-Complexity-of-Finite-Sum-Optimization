"""Reconcile the citation keys used in theory/*.tex and report/sections/*.tex with bibliography/references.bib.

For every \\cite key that is not in references.bib:
  * if references.bib has an entry with the same first-author surname and year (and a similar title when the
    metadata is available in theory/theory_citations.json), the key in the .tex files is rewritten to that entry;
  * otherwise an entry is appended to references.bib from theory_citations.json (marked % ADDED-FROM-THEORY-NOTES;
    % UNVERIFIED-METADATA when no metadata is available).
Usage: python experiments/reconcile_bib.py [--dry-run]
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIB = os.path.join(ROOT, "bibliography", "references.bib")
META = os.path.join(ROOT, "theory", "theory_citations.json")
TEX = glob.glob(os.path.join(ROOT, "theory", "*.tex")) + glob.glob(os.path.join(ROOT, "report", "sections", "*.tex"))
DRY = "--dry-run" in sys.argv


def bib_entries(text):
    """key -> (entrytype, fields dict) for a BibTeX file (simple parser, enough for our generated files)."""
    out = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        start = m.end()
        depth, i = 1, start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        body = text[start:i - 1]
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|\"[^\"]*\"|[^,\n]+)", body):
            fields[fm.group(1).lower()] = fm.group(2).strip().strip("{}\"")
        out[m.group(2)] = (m.group(1).lower(), fields)
    return out


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def main():
    bib_text = open(BIB, encoding="utf-8").read()
    entries = bib_entries(bib_text)
    meta = {}
    if os.path.exists(META):
        for m in json.load(open(META, encoding="utf-8")):
            if isinstance(m, dict) and m.get("key"):
                meta[m["key"]] = m
    cited = {}
    for f in TEX:
        txt = open(f, encoding="utf-8").read()
        for cm in re.finditer(r"\\cite[tp]?\*?(?:\[[^\]]*\])?\{([^}]*)\}", txt):
            for k in cm.group(1).split(","):
                k = k.strip()
                if k:
                    cited.setdefault(k, set()).add(f)
    missing = [k for k in cited if k not in entries]
    print(f"{len(cited)} distinct cited keys, {len(missing)} missing from references.bib")
    renames = {}
    additions = []
    for k in missing:
        km = re.match(r"([a-z]+)(\d{4})(.*)", k)
        cand = []
        if km:
            surname, year = km.group(1), km.group(2)
            for ek, (et, fields) in entries.items():
                em = re.match(r"([a-z]+)(\d{4})", ek)
                if em and em.group(1) == surname and abs(int(em.group(2)) - int(year)) <= 3:
                    cand.append(ek)
        info = meta.get(k)
        chosen = None
        if cand:
            if info and info.get("title"):
                t0 = set(norm_title(info["title"]).split())
                best, score = None, 0.0
                for ek in cand:
                    t1 = set(norm_title(entries[ek][1].get("title", "")).split())
                    s = len(t0 & t1) / max(1, len(t0 | t1))
                    if s > score:
                        best, score = ek, s
                if best and score >= 0.4:
                    chosen = best
            elif len(cand) == 1:
                chosen = cand[0]
        if chosen:
            renames[k] = chosen
        else:
            additions.append(k)
    for k, v in renames.items():
        print(f"  rename {k} -> {v}")
    for k in additions:
        print(f"  add    {k} ({'metadata' if k in meta else 'NO metadata'})")
    if DRY:
        return
    # apply renames in the .tex files
    for f in TEX:
        txt = open(f, encoding="utf-8").read()
        new = txt
        for k, v in renames.items():
            new = re.sub(r"(\\cite[tp]?\*?(?:\[[^\]]*\])?\{[^}]*?)\b" + re.escape(k) + r"\b", r"\g<1>" + v, new)
        if new != txt:
            open(f, "w", encoding="utf-8").write(new)
            print("  updated", os.path.relpath(f, ROOT))
    # append missing entries
    add_text = ""
    for k in additions:
        info = meta.get(k, {})
        title = info.get("title") or k
        authors = info.get("authors") or "Unknown"
        year = info.get("year") or (re.search(r"\d{4}", k).group(0) if re.search(r"\d{4}", k) else "")
        venue = info.get("venue") or ""
        url = info.get("url") or ""
        tag = "% ADDED-FROM-THEORY-NOTES" + ("" if info else " % UNVERIFIED-METADATA")
        add_text += f"\n{tag}\n@misc{{{k},\n  title = {{{title}}},\n  author = {{{authors}}},\n  year = {{{year}}},\n  howpublished = {{{venue}}},\n  url = {{{url}}}\n}}\n"
    if add_text:
        open(BIB, "a", encoding="utf-8").write(add_text)
        print(f"  appended {len(additions)} entries to references.bib")


if __name__ == "__main__":
    main()
