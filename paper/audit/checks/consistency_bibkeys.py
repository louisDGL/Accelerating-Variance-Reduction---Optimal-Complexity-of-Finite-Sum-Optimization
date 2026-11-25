"""consistency (f): cited keys vs .bib entries; refs/labels; figure/table callouts.
Usage (from project root): python3 paper/audit/checks/consistency_bibkeys.py
"""
import re, sys, os
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
tex = open(os.path.join(root, 'paper_saga_to_muon.tex'), encoding='utf8').read()
bib = open(os.path.join(root, 'paper_saga_to_muon.bib'), encoding='utf8').read()
# strip comments
body = '\n'.join(re.sub(r'(?<!\\)%.*', '', l) for l in tex.split('\n'))
cited = set()
for m in re.finditer(r'\\cite(?:\[[^\]]*\])?\{([^}]*)\}', body):
    for k in m.group(1).split(','):
        cited.add(k.strip())
entries = re.findall(r'@(\w+)\s*\{\s*([^,\s]+)\s*,', bib)
keys = [k for t, k in entries if t.lower() not in ('comment', 'string', 'preamble')]
print('cited keys:', len(cited), ' bib entries:', len(keys), ' distinct:', len(set(keys)))
print('cited but absent from bib:', sorted(cited - set(keys)))
print('in bib but not cited:', sorted(set(keys) - cited))
dups = sorted(k for k in set(keys) if keys.count(k) > 1)
print('duplicate bib keys:', dups)
labels = re.findall(r'\\label\{([^}]*)\}', body)
refs = set(re.findall(r'\\(?:eq)?ref\{([^}]*)\}', body))
print('labels:', len(labels), 'dups:', sorted(l for l in set(labels) if labels.count(l) > 1))
print('refs to undefined labels:', sorted(refs - set(labels)))
print('labels never referenced:', sorted(set(labels) - refs))
for l in sorted(set(labels)):
    n = len(re.findall(r'\\(?:eq)?ref\{' + re.escape(l) + r'\}', body))
    if l.startswith(('fig:', 'tab:')):
        print('  callouts of', l, ':', n)
