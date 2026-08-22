"""Count executable code lines per package: total minus docstrings, comments, blanks."""
import ast, subprocess, sys, collections

rev = sys.argv[1]
files = subprocess.run(["git","ls-tree","-r","--name-only",rev,"src/"],
                       capture_output=True,text=True,check=True).stdout.split()
files = [f for f in files if f.endswith(".py")]

per = collections.defaultdict(lambda: [0,0,0])  # pkg -> [code, doc, blank]
for f in files:
    src = subprocess.run(["git","show",f"{rev}:{f}"],capture_output=True,text=True,check=True).stdout
    lines = src.splitlines()
    doc = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        tree = None
    if tree:
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
               and isinstance(node.value.value, str):
                doc.update(range(node.lineno, node.end_lineno + 1))
    parts = f.split("/")
    pkg = parts[2] if len(parts) > 3 else "(top-level)"
    for i, ln in enumerate(lines, 1):
        s = ln.strip()
        if not s:                      per[pkg][2] += 1
        elif i in doc or s.startswith("#"): per[pkg][1] += 1
        else:                          per[pkg][0] += 1

tot = [0,0,0]
print(f"{'package':<16}{'code':>8}{'doc':>8}{'blank':>8}{'total':>8}")
print("-"*48)
for pkg in sorted(per):
    c,d,b = per[pkg]
    for i,v in enumerate((c,d,b)): tot[i] += v
    print(f"{pkg:<16}{c:>8}{d:>8}{b:>8}{c+d+b:>8}")
print("-"*48)
print(f"{'TOTAL':<16}{tot[0]:>8}{tot[1]:>8}{tot[2]:>8}{sum(tot):>8}")
