import ast, importlib, inspect, sys
def resolve(path):
    parts = path.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except Exception:
            continue
        for p in parts[i:]:
            obj = getattr(obj, p)
        return obj
bad = 0
for f in sys.argv[1:]:
    tree = ast.parse(open(f).read())
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.Module)):
            continue
        aliases = {}
        for s in ast.walk(fn):
            if isinstance(s, ast.ImportFrom) and s.module and s.module.split(".")[0] in ("robots", "dream"):
                for a in s.names: aliases[a.asname or a.name] = f"{s.module}.{a.name}"
            elif isinstance(s, ast.Import):
                for a in s.names:
                    if a.name.split(".")[0] in ("robots", "dream"): aliases[a.name.split(".")[0]] = a.name.split(".")[0]
        for c in ast.walk(fn):
            if not isinstance(c, ast.Call): continue
            chain, cur = [], c.func
            while isinstance(cur, ast.Attribute): chain.append(cur.attr); cur = cur.value
            if not (isinstance(cur, ast.Name) and cur.id in aliases): continue
            path = ".".join([aliases[cur.id]] + chain[::-1])
            try:
                obj = resolve(path)
                sig = inspect.signature(obj)
            except Exception as e:
                continue
            if any(isinstance(a, ast.Starred) for a in c.args) or any(k.arg is None for k in c.keywords): continue
            try:
                sig.bind(*([None] * len(c.args)), **{k.arg: None for k in c.keywords})
            except TypeError as e:
                print(f"SIG {f}:{c.lineno} {path}{sig}: {e}"); bad += 1
print("sig problems:", bad)
