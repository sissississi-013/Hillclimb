import ast, importlib, inspect, sys
files = sys.argv[1:]
bad = 0
for f in files:
    tree = ast.parse(open(f).read())
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.Module, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        aliases = {}
        for sub in ast.walk(node):
            if isinstance(sub, ast.ImportFrom) and sub.module and sub.module.split(".")[0] in ("robots", "dream"):
                for a in sub.names:
                    aliases[a.asname or a.name] = f"{sub.module}.{a.name}"
            elif isinstance(sub, ast.Import):
                for a in sub.names:
                    if a.name.split(".")[0] in ("robots", "dream"):
                        aliases[(a.asname or a.name).split(".")[0]] = (a.asname and a.name) or a.name.split(".")[0]
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute):
                chain = []
                cur = sub
                while isinstance(cur, ast.Attribute):
                    chain.append(cur.attr); cur = cur.value
                if isinstance(cur, ast.Name) and cur.id in aliases:
                    chain.reverse()
                    path = aliases[cur.id]
                    try:
                        mod_path, _, name = path.rpartition(".")
                        try:
                            obj = importlib.import_module(path)
                        except Exception:
                            obj = getattr(importlib.import_module(mod_path), name)
                        for i, c in enumerate(chain[:1]):
                            if not hasattr(obj, c):
                                try:
                                    obj = importlib.import_module(f"{path}.{c}")
                                    continue
                                except Exception:
                                    pass
                                print(f"MISSING {f}:{sub.lineno} {path}.{c}"); bad += 1
                    except Exception as e:
                        print(f"IMPORTFAIL {f}:{sub.lineno} {path}: {e}"); bad += 1
        for alias, path in aliases.items():
            mod_path, _, name = path.rpartition(".")
            try:
                importlib.import_module(path)
            except Exception:
                try:
                    getattr(importlib.import_module(mod_path), name)
                except Exception as e:
                    print(f"IMPORTFAIL {f} {path}: {e}"); bad += 1
print("problems:", bad)
