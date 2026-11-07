#!/usr/bin/env bash
# Station bring-up: check every dream/robots attribute and call signature our code uses against the INSTALLED Dream
# library (static AST scan + import; nothing moves). Built against dreamlabs 1.0.37 + dreamlabs-client 1.1.3.
#   bash tools/api_check.sh            # uses ~/dream-env/bin/python
set -euo pipefail
PY=${PY:-$HOME/dream-env/bin/python}
cd "$(dirname "$0")/.."
"$PY" -c "import importlib.metadata as m; print('dreamlabs', m.version('dreamlabs'), '| client', m.version('dreamlabs-client'))" || true
"$PY" tools/_api_attrs.py hillclimb/*.py tools/*.py
"$PY" tools/_api_sigs.py hillclimb/*.py tools/*.py
