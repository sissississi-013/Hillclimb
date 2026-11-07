#!/usr/bin/env bash
# Free, local, end-to-end check of the fine-tune data on the station: add_eef -> one parent -> Dream's checks.
# NEVER uploads, never calls finetune, needs no API key. Rehearsed offline in docs/research/09-pipeline-rehearsal.md.
#
#   tools/rehearse_pipeline.sh                                 # rehearse FT#1 demos into a temp parent, then delete it
#   tools/rehearse_pipeline.sh --self                          # + harvested ~/hackathon_data/self/hc-*-self (FT#2)
#   tools/rehearse_pipeline.sh --parent ~/hackathon_data/hc-ft1          # build the REAL FT#1 parent and keep it
#   tools/rehearse_pipeline.sh --parent ~/hackathon_data/hc-ft2 --self   # build the REAL FT#2 parent and keep it
#   tools/rehearse_pipeline.sh --parent DIR SRC [SRC ...]      # explicit collected folders instead of the defaults
#
# Default sources: $DATA/hc-{unplug,open,close,plug,full} that exist (DEMO-PROTOCOL folders). Each SRC becomes
# <parent>/<basename SRC>-eef via the kit's add_eef convert() (= `add_eef.py SRC --arm metal --out ... --replace`;
# videos hard-linked, so the parent costs ~no disk and rebuilding it never touches the collected folders). An existing --parent is rebuilt from scratch,
# but only if it holds nothing except *-eef datasets. Env: DATA (~/hackathon_data), DREAM_PY (~/dream-env/bin/python),
# HC_KIT (kit checkout; default the first of ~/physical-ai-hacks, ~/physical-ai-hack-2026).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${DATA:-$HOME/hackathon_data}"
DREAM_PY="${DREAM_PY:-$HOME/dream-env/bin/python}"
PARENT="" ; WITH_SELF=0 ; SRCS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --parent) PARENT="$2"; shift 2 ;;
    --self) WITH_SELF=1; shift ;;
    --data) DATA="$2"; shift 2 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) SRCS+=("$1"); shift ;;
  esac
done
DATA="$(cd "$DATA" && pwd)"

if [ -z "${HC_KIT:-}" ]; then
  for k in "$HOME/physical-ai-hacks" "$HOME/physical-ai-hack-2026" "$HOME/hc-vendor/physical-ai-hack-2026"; do
    if [ -f "$k/tools/add_eef.py" ]; then HC_KIT="$k"; break; fi
  done
fi
[ -n "${HC_KIT:-}" ] && [ -f "$HC_KIT/tools/add_eef.py" ] || { echo "kit not found: set HC_KIT=<kit checkout>" >&2; exit 2; }
export HC_KIT
[ -x "$DREAM_PY" ] || { echo "no Dream python at $DREAM_PY: set DREAM_PY" >&2; exit 2; }

if [ ${#SRCS[@]} -eq 0 ]; then
  for t in unplug open close plug full; do
    if [ -f "$DATA/hc-$t/meta/info.json" ]; then SRCS+=("$DATA/hc-$t"); else echo "skip hc-$t: not recorded yet"; fi
  done
  if [ "$WITH_SELF" = 1 ]; then
    for s in "$DATA"/self/hc-*-self; do [ -f "$s/meta/info.json" ] && SRCS+=("$s"); done
  fi
fi
[ ${#SRCS[@]} -gt 0 ] || { echo "no collected folders found under $DATA" >&2; exit 1; }

KEEP=1
if [ -z "$PARENT" ]; then
  KEEP=0; PARENT="$DATA/.rehearse-$$"   # hidden: never picked up as a session; same disk, so videos hard-link
  trap 'rm -rf "$PARENT"' EXIT
fi
mkdir -p "$PARENT"; PARENT="$(cd "$PARENT" && pwd)"
[ "$PARENT" != "$DATA" ] || { echo "--parent must be a NEW folder, not $DATA itself" >&2; exit 2; }
for child in "$PARENT"/* "$PARENT"/.[!.]*; do
  [ -e "$child" ] || [ -L "$child" ] || continue
  case "$child" in
    *-eef) [ -L "$child" ] || [ -f "$child/meta/info.json" ] || { echo "refusing: $child is not a dataset" >&2; exit 2; } ;;
    *) echo "refusing to rebuild $PARENT: it holds $(basename "$child") (only *-eef datasets allowed)" >&2; exit 2 ;;
  esac
done
rm -rf "$PARENT"/*-eef      # derived copies only: the videos are hard links, the collected folders stay as they are

seen=" "
for src in "${SRCS[@]}"; do
  src="$(cd "$src" && pwd)"; name="$(basename "$src")"
  case "$name" in *-eef) echo "refusing $src: pass the collected folder, not its -eef output" >&2; exit 2 ;; esac
  case "$seen" in *" $name "*) echo "two sources named $name: the session names would collide" >&2; exit 2 ;; esac
  seen="$seen$name "
  echo "== add_eef $src -> $PARENT/$name-eef"
  # = "$DREAM_PY" "$HC_KIT/tools/add_eef.py" "$src" --arm metal --out "$PARENT/$name-eef" --replace, in-process via
  # ft.py so a pyarrow exit deadlock can't hang the script (09-pipeline-rehearsal.md gotchas)
  "$DREAM_PY" "$REPO/tools/ft.py" eef "$src" --out "$PARENT/$name-eef" --replace
done

echo "== ft.py check $PARENT"
"$DREAM_PY" "$REPO/tools/ft.py" check "$PARENT"
echo
if [ "$KEEP" = 1 ]; then
  echo "PASS: $PARENT is ready. Nothing was uploaded. To queue (PAID, Sissi's OK first):"
  echo "  $DREAM_PY $REPO/tools/ft.py launch $PARENT --name <name> --suffix <suffix> --steps 1000"
else
  echo "PASS (rehearsal): the temp parent is deleted. Build the real one with --parent $DATA/hc-ft1"
fi
