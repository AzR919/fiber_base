#!/bin/bash
# Sync between the local repo and the DRAC clusters (nibi, fir), with the local repo as the hub. Replaces the
# sync_fiber* scripts (no watcher, no Mutagen): every transfer is explicit. Run from WSL:
#   wsl bash misc/cluster.sh <command> [args]
#
#   check                      which ControlMaster sockets are alive (never opens one: 2FA is the user's)
#   push-code <cluster>        local code -> cluster (no delete; results, logs, plots, wandb, ignore, .git excluded)
#   pull <cluster>             cluster results/ logs/ plots/ -> local (no delete; local is the union of clusters)
#   push-results <cluster> <results dir name>...   local results/<dir> -> cluster (checkpoints a job there needs)
#   ls-data <cluster>          data file listing (path, size, date) -> ignore/manifests/<cluster>_data.tsv
#   diff-data <from> <to>      data files present on <from> but missing or different in size on <to>
#   copy-data <from> <to> <path under data/>...    data via the local buffer, one path at a time, buffer emptied
#                              after each (refuses if the path is bigger than the free space minus a 20 GB margin)
#   buffer                     local buffer sizes and free disk space
set -euo pipefail

L=/mnt/c/Users/azaan/Documents/SFU/Lab/code_base/fiber_base
R=/home/azr/projects/def-maxwl/azr
CODE=$R/code/fiber_base
SSH="ssh -o BatchMode=yes"
BUF=$L/ignore/cluster_data
MAN=$L/ignore/manifests
MARGIN_GB=20

alive() { ssh -o BatchMode=yes -O check "$1" >/dev/null 2>&1; }
need() { alive "$1" || { echo "socket to $1 is down: open it with  wsl ssh -fN $1" >&2; exit 2; }; }
free_gb() { df -BG --output=avail /mnt/c | tail -1 | tr -dc 0-9; }

cmd="${1:-}"; shift || true
case "$cmd" in
  check)
    for c in nibi fir; do if alive $c; then echo "$c: up"; else echo "$c: down"; fi; done ;;
  push-code)
    c=$1; need $c
    cd $L
    rsync -rlic --info=name1 -e "$SSH" \
      --exclude=.git/ --exclude=results/ --exclude=logs/ --exclude=plots/ --exclude=wandb/ --exclude=ignore/ \
      --exclude=__pycache__/ --exclude=.claude/ --exclude=.vscode/ --exclude=reports/latex/build/ \
      ./ $c:$CODE/ ;;
  pull)
    c=$1; need $c
    for d in results logs plots; do
      mkdir -p $L/$d
      rsync -rt --info=stats0 -e "$SSH" $c:$CODE/$d/ $L/$d/
    done
    du -sh $L/results $L/logs $L/plots ;;
  push-results)
    c=$1; shift; need $c
    for d in "$@"; do
      d=$(basename "$d")
      [ -d "$L/results/$d" ] || { echo "not in the local buffer: results/$d" >&2; exit 1; }
      rsync -rt --info=stats0 -e "$SSH" "$L/results/$d" $c:$CODE/results/
      echo "pushed results/$d -> $c"
    done ;;
  ls-data)
    c=$1; need $c; mkdir -p $MAN
    $SSH $c "cd $R/data && find DATA_FIBER misc/annotations misc/grch38_ccres.bed misc/hg38.fa misc/hg38.fa.fai \
      -type f -printf '%P\t%s\t%TY-%Tm-%Td\n' 2>/dev/null | sort" > $MAN/${c}_data.tsv
    echo "$(wc -l < $MAN/${c}_data.tsv) files -> $MAN/${c}_data.tsv" ;;
  diff-data)
    a=$1; b=$2
    for c in $a $b; do [ -s $MAN/${c}_data.tsv ] || { echo "run ls-data $c first" >&2; exit 1; }; done
    # rows of <from> whose (path, size) is not in <to>; prefix path with data dir relative to misc/ for misc files
    awk -F'\t' 'NR==FNR {s[$1]=$2; next} !($1 in s) {print "missing\t"$2"\t"$1; next} s[$1]!=$2 {print "size\t"$2"\t"$1}' \
      $MAN/${b}_data.tsv $MAN/${a}_data.tsv | sort -k3
    ;;
  copy-data)
    a=$1; b=$2; shift 2; need $a; need $b
    for p in "$@"; do
      mb=$($SSH $a "du -s -BM $R/data/$p | cut -f1" | tr -dc 0-9)
      f=$(free_gb)
      if [ $((mb / 1024 + MARGIN_GB)) -gt "$f" ]; then
        echo "skip $p: $mb MB, only $f GB free locally (margin $MARGIN_GB GB); split it into smaller paths" >&2; continue
      fi
      mkdir -p "$BUF/$(dirname "$p")"
      echo "== $p ($mb MB): $a -> local -> $b"
      rsync -rt --info=stats0 -e "$SSH" "$a:$R/data/$p" "$BUF/$(dirname "$p")/"
      $SSH $b "mkdir -p $R/data/$(dirname "$p")"
      rsync -rt --info=stats0 -e "$SSH" "$BUF/$p" "$b:$R/data/$(dirname "$p")/"
      # keep small things (regions, annotations) in the buffer; drop anything over 1 GB once it has arrived
      if [ "$mb" -ge 1024 ]; then rm -rf "${BUF:?}/$p"; echo "   removed from the local buffer"; fi
    done ;;
  buffer)
    du -sh $L/results $L/logs $L/plots $BUF 2>/dev/null || true
    echo "free on C: $(free_gb) GB" ;;
  *)
    sed -n '2,18p' "$0"; exit 1 ;;
esac
