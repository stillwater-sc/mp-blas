#!/usr/bin/env bash
# Sweep matmul_accumulator_bench over thread counts and core placements.
#
# MTL5's thread pool reads MTL5_NUM_THREADS once, when it is first used, so
# each thread count is a separate run. The pool doesn't pin its workers. On a
# hybrid Intel CPU the placement is chosen here with taskset, from the core
# lists Linux publishes in /sys/devices/cpu_core and /sys/devices/cpu_atom.
#
# Placements (-p, comma-separated):
#   all      no pinning; the scheduler decides (threads up to all hardware threads)
#   pcore    one hardware thread per P-core, no SMT siblings
#   pcore2   every P-core hardware thread, SMT siblings included
#   ecore    E-cores only
# A placement caps the thread counts at the number of CPUs it allows.
#
# Usage:
#   run_sweep.sh [-b BINARY] [-n N] [-r REPS] [-y TYPE] [-c CONFIGS] [-t "1 2 4 8"] [-p all,pcore,ecore] [-o out.csv]
#
# -c passes --configs through, e.g. -c default,plain,fma -n 1024 to time the
# native configurations at a size where they take long enough to scale.
#
# Prints the CSV of every run, then a speedup table per placement, type and
# configuration, relative to that placement's first thread count.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
bin=""
n=128
reps=3
type=all
configs=default,plain,fma,quire
threads=""
placements="all,pcore,ecore"
out=""

while getopts "b:n:r:y:c:t:p:o:h" opt; do
	case $opt in
		b) bin=$OPTARG ;;
		n) n=$OPTARG ;;
		r) reps=$OPTARG ;;
		y) type=$OPTARG ;;
		c) configs=$OPTARG ;;
		t) threads=$OPTARG ;;
		p) placements=$OPTARG ;;
		o) out=$OPTARG ;;
		*) sed -n '2,23p' "$0"; exit 0 ;;
	esac
done

if [[ -z $bin ]]; then
	for c in ./matmul_accumulator_bench \
	         ./applications/level3/matmul_accumulator_bench/matmul_accumulator_bench \
	         "$here/../../../build/applications/level3/matmul_accumulator_bench/matmul_accumulator_bench"; do
		if [[ -x $c ]]; then bin=$c; break; fi
	done
fi
[[ -x $bin ]] || { echo "matmul_accumulator_bench not found; pass -b" >&2; exit 1; }

# Expand a cpulist such as "0-3,8,10-11" into one CPU per line.
expand() {
	tr ',' '\n' <<<"$1" | while IFS=- read -r lo hi; do seq "$lo" "${hi:-$lo}"; done
}

# The CPUs a placement allows, comma-separated, or empty for no pinning.
cpus_for() {
	local core=/sys/devices/cpu_core/cpus atom=/sys/devices/cpu_atom/cpus
	case $1 in
		all) echo "" ;;
		pcore2|pcore|ecore)
			if [[ ! -r $core || ! -r $atom ]]; then
				echo "placement $1 needs a hybrid CPU (/sys/devices/cpu_core, cpu_atom)" >&2
				return 1
			fi ;;&
		pcore2) expand "$(cat $core)" | paste -sd, ;;
		pcore)
			# the first SMT sibling of each P-core
			expand "$(cat $core)" | while read -r c; do
				expand "$(cat /sys/devices/system/cpu/cpu$c/topology/thread_siblings_list)" | head -1
			done | sort -n -u | paste -sd, ;;
		ecore) expand "$(cat $atom)" | paste -sd, ;;
		*) echo "unknown placement $1" >&2; return 1 ;;
	esac
}

csv=$(mktemp)
trap 'rm -f "$csv"' EXIT
header_done=0

IFS=, read -ra plist <<<"$placements"
for p in "${plist[@]}"; do
	cpus=$(cpus_for "$p")
	if [[ -n $cpus ]]; then
		ncpu=$(tr ',' '\n' <<<"$cpus" | wc -l)
	else
		ncpu=$(nproc --all)
	fi
	tlist=${threads:-"1 2 4 6 8 12 16 20"}
	ran=0
	for t in $tlist; do
		(( t <= ncpu )) || continue
		ran=1
		echo "[$p] $t thread(s)${cpus:+ on CPUs $cpus}" >&2
		if [[ -n $cpus ]]; then
			res=$(MTL5_NUM_THREADS=$t taskset -c "$cpus" "$bin" --csv --n "$n" --reps "$reps" --type "$type" --configs "$configs")
		else
			res=$(MTL5_NUM_THREADS=$t "$bin" --csv --n "$n" --reps "$reps" --type "$type" --configs "$configs")
		fi
		if (( !header_done )); then
			echo "placement,$(head -1 <<<"$res")" >>"$csv"
			header_done=1
		fi
		tail -n +2 <<<"$res" | sed "s/^/$p,/" >>"$csv"
	done
	if (( !ran )); then echo "[$p] no requested thread count fits $ncpu CPU(s)" >&2; fi
done

cat "$csv"
[[ -n $out ]] && cp "$csv" "$out"

# Speedup over the first thread count of each placement/type/config series.
echo
awk -F, 'NR > 1 && $6 != "skipped" {
	key = $1 "," $2 "," $3
	if (!(key in base)) { base[key] = $6; order[++nk] = key }
	line[key] = line[key] sprintf("  %2d:%7.2fx", $4, base[key] / $6)
	rate[key] = rate[key] sprintf("  %2d:%9.1f", $4, $8)
}
END {
	print "speedup over the first thread count (min time), per placement/type/config"
	for (i = 1; i <= nk; ++i) printf "  %-28s%s\n", order[i], line[order[i]]
	print ""
	print "throughput, million multiply-adds per second"
	for (i = 1; i <= nk; ++i) printf "  %-28s%s\n", order[i], rate[order[i]]
}' "$csv"
