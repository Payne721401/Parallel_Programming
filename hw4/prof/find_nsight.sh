#!/bin/bash
# Sourced by the diagnostic launcher. No downloads, sudo, or MPS changes.
find_nsight_tools() {
  local init candidate
  if ! type module >/dev/null 2>&1; then
    for init in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh /usr/share/Modules/init/bash; do
      if [[ -r "$init" ]]; then
        set +u
        source "$init"
        set -u
        type module >/dev/null 2>&1 && break
      fi
    done
  fi
  if ! command -v nsys >/dev/null || ! command -v ncu >/dev/null; then
    if type module >/dev/null 2>&1; then
      module -t avail cuda nsight > "$OUT/modules.txt" 2>&1 || true
      cat "$OUT/modules.txt"
      # cuda/12.4 is the module used by this repository's Nano5 HW3 tools.
      # Only try modules actually advertised by this installation.
      for candidate in cuda/12.4 cuda/12.6; do
        if grep -Fq "$candidate" "$OUT/modules.txt"; then
          echo "Trying module: $candidate"
          module load "$candidate" || true
          if command -v nsys >/dev/null && command -v ncu >/dev/null; then break; fi
        fi
      done
    fi
  fi
  NSYS_BIN=${NSYS_BIN:-$(command -v nsys || true)}
  NCU_BIN=${NCU_BIN:-$(command -v ncu || true)}
  if [[ -z "$NSYS_BIN" || -z "$NCU_BIN" ]]; then
    local roots=()
    for candidate in /opt /usr/local /apps /software; do
      [[ -d "$candidate" ]] && roots+=("$candidate")
    done
    if (( ${#roots[@]} )); then
      timeout 10s find -L "${roots[@]}" -maxdepth 8 -type f \( -name nsys -o -name ncu \) -executable \
        > "$OUT/tool-paths.txt" 2> "$OUT/tool-search-errors.txt" || true
      while IFS= read -r candidate; do
        [[ -z "$NSYS_BIN" && ${candidate##*/} == nsys ]] && NSYS_BIN=$candidate
        [[ -z "$NCU_BIN" && ${candidate##*/} == ncu ]] && NCU_BIN=$candidate
      done < <(sort -Vr "$OUT/tool-paths.txt")
    fi
  fi
  echo "NSYS_BIN=${NSYS_BIN:-NOT_FOUND}"
  echo "NCU_BIN=${NCU_BIN:-NOT_FOUND}"
  [[ -n "$NSYS_BIN" ]] && "$NSYS_BIN" --version
  [[ -n "$NCU_BIN" ]] && "$NCU_BIN" --version
  [[ -x "$NSYS_BIN" && -x "$NCU_BIN" ]]
}
