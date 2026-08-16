#!/usr/bin/env bash
# PreToolUse (Bash) hook: block Python package installs on the host. Everything
# Python in this repo runs in the container built from api/Dockerfile, so a
# host-side pip/uv/poetry/conda install builds a second environment the API
# never sees — the install looks like it worked and the container behaves as if
# it never happened. Reads the hook JSON on stdin and emits a "deny" permission
# decision when a command segment installs packages outside a container runtime.
# Exits 0 with no output (= allow) for everything else.
set -euo pipefail

input=$(cat)
cmd=$(printf '%s' "$input" | jq -r '.tool_input.command // ""')
[[ -n $cmd ]] || exit 0

# Split on shell separators, but only outside quotes, so a containerized payload
# like `podman exec api sh -c "cd /app; pip install x"` stays attached to the
# podman that makes it legitimate.
split_segments() {
  local s=$1 i ch q='' seg=''
  for ((i = 0; i < ${#s}; i++)); do
    ch=${s:i:1}
    if [[ -n $q ]]; then
      seg+=$ch
      [[ $ch == "$q" ]] && q=''
      continue
    fi
    case $ch in
      \' | \") q=$ch; seg+=$ch ;;
      \\) i=$((i + 1)); seg+=${s:i:1} ;;
      ';' | '&' | '|' | '(' | ')' | '`' | $'\n') printf '%s\n' "$seg"; seg='' ;;
      *) seg+=$ch ;;
    esac
  done
  printf '%s\n' "$seg"
}

# Prints "<tool> <subcommand>" if the token list is a host Python install,
# nothing otherwise. First arg is the recursion depth for `sh -c` payloads.
offending_install() {
  local depth=$1; shift
  local -a toks=("$@")
  ((depth < 3)) || return 0

  # Step over env assignments and wrappers that only prefix the real command,
  # so `sudo -H pip install` and `env FOO=1 pip install` still resolve to pip.
  local i=0
  while ((i < ${#toks[@]})); do
    # Tested before the leading path is stripped: VIRTUAL_ENV=/x/y is an
    # assignment, and stripping to the last slash would hide the "=".
    case ${toks[i]} in
      *=* | -*) i=$((i + 1)); continue ;;
    esac
    case ${toks[i]##*/} in
      sudo | env | command | nohup | stdbuf | caffeinate | time | nice) i=$((i + 1)); continue ;;
    esac
    break
  done
  ((i < ${#toks[@]})) || return 0

  local head=${toks[i]##*/}

  # A container runtime carries the rest of the segment into the container,
  # which is exactly where these installs belong.
  case $head in
    podman | podman-remote | podman-compose | docker | docker-compose | nerdctl) return 0 ;;
  esac

  # `sh -c "pip install x"` hides the install one level down; check the payload.
  case $head in
    sh | bash | zsh | dash | ksh)
      local j=$((i + 1)) payload out
      local -a sub_toks
      while ((j < ${#toks[@]})); do
        if [[ ${toks[j]} == -c ]]; then
          payload=${toks[*]:j+1}
          payload=${payload#[\"\']}; payload=${payload%[\"\']}
          while IFS= read -r seg; do
            read -ra sub_toks <<<"$seg" || true
            ((${#sub_toks[@]})) || continue
            out=$(offending_install $((depth + 1)) "${sub_toks[@]}")
            if [[ -n $out ]]; then printf '%s' "$out"; return 0; fi
          done < <(split_segments "$payload")
          return 0
        fi
        j=$((j + 1))
      done
      return 0
      ;;
  esac

  # Positional arguments only — flags between the tool and its subcommand
  # (`pip --quiet install foo`) must not hide what the subcommand is.
  local -a argv=()
  local k
  for ((k = i + 1; k < ${#toks[@]}; k++)); do
    [[ ${toks[k]} == -* ]] || argv+=("${toks[k]}")
  done
  local a0=${argv[0]:-} a1=${argv[1]:-}

  case $head in
    pip | pip[0-9] | pip[0-9].[0-9] | pip[0-9].[0-9][0-9])
      case $a0 in install | uninstall) printf 'pip %s' "$a0" ;; esac ;;
    python | python[0-9] | python[0-9].[0-9] | python[0-9].[0-9][0-9])
      # Only the -m module form installs anything: python -m pip install ...
      [[ " ${toks[*]} " == *" -m "* ]] || return 0
      case $a0 in
        pip) case $a1 in install | uninstall) printf 'python -m pip %s' "$a1" ;; esac ;;
        ensurepip) printf 'python -m ensurepip' ;;
      esac ;;
    uv)
      case $a0 in
        add | remove | sync) printf 'uv %s' "$a0" ;;
        pip) case $a1 in install | uninstall | sync) printf 'uv pip %s' "$a1" ;; esac ;;
        tool) case $a1 in install | upgrade | uninstall) printf 'uv tool %s' "$a1" ;; esac ;;
      esac ;;
    poetry)
      case $a0 in add | install | update | remove | sync) printf 'poetry %s' "$a0" ;; esac ;;
    pipenv)
      case $a0 in install | uninstall | sync | update) printf 'pipenv %s' "$a0" ;; esac ;;
    pipx)
      case $a0 in install | inject | upgrade | upgrade-all | reinstall) printf 'pipx %s' "$a0" ;; esac ;;
    conda | mamba | micromamba)
      case $a0 in
        install | create | update) printf '%s %s' "$head" "$a0" ;;
        env) case $a1 in create | update) printf '%s env %s' "$head" "$a1" ;; esac ;;
      esac ;;
    easy_install) printf 'easy_install' ;;
  esac
  return 0
}

declare -a segment_toks
found=''
while IFS= read -r segment; do
  read -ra segment_toks <<<"$segment" || true
  ((${#segment_toks[@]})) || continue
  found=$(offending_install 0 "${segment_toks[@]}")
  if [[ -n $found ]]; then break; fi
done < <(split_segments "$cmd")

[[ -n $found ]] || exit 0

jq -n --arg tool "$found" '{
  hookSpecificOutput: {
    hookEventName: "PreToolUse",
    permissionDecision: "deny",
    permissionDecisionReason: (
      "Blocked by policy: Python packages for this project are installed inside the podman container, never on the host. Detected `" + $tool +
      "` running outside a container runtime. To add or change a dependency, edit api/requirements.txt and rebuild the image (`podman compose up --build`) — a running container serves baked-in code, so a rebuild is what makes the change real. To try a package without committing to it, run it in the container instead: `podman compose run --rm api pip install <pkg>`. If this genuinely has to happen on the host, stop and ask the user to run it."
    )
  }
}'
