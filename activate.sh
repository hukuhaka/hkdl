#!/bin/sh

if [ -n "${ZSH_VERSION-}" ]; then
    case ${ZSH_EVAL_CONTEXT-} in
        *:file) ;;
        *)
            echo "error: run: source ./activate.sh" >&2
            exit 2
            ;;
    esac
    _hkdl_activate_script=${(%):-%N}
    _hkdl_activate_shell=zsh
elif [ -n "${BASH_VERSION-}" ]; then
    if [ "${BASH_SOURCE[0]}" = "$0" ]; then
        echo "error: run: source ./activate.sh" >&2
        exit 2
    fi
    _hkdl_activate_script=${BASH_SOURCE[0]}
    _hkdl_activate_shell=bash
else
    echo "error: activate.sh supports Bash and Zsh; run: source ./activate.sh" >&2
    return 2 2>/dev/null
    exit 2
fi

_hkdl_activate_root=$(
    CDPATH= cd -- "$(dirname -- "$_hkdl_activate_script")" && pwd -P
)
if [ ! -r "$_hkdl_activate_root/.venv/bin/activate" ] ||
    [ ! -x "$_hkdl_activate_root/.venv/bin/hkdl" ]
then
    echo "error: HKDL environment is missing; run: ./setup.sh" >&2
    unset _hkdl_activate_root _hkdl_activate_script _hkdl_activate_shell
    return 1
fi

_hkdl_activate_completion=$(
    "$_hkdl_activate_root/.venv/bin/hkdl" completion "$_hkdl_activate_shell"
) || {
    echo "error: failed to generate HKDL completion" >&2
    unset _hkdl_activate_completion _hkdl_activate_root \
        _hkdl_activate_script _hkdl_activate_shell
    return 1
}

if [ "$_hkdl_activate_shell" = zsh ] && ! whence compdef >/dev/null 2>&1; then
    autoload -Uz compinit
    compinit -D
fi

. "$_hkdl_activate_root/.venv/bin/activate"
if ! eval "$_hkdl_activate_completion"; then
    deactivate
    echo "error: failed to register HKDL completion" >&2
    unset _hkdl_activate_completion _hkdl_activate_root \
        _hkdl_activate_script _hkdl_activate_shell
    return 1
fi

unset _hkdl_activate_completion _hkdl_activate_root \
    _hkdl_activate_script _hkdl_activate_shell
