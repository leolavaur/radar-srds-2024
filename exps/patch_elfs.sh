#!/usr/bin/env zsh

ENV=".venv"
LIB_PATH="lib/python3.10/site-packages"

declare -a targets=(
    "ray/core/src/ray/thirdparty/redis/src/redis-server"
    "ray/core/src/ray/gcs/gcs_server"
    "ray/core/src/ray/raylet/raylet"
)

declare -a binaries=(
    ruff
)

for target in $targets; do
    if [[ -f "$ENV/$LIB_PATH/$target" ]]; then
        patchelf --set-interpreter "$(cat $NIX_CC/nix-support/dynamic-linker)" "$ENV/$LIB_PATH/$target"
        echo "patched $target"
    else
        echo "not found $target"
    fi
done

for binary in $binaries; do
    if [[ -f "$ENV/bin/$binary" ]]; then
        patchelf --set-interpreter "$(cat $NIX_CC/nix-support/dynamic-linker)" "$ENV/bin/$binary"
        echo "patched $binary"
    else
        echo "not found $binary"
    fi
done