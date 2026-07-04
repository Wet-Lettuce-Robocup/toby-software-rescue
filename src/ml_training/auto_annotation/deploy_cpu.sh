#!/usr/bin/env bash
# Sample commands to deploy nuclio functions on CPU

set -eu

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
FUNCTIONS_DIR=${1:-$SCRIPT_DIR}

export DOCKER_BUILDKIT=1

docker build -t cvat.openvino.base "$SCRIPT_DIR/openvino/base"

nuctl create project cvat --platform local

shopt -s globstar

for func_config in "$FUNCTIONS_DIR"/**/function.yaml
do
    func_root="$(dirname "$func_config")"
    func_rel_path="$(grealpath --relative-to="$SCRIPT_DIR" "$(dirname "$func_root")")"

    func_name="custom-model-yolov8"

    if [ -f "$func_root/Dockerfile" ]; then
        docker build -t "cvat.${func_rel_path//\//.}.base" "$func_root"
    fi

     # CRITICAL CLEANUP: Unlocks Nuclio by removing existing stuck/provisioning versions
    echo "Cleaning up any old deployments for $func_name..."
    nuctl delete function custom-model-yolov8 --namespace cvat --platform local

    docker rm -f $(docker ps -a -q --filter name="$func_name") 2>/dev/null || true
    docker rm -f $(docker ps -a -q --filter name="nuclio-nuclio-$func_name") 2>/dev/null || true

    echo "Deploying $func_rel_path function..."
    nuctl deploy --project-name cvat --path "$func_root" \
        --file "$func_config" --platform local \
        --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk \
        --env CVAT_FUNCTIONS_REDIS_PORT=6666 \
        --platform-config '{"attributes": {"network": "cvat_cvat"}}'
done

nuctl get function --platform local
