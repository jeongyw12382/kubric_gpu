export CUDA_VISIBLE_DEVICES=0
OUTNAME=${1:-0}

docker run --rm --interactive \
    --env KUBRIC_USE_GPU=1 \
    --gpus \"device=${CUDA_VISIBLE_DEVICES}\" \
    --user $(id -u):$(id -g) \
    --volume "$(pwd):/kubric" \
    kubricdockerhub/kubruntu \
    python3 -m challenges.gso.run \
    --scratch_dir=output/output_cache/${OUTNAME} \
    --job-dir=output/${OUTNAME}