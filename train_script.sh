trap "exit" INT TERM ERR
trap "kill 0" EXIT

CUDA_VISIBLE_DEVICES=0,1,2,3 \
runjob --ngpus=4 --queue=local python -m cosypose.scripts.run_pose_training \
        --config bop-bm3-coarse \
        --version dsg \
        --resume bop-bm3-coarse-dsg-300643