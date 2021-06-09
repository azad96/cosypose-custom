trap "exit" INT TERM ERR
trap "kill 0" EXIT

runjob --ngpus=4 --queue=local python -m cosypose.scripts.run_pose_training \
        --config bop-kuatless-refiner \
        --version 332k \
        --resume bop-kuatless-refiner-332k-838608
