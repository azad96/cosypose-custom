trap "exit" INT TERM ERR
trap "kill 0" EXIT

CUDA_VISIBLE_DEVICES=0 python cosypose/scripts/run_singleview_seg.py