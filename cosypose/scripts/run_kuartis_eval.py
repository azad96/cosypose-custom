import subprocess
import shutil
from tqdm import tqdm
import torch
import os
import argparse
import sys
from time import time
from pathlib import Path
from cosypose.config import PROJECT_DIR, RESULTS_DIR


TOOLKIT_DIR = Path(PROJECT_DIR / 'deps' / 'bop_toolkit_challenge')
EVAL_SCRIPT_PATH = TOOLKIT_DIR / 'scripts/eval_kuartis.py'

sys.path.append(TOOLKIT_DIR.as_posix())
from bop_toolkit_lib import inout  # noqa
# from bop_toolkit_lib.config import results_path as BOP_RESULTS_PATH  # noqa


def main():
    parser = argparse.ArgumentParser('Bop evaluation')
    # parser.add_argument('--result_name', default='kuatless-1080-810-n_views=1-v6-epoch190', type=str)
    parser.add_argument('--result_name', default='kuatless-720-540-n_views=1-v4', type=str)
    # parser.add_argument('--targets_filename', default='kuatless_test_1080_810.json', type=str)
    parser.add_argument('--targets_filename', default='kuatless_test_720_540.json', type=str)
    parser.add_argument('--convert_only', action='store_true')
    args = parser.parse_args()
    run_evaluation(args)


def run_evaluation(args):
    result_path = os.path.join(RESULTS_DIR, args.result_name, 'results.pth.tar')
    # coarse_csv_name = 'coarsev4-check_kuatless-test.csv'
    # coarse_csv_path = os.path.join(RESULTS_DIR, args.result_name, coarse_csv_name)
    # coarse_method = 'pix2pose_detections/coarse/iteration=1'
    # convert_results(result_path, coarse_csv_path, method=coarse_method)

    # refiner_csv_name = 'refiner-v6-epoch190__kuatless-test_pbr_1080_810.csv'
    refiner_csv_name = 'refiner-v4__kuatless-test_pbr_720_540.csv'
    refiner_csv_path = os.path.join(RESULTS_DIR, args.result_name, refiner_csv_name)
    refiner_method = 'pix2pose_detections/refiner/iteration=1'
    convert_results(result_path, refiner_csv_path, method=refiner_method)

    # csv_paths = ','.join([coarse_csv_path, refiner_csv_path])
    csv_paths = refiner_csv_path

    if not args.convert_only:
        run_bop_evaluation(csv_paths, args.targets_filename)


def convert_results(results_path, out_csv_path, method):
    predictions = torch.load(results_path)['predictions']
    predictions = predictions[method]
    print("Predictions from:", results_path)
    print("Method:", method)
    print("Number of predictions: ", len(predictions))

    preds = []
    for n in tqdm(range(len(predictions))):
        TCO_n = predictions.poses[n]
        t = TCO_n[:3, -1] * 1e3  # m -> mm conversion
        R = TCO_n[:3, :3]
        row = predictions.infos.iloc[n]
        obj_id = int(row.label.split('_')[-1])
        score = row.score
        time = -1
        pred = dict(scene_id=row.scene_id,
                    im_id=row.view_id,
                    obj_id=obj_id,
                    score=score,
                    t=t, R=R, time=time)
        preds.append(pred)
    print("Wrote:", out_csv_path)
    inout.save_bop_results(out_csv_path, preds)
    return out_csv_path


def run_bop_evaluation(filenames, targets_filename):
    myenv = os.environ.copy()
    myenv['PYTHONPATH'] = TOOLKIT_DIR.as_posix()
    myenv['COSYPOSE_DIR'] = PROJECT_DIR.as_posix()
    script_path = EVAL_SCRIPT_PATH
    s = time()
    subprocess.call(['python', script_path.as_posix(),
                     '--renderer_type', 'python',
                     '--targets_filename', targets_filename,
                     '--result_filenames', filenames],
                    env=myenv, cwd=TOOLKIT_DIR.as_posix())
    f = time()
    print('Evaluation time: {} minutes'.format((f-s)/60))


if __name__ == '__main__':
    main()
