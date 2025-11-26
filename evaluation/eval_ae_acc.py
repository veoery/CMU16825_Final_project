# Ref: https://github.com/DavidXu-JJ/DeepCAD/blob/tags/CAD-MLLM/evaluation/evaluate_ae_acc.py
import h5py
from tqdm import tqdm
import os
import argparse
import numpy as np
import sys
sys.path.append("..")

# Try importing cadlib, fall back to mock definitions if unavailable
try:
    from cadlib.macro import *
    CADLIB_AVAILABLE = True
except ImportError:
    print("Warning: cadlib not available, using mock CAD command definitions")
    print("For full functionality, install: https://github.com/DavidXu-JJ/DeepCAD (branch: tags/CAD-MLLM)")
    CADLIB_AVAILABLE = False

    # Mock CAD command definitions for testing with mock data
    SOL_IDX = 0
    EOS_IDX = 1
    LINE_IDX = 2
    ARC_IDX = 3
    CIRCLE_IDX = 4
    EXT_IDX = 5

    ALL_COMMANDS = ['SOL', 'EOS', 'LINE', 'ARC', 'CIRCLE', 'EXT']

    # Command argument mask: which parameters are valid for each command
    # Shape: (num_commands, num_params)
    CMD_ARGS_MASK = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # SOL - no params
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # EOS - no params
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # LINE - x, y
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # ARC - x, y, r, angle
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # CIRCLE - x, y, r
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # EXT - distance, boolean
    ])

TOLERANCE = 3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default=None, required=True)
    args = parser.parse_args()

    result_dir = args.src
    filenames = sorted(os.listdir(result_dir))

    # overall accuracy
    avg_cmd_acc = [] # ACC_cmd
    avg_param_acc = [] # ACC_param

    # accuracy w.r.t. each command type
    each_cmd_cnt = np.zeros((len(ALL_COMMANDS),))
    each_cmd_acc = np.zeros((len(ALL_COMMANDS),))

    # accuracy w.r.t each parameter
    args_mask = CMD_ARGS_MASK.astype(np.float64)
    N_ARGS = args_mask.shape[1]
    each_param_cnt = np.zeros([*args_mask.shape])
    each_param_acc = np.zeros([*args_mask.shape])

    for name in tqdm(filenames):
        path = os.path.join(result_dir, name)
        with h5py.File(path, "r") as fp:
            out_vec = fp["out_vec"][:].astype(np.int64)
            gt_vec = fp["gt_vec"][:].astype(np.int64)

        out_cmd = out_vec[:, 0]
        gt_cmd = gt_vec[:, 0]

        out_param = out_vec[:, 1:]
        gt_param = gt_vec[:, 1:]

        cmd_acc = (out_cmd == gt_cmd).astype(np.int64)
        param_acc = []
        for j in range(len(gt_cmd)):
            cmd = gt_cmd[j]
            each_cmd_cnt[cmd] += 1
            each_cmd_acc[cmd] += cmd_acc[j]
            if cmd in [SOL_IDX, EOS_IDX]:
                continue

            if out_cmd[j] == gt_cmd[j]: # NOTE: only account param acc for correct cmd
                tole_acc = (np.abs(out_param[j] - gt_param[j]) < TOLERANCE).astype(np.int64)
                # filter param that do not need tolerance (i.e. requires strictly equal)
                if cmd == EXT_IDX:
                    tole_acc[-2:] = (out_param[j] == gt_param[j]).astype(np.int64)[-2:]
                elif cmd == ARC_IDX:
                    tole_acc[3] = (out_param[j] == gt_param[j]).astype(np.int64)[3]

                valid_param_acc = tole_acc[args_mask[cmd].astype(bool)].tolist()
                param_acc.extend(valid_param_acc)

                each_param_cnt[cmd, np.arange(N_ARGS)] += 1
                each_param_acc[cmd, np.arange(N_ARGS)] += tole_acc

        param_acc = np.mean(param_acc)
        avg_param_acc.append(param_acc)
        cmd_acc = np.mean(cmd_acc)
        avg_cmd_acc.append(cmd_acc)

    save_path = result_dir + "_acc_stat.txt"
    fp = open(save_path, "w")
    # overall accuracy (averaged over all data)
    avg_cmd_acc = np.mean(avg_cmd_acc)
    print("avg command acc (ACC_cmd):", avg_cmd_acc, file=fp)
    avg_param_acc = np.mean(avg_param_acc)
    print("avg param acc (ACC_param):", avg_param_acc, file=fp)

    # acc of each command type
    each_cmd_acc = each_cmd_acc / (each_cmd_cnt + 1e-6)
    print("each command count:", each_cmd_cnt, file=fp)
    print("each command acc:", each_cmd_acc, file=fp)

    # acc of each parameter type
    each_param_acc = each_param_acc * args_mask
    each_param_cnt = each_param_cnt * args_mask
    each_param_acc = each_param_acc / (each_param_cnt + 1e-6)
    for i in range(each_param_acc.shape[0]):
        print(ALL_COMMANDS[i] + " param acc:", each_param_acc[i][args_mask[i].astype(bool)], file=fp)
    fp.close()

    with open(save_path, "r") as fp:
        res = fp.readlines()
        for l in res:
            print(l, end='')