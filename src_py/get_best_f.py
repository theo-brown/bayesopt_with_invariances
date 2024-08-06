import argparse

import h5py

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()

max_f = 0
with h5py.File(args.filename, "r") as h5:
    for kernel in h5.attrs["eval_kernels"]:
        for i in range(h5.attrs["repeats"]):
            this_max_f = max(h5[f"{kernel}/{i}/reported_f"][:])
            if this_max_f > max_f:
                max_f = this_max_f

print(max_f)