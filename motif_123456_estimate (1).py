import argparse
import json

import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(description="Motif parameter estimator")
    parser.add_argument(
        "--input",
        default="motif_data_known_alpha.json",
        required=False,
        help="File with input data (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="estimated_params1.json",
        required=False,
        help="File where the estimated parameters will be saved (default: %(default)s)",
    )
    parser.add_argument(
        "--estimate-alpha",
        choices=["yes", "no"],
        default="no",
        required=False,
        help="Should alpha be estimated? (default: %(default)s)",
    )
    args = parser.parse_args()
    return args.input, args.output, args.estimate_alpha


input_file, output_file, estimate_alpha = parse_arguments()

with open(input_file, "r") as inputfile:
    data = json.load(inputfile)

X = np.asarray(data["X"])
k, w = X.shape

if estimate_alpha == "yes":
    alpha = 0.5  # TO DO: estimate alpha from X
else:
    alpha = data["alpha"]

# TO DO: MAIN PART: Estimate Theta and Theta_bg using EM and save to output_file.
# Theta is a matrix of size 4 x w.
# Theta_bg is a vector of length 4.
# The example below is only a placeholder.

Theta = np.zeros((4, w))
Theta[:3, :] = np.random.random((3, w)) / 4
Theta[3, :] = 1 - np.sum(Theta[:3, :], axis=0)

Theta_bg = np.zeros(4)
Theta_bg[:3] = np.random.random(3) / 4
Theta_bg[3] = 1 - np.sum(Theta_bg[:3])

estimated_params1 = {
    "alpha": alpha,
    "Theta": Theta.tolist(),
    "Theta_bg": Theta_bg.tolist(),
}

with open(output_file, "w") as outfile:
    json.dump(estimated_params1, outfile)
