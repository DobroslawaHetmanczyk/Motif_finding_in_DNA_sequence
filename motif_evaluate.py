import argparse
import json
import sys
 
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate motif parameter estimates using total variation distance"
    )
    parser.add_argument(
        "--true",
        required=True,
        metavar="FILE",
        help="JSON file with the TRUE (ground-truth) parameters",
    )
    parser.add_argument(
        "--estimated",
        required=True,
        metavar="FILE",
        help="JSON file with the ESTIMATED parameters (output of motif_estimate.py)",
    )
    parser.add_argument(
        "--estimate-alpha",
        choices=["yes", "no"],
        default="no",
        help="Whether alpha was estimated (bonus part). Default: no",
    )
    args = parser.parse_args()
    return args.true, args.estimated, args.estimate_alpha
 

def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return 0.5 * np.sum(np.abs(p - q))
 
 

 
def validate_distribution(arr: np.ndarray, name: str, tol: float = 1e-6):
    s = arr.sum()
    if abs(s - 1.0) > tol:
        print(f"  WARNING: '{name}' sums to {s:.6f} (expected 1.0)", file=sys.stderr)
 
 
def load_and_validate(filepath: str, label: str) -> dict:
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: file not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: could not parse JSON in {filepath}: {e}", file=sys.stderr)
        sys.exit(1)
 
    for key in ("Theta", "Theta_bg"):
        if key not in data:
            print(f"ERROR: '{key}' missing in {filepath}", file=sys.stderr)
            sys.exit(1)
 
    theta    = np.array(data["Theta"],    dtype=float)   # (4, w)
    theta_bg = np.array(data["Theta_bg"], dtype=float)   # (4,)
 
    if theta.ndim != 2 or theta.shape[0] != 4:
        print(f"ERROR: Theta in {filepath} must be shape (4, w), got {theta.shape}",
              file=sys.stderr)
        sys.exit(1)
 
    if theta_bg.ndim != 1 or theta_bg.shape[0] != 4:
        print(f"ERROR: Theta_bg in {filepath} must have length 4, got {theta_bg.shape}",
              file=sys.stderr)
        sys.exit(1)
 
    w = theta.shape[1]
    for j in range(w):
        validate_distribution(theta[:, j], f"{label} Theta col {j+1}")
    validate_distribution(theta_bg, f"{label} Theta_bg")
 
    return data
 
 
 
def main():
    true_file, estim_file, estimate_alpha_flag = parse_arguments()
    estimate_alpha = (estimate_alpha_flag == "yes")
 
    true_data = load_and_validate(true_file,  label="true")
    estim_data = load_and_validate(estim_file, label="estimated")
 
    theta_orig = np.array(true_data["Theta"],    dtype=float)   # (4, w_true)
    theta_b_orig = np.array(true_data["Theta_bg"], dtype=float)
 
    theta_estim = np.array(estim_data["Theta"],    dtype=float)  # (4, w_estim)
    theta_b_estim = np.array(estim_data["Theta_bg"], dtype=float)
 
    w_true = theta_orig.shape[1]
    w_estim = theta_estim.shape[1]
 
    if w_true != w_estim:
        print(
            f"ERROR: motif length mismatch – true has w={w_true}, "
            f"estimated has w={w_estim}",
            file=sys.stderr,
        )
        sys.exit(1)
 
    w = w_true
 
    dtv_bg = total_variation(theta_b_orig, theta_b_estim)
 
    dtv_positions = np.array([
        total_variation(theta_orig[:, j], theta_estim[:, j])
        for j in range(w)
    ])

    if estimate_alpha:
        if "alpha" not in true_data:
            print(
                "ERROR: --estimate-alpha yes requires 'alpha' field in the true params file.",
                file=sys.stderr,
            )
            sys.exit(1)
        if "alpha" not in estim_data:
            print(
                "ERROR: 'alpha' field missing in estimated params file.",
                file=sys.stderr,
            )
            sys.exit(1)
 
        alpha_orig  = float(true_data["alpha"])
        alpha_estim = float(estim_data["alpha"])
        alpha_err   = abs(alpha_orig - alpha_estim)
 
        d_plus_tv = (alpha_err + dtv_bg + dtv_positions.sum()) / (w + 2)
 
        print("EVALUATION RESULTS  (bonus: alpha estimated)")
        print(f"Motif length w: {w}")
        print(f"True alpha: {alpha_orig:.6f}")
        print(f"Estimated alpha: {alpha_estim:.6f}")
        print(f"|alpha_orig - alpha_estim|: {alpha_err:.6f}")
        print(f"dtv(Theta_bg): {dtv_bg:.6f}")
        for j in range(w):
            print(f"dtv(Theta col {j+1:2d}): {dtv_positions[j]:.6f}")
        print(f"Mean positional dtv: {dtv_positions.mean():.6f}")
        print(f"d+tv (final score): {d_plus_tv:.6f}")

 
    else:
        dtv_avg = (dtv_bg + dtv_positions.sum()) / (w + 1)
 
        print("EVALUATION RESULTS  (obligatory: alpha known)")
        print(f"Motif length w: {w}")
        print(f"dtv(Theta_bg): {dtv_bg:.6f}")
        for j in range(w):
            print(f"dtv(Theta col {j+1:2d}): {dtv_positions[j]:.6f}")
        print(f"Mean positional dtv: {dtv_positions.mean():.6f}")
        print(f"dtv  (final score): {dtv_avg:.6f}")
 

 
if __name__ == "__main__":
    main()

# python motif_evaluate.py --true test_true_params.json --estimated estimated_params_test.json --estimate-alpha yes
