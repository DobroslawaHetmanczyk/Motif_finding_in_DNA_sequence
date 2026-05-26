import argparse
import json
import numpy as np
 
 
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate synthetic motif test data")
    parser.add_argument("--output-data",  default="test_data.json",
                        help="Where to save the generated observations (default: %(default)s)")
    parser.add_argument("--output-true",  default="test_true_params.json",
                        help="Where to save the true parameters  (default: %(default)s)")
    parser.add_argument("--k", type=int, default=500, help="Number of sequences (default: 500)")
    parser.add_argument("--w", type=int, default=6, help="Sequence length (default: 6)")
    parser.add_argument("--alpha", type=float, default=0.40, help="Motif probability (default: 0.40)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()
    return args
 
 
def main():
    args = parse_arguments()
    rng = np.random.default_rng(args.seed)
 
    k, w, alpha = args.k, args.w, args.alpha

    # Motif: each column drawn from a peaked Dirichlet (concentration 0.3
    #        pushes mass onto one letter, making the motif clearly visible)
    Theta = rng.dirichlet(np.full(4, 0.3), size=w).T   # (4, w)
 
    # Background: drawn from a mild Dirichlet (more uniform)
    Theta_bg = rng.dirichlet(np.full(4, 2.0))           # (4,)
 
    letters = np.arange(1, 5)   # {1, 2, 3, 4} ≡ {A, C, G, T}

    X = []
    Z = []
    for _ in range(k):
        z = int(rng.random() < alpha)
        Z.append(z)
        if z == 1:
            # Motif: each position sampled from its column of Theta
            seq = [int(rng.choice(letters, p=Theta[:, j])) for j in range(w)]
        else:
            # Background: all positions i.i.d. from Theta_bg
            seq = [int(rng.choice(letters, p=Theta_bg)) for _ in range(w)]
        X.append(seq)
 
    motif_count = sum(Z)
    print(f"Generated {k} sequences of length {w}")
    print(f"alpha (true) = {alpha}")
    print(f"motif sequences: {motif_count} / {k}  ({motif_count/k:.1%})")
    print(f"Theta_bg (true) = {np.round(Theta_bg, 4)}")
    print("Dominant letter per position:")
    for j in range(w):
        a = int(np.argmax(Theta[:, j]))
        print(f"pos {j+1}: {'ACGT'[a]}  ({Theta[a, j]:.3f})")

    data_out = {"alpha": alpha, "X": X}
    with open(args.output_data, "w") as f:
        json.dump(data_out, f)
    print(f"\nObservations saved to '{args.output_data}'")
 
    true_out = {
        "alpha": alpha,
        "Theta": Theta.tolist(),
        "Theta_bg": Theta_bg.tolist(),
    }
    with open(args.output_true, "w") as f:
        json.dump(true_out, f, indent=2)
    print(f"True parameters saved to '{args.output_true}'")
 
 
if __name__ == "__main__":
    main()