import argparse
import json
import sys
 
import numpy as np
from scipy.special import logsumexp
 
 
def parse_arguments():
    parser = argparse.ArgumentParser(description="Motif parameter estimator via EM")
    parser.add_argument("--input",  default="motif_data_known_alpha.json",
                        help="Input JSON file (default: %(default)s)")
    parser.add_argument("--output", default="estimated_params.json",
                        help="Output JSON file (default: %(default)s)")
    parser.add_argument("--estimate-alpha", choices=["yes", "no"], default="no",
                        help="Whether to estimate alpha (default: %(default)s)")
    args = parser.parse_args()
    return args.input, args.output, args.estimate_alpha
 
 

def compute_log_likelihood(X, log_theta, log_theta_b, alpha, k, w):
    log_p_motif = np.sum(log_theta[X - 1, np.arange(w)], axis=1)  
 
    log_p_bg = np.sum(log_theta_b[X - 1], axis=1)             
 
    log_alpha = np.log(alpha)
    log_1malpha = np.log(1.0 - alpha)

    stacked = np.stack([log_alpha + log_p_motif,
                        log_1malpha + log_p_bg], axis=1)
    log_lik = np.sum(logsumexp(stacked, axis=1))
    return log_lik
 

def run_em(X, k, w, alpha, estimate_alpha,
           max_iter=500, tol=1e-8, pseudo=1e-3, rng=None):

    if rng is None:
        rng = np.random.default_rng()
 
    theta_raw = rng.dirichlet(np.ones(4), size=w).T       
    log_theta = np.log(theta_raw)

    theta_b_raw = rng.dirichlet(np.ones(4))             
    log_theta_b = np.log(theta_b_raw)
 
    prev_ll = -np.inf
 
    for iteration in range(max_iter):
        log_p_motif = np.sum(log_theta[X - 1, np.arange(w)], axis=1)  
        log_p_bg = np.sum(log_theta_b[X - 1], axis=1)          
 
        log_alpha = np.log(alpha)
        log_1ma = np.log(1.0 - alpha)
 
        log_num_motif = log_alpha + log_p_motif       
        log_num_bg = log_1ma + log_p_bg          
 
        log_denom = logsumexp(np.stack([log_num_motif, log_num_bg], axis=1), axis=1)

        log_gamma = log_num_motif - log_denom          
        gamma = np.exp(log_gamma)                  

        sum_gamma = gamma.sum()                     
        sum_1mgamma = (1.0 - gamma).sum()            
 
        new_theta = np.full((4, w), pseudo)
        for a in range(4):
            new_theta[a] += np.sum(gamma[:, None] * (X == a + 1), axis=0)  

        new_theta /= new_theta.sum(axis=0, keepdims=True)
        log_theta = np.log(new_theta)
 
        new_theta_b = np.full(4, pseudo)
        for a in range(4):
            new_theta_b[a] += np.sum((1.0 - gamma) * np.sum(X == a + 1, axis=1))
        new_theta_b /= new_theta_b.sum()
        log_theta_b = np.log(new_theta_b)

        if estimate_alpha:
            alpha = np.clip(sum_gamma / k, 1e-6, 1.0 - 1e-6)
 
        ll = compute_log_likelihood(X, log_theta, log_theta_b, alpha, k, w)
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll
 
    return log_theta, log_theta_b, alpha, prev_ll
 
 
def run_em_multi_restart(X, k, w, alpha_init, estimate_alpha,
                         n_restarts=25, max_iter=500, tol=1e-8,
                         pseudo=1e-3, seed=0):

    best_ll = -np.inf
    best_log_theta = None
    best_log_theta_b = None
    best_alpha = alpha_init
 
    for restart in range(n_restarts):
        rng = np.random.default_rng(seed + restart * 1000)
        log_theta, log_theta_b, alpha_out, ll = run_em(
            X, k, w,
            alpha = alpha_init,
            estimate_alpha = estimate_alpha,
            max_iter = max_iter,
            tol = tol,
            pseudo = pseudo,
            rng = rng,
        )
        if ll > best_ll:
            best_ll = ll
            best_log_theta = log_theta
            best_log_theta_b = log_theta_b
            best_alpha = alpha_out
 
    print(f"Best log-likelihood over {n_restarts} restarts: {best_ll:.4f}",
          file=sys.stderr)
    return best_log_theta, best_log_theta_b, best_alpha
 
 
def main():
    input_file, output_file, estimate_alpha_flag = parse_arguments()
    estimate_alpha = (estimate_alpha_flag == "yes")
 
    with open(input_file, "r") as f:
        data = json.load(f)
 
    X = np.asarray(data["X"], dtype=int)   
    k, w = X.shape
 
    if estimate_alpha:
        alpha_init = 0.5   
    else:
        alpha_init = float(data["alpha"])
 
    print(f"Loaded {k} sequences of length {w}, alpha={'unknown' if estimate_alpha else alpha_init}",
          file=sys.stderr)
 

    log_theta, log_theta_b, alpha_final = run_em_multi_restart(
        X, k, w,
        alpha_init = alpha_init,
        estimate_alpha = estimate_alpha,
        n_restarts = 30,
        max_iter = 1000,
        tol = 1e-9,
        pseudo = 1e-3,
        seed = 42,
    )
 
    Theta = np.exp(log_theta)        
    Theta_bg = np.exp(log_theta_b)    
 
    estimated_params = {
        "alpha": float(alpha_final),
        "Theta": Theta.tolist(),      
        "Theta_bg": Theta_bg.tolist(),
    }
 
    with open(output_file, "w") as f:
        json.dump(estimated_params, f, indent=2)
 
    print(f"Estimated parameters saved to '{output_file}'", file=sys.stderr)
    print(f"alpha = {alpha_final:.4f}", file=sys.stderr)
    print(f"Theta_bg = {np.round(Theta_bg, 4)}", file=sys.stderr)
    print("Theta (4xw) =", file=sys.stderr)
    for a, letter in enumerate("ACGT"):
        print(f"{letter}: {np.round(Theta[a], 4)}", file=sys.stderr)
 
 
if __name__ == "__main__":
    main()