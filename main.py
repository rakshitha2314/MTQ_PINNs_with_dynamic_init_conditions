import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse
from utils import load_vi_data
from vi_problems import BoxConstrainedVI, SphericalConstrainedVI
from optimized_algorithm1 import algorithm1_optimized
from optimized_algorithm2 import algorithm2_optimized

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Algorithms 1 and 2 on VI problems')
    parser.add_argument('--algorithm', type=int, default=0, choices=[1, 2, 0], 
                        help='Algorithm to run (1, 2, or 0 for both)')
    parser.add_argument('--problem', type=str, default='box', choices=['box', 'sphere'], 
                        help='Problem type (box or sphere)')
    parser.add_argument('--instance', type=str, default='E1', choices=['E1', 'E2', 'E3', 'E4', 'E5', 'E6'],
                        help='Problem instance')
    parser.add_argument('--max_iter', type=int, default=10000, 
                        help='Maximum iterations for Algorithm 1')
    parser.add_argument('--max_round', type=int, default=5, 
                        help='Maximum rounds for Algorithm 2')
    parser.add_argument('--max_iter_per_round', type=int, default=1000, 
                        help='Maximum iterations per round for Algorithm 2')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size')
    parser.add_argument('--time_interval', type=float, default=10.0, 
                        help='Time interval endpoint T (interval is [0, T])')
    parser.add_argument('--ir_threshold', type=float, default=10.0, 
                        help='Improvement rate threshold for Algorithm 2')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cpu', 'cuda'], help='Device to use (cpu or cuda)')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Problem parameters
    instance_to_size = {
        'E1': 30000, 'E2': 50000, 'E3': 100000,
        'E4': 30000, 'E5': 50000, 'E6': 100000
    }
    
    instance_to_url = {
        'E1': 'https://drive.google.com/file/d/1GSxj2mLaHDTwdHxsBq7WvH9cADkDJj9v/view?usp=sharing',
        'E2': 'https://drive.google.com/file/d/1kygJI9bEBu-nxZhlc121AP-5IpKDmkDW/view?usp=sharing',
        'E3': 'https://drive.google.com/file/d/1MkN75WNp-HnbFnugb3FCT9LSqwgufYKo/view?usp=sharing',
        'E4': 'https://drive.google.com/file/d/1GSxj2mLaHDTwdHxsBq7WvH9cADkDJj9v/view?usp=sharing',
        'E5': 'https://drive.google.com/file/d/1kygJI9bEBu-nxZhlc121AP-5IpKDmkDW/view?usp=sharing',
        'E6': 'https://drive.google.com/file/d/1MkN75WNp-HnbFnugb3FCT9LSqwgufYKo/view?usp=sharing'
    }
    
    # Determine problem size and URL
    n_dim = instance_to_size[args.instance]
    data_url = instance_to_url[args.instance]
    
    print(f"Running {args.instance} ({n_dim} variables) on {args.device}")
    print(f"Problem type: {args.problem}")
    
    # Try to load data from URL
    # When calling load_vi_data
    data = load_vi_data(data_url, 'E1_Problem data')  # Pass the directory name
    
    # If data cannot be loaded, generate random data
    if data is None:
        print("Generating random data...")
        a = np.random.uniform(0, 10, n_dim)
        b = np.random.uniform(0, 10, n_dim)
        c = np.random.uniform(0, 10, n_dim)
        d = np.random.uniform(0, 10, n_dim)
        e = np.random.uniform(0, 10, n_dim)
        
        data = {
            'a': torch.tensor(a, dtype=torch.float32),
            'b': torch.tensor(b, dtype=torch.float32),
            'c': torch.tensor(c, dtype=torch.float32),
            'd': torch.tensor(d, dtype=torch.float32),
            'e': torch.tensor(e, dtype=torch.float32)
        }
        
        if args.problem == 'box':
            l = np.random.uniform(0, 10, n_dim)
            h = np.random.uniform(70, 100, n_dim)
            data['l'] = torch.tensor(l, dtype=torch.float32)
            data['h'] = torch.tensor(h, dtype=torch.float32)
    
    # Create the VI problem
    # Create the VI problem
    if args.problem == 'box':
        vi_problem = BoxConstrainedVI(
            n_dim=n_dim,
            a=data['a'],
            b=data['b'],
            c=data['c'],
            d=data['d'],
            e=data['e'],
            l=data['l'],
            h=data['h'],
            device=args.device
        )
    else:  # sphere
        beta_values = {'E4': 1000, 'E5': 2000, 'E6': 3000}
        beta = beta_values[args.instance]
        
        vi_problem = SphericalConstrainedVI(
            n_dim=n_dim,
            beta=beta,
            a=data['a'],
            b=data['b'],
            c=data['c'],
            d=data['d'],
            e=data['e'],
            device=args.device
        )
    
    # Create initial condition (all ones vector as in the paper)
    y0 = torch.ones(n_dim, dtype=torch.float32)
    time_interval = [0, args.time_interval]
    
    results = {}
    
    # Run Algorithm 1 if selected
    if args.algorithm == 1 or args.algorithm == 0:
        print("\nRunning Algorithm 1...")
        x_best_alg1, loss_history_alg1, vi_error_history_alg1 = algorithm1_optimized(
            vi_problem=vi_problem,
            y0=y0,
            time_interval=time_interval,
            max_iter=args.max_iter,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device
        )
        
        results['algorithm1'] = {
            'x_best': x_best_alg1,
            'loss_history': loss_history_alg1,
            'vi_error_history': vi_error_history_alg1
        }
    
    # Run Algorithm 2 if selected
    if args.algorithm == 2 or args.algorithm == 0:
        print("\nRunning Algorithm 2...")
        x_best_alg2, loss_history_alg2, vi_error_history_alg2 = algorithm2_optimized(
            vi_problem=vi_problem,
            y0=y0,
            time_interval=time_interval,
            max_round=args.max_round,
            max_iter_per_round=args.max_iter_per_round,
            batch_size=args.batch_size,
            lr=args.lr,
            ir_threshold=args.ir_threshold,
            device=args.device
        )
        
        results['algorithm2'] = {
            'x_best': x_best_alg2,
            'loss_history': loss_history_alg2,
            'vi_error_history': vi_error_history_alg2
        }
    
    # Save results
    result_filename = f"{args.results_dir}/{args.instance}_{args.problem}_results.npz"
    np.savez(result_filename, **results)
    print(f"Results saved to {result_filename}")
    
    # Plot comparison if both algorithms were run
    if args.algorithm == 0:
        plt.figure(figsize=(12, 5))
        
        # Plot VI error history
        plt.subplot(1, 2, 1)
        plt.semilogy(vi_error_history_alg1, label=f'Algorithm 1 (Final: {vi_error_history_alg1[-1]:.6f})')
        plt.semilogy(vi_error_history_alg2, label=f'Algorithm 2 (Final: {vi_error_history_alg2[-1]:.6f})')
        plt.xlabel('Iteration')
        plt.ylabel('VI Error (log scale)')
        plt.title(f'{args.instance} - VI Error vs. Iteration')
        plt.legend()
        plt.grid(True)
        
        # Plot loss history
        plt.subplot(1, 2, 2)
        plt.semilogy(loss_history_alg1, label=f'Algorithm 1')
        plt.semilogy(loss_history_alg2, label=f'Algorithm 2')
        plt.xlabel('Iteration')
        plt.ylabel('Loss (log scale)')
        plt.title(f'{args.instance} - Loss vs. Iteration')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{args.results_dir}/{args.instance}_{args.problem}_comparison.png")
        plt.show()

if __name__ == "__main__":
    main()