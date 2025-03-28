import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.autograd import grad
from optimized_algorithm1 import StateFunction

def algorithm2_optimized(vi_problem, y0, time_interval, max_round=5, max_iter_per_round=1000,
                        batch_size=16, lr=0.001, ir_threshold=None, device='cpu',
                        checkpoint_interval=100):
    """
    Optimized implementation of Algorithm 2 from the paper
    
    Parameters:
    -----------
    vi_problem: object
        The VI problem to solve with methods: 
        - ode_system(y): computes the ODE dynamics
        - vi_error(x): computes the VI error
        - projection_func(x): projects onto the feasible set
    y0: torch.Tensor or numpy.ndarray
        Initial condition for the first round
    time_interval: list or tuple
        Time interval [0, T]
    max_round: int
        Maximum number of rounds
    max_iter_per_round: int
        Maximum number of iterations per round
    batch_size: int
        Batch size for training
    lr: float
        Learning rate
    ir_threshold: float or None
        Improvement rate threshold (if None, only max_iter_per_round is used)
    device: str
        The device to use ('cpu' or 'cuda')
    checkpoint_interval: int
        Interval for saving checkpoints and printing progress
        
    Returns:
    --------
    x_best: torch.Tensor
        The best prediction to the VI
    """
    n_dim = vi_problem.n_dim
    T = time_interval[1]
    
    print(f"Running Algorithm 2 on device: {device}")
    print(f"Problem dimension: {n_dim}")
    print(f"Time interval: {time_interval}")
    print(f"Max rounds: {max_round}")
    print(f"Max iterations per round: {max_iter_per_round}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"IR threshold: {ir_threshold}")
    
    start_time = time.time()
    
    # Initialize the best prediction as the initial condition
    if isinstance(y0, np.ndarray):
        x_best = torch.tensor(y0, dtype=torch.float32).to(device)
    else:
        x_best = y0.to(device)
    
    epsilon_best = vi_problem.vi_error(x_best)
    
    # Store history for plotting
    loss_history = []
    vi_error_history = []
    
    # Main loop for rounds
    for round_idx in range(max_round):
        round_start_time = time.time()
        print(f"\nStarting Round {round_idx + 1}/{max_round}")
        print(f"Initial VI Error for this round: {epsilon_best.item():.6f}")
        
        # Construct a new neural network model
        model = StateFunction(n_dim, device=device)
        model.set_initial_condition(x_best)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Store the old error for IR calculation
        epsilon_old = epsilon_best.clone()
        
        # Training loop for the current round
        for iter in range(max_iter_per_round):
            # Sample time points uniformly
            t_batch = torch.rand(batch_size, dtype=torch.float32).to(device) * T
            t_batch.requires_grad_(True)
            
            # Forward pass
            y_pred = model(t_batch)
            
            # Compute the derivatives for the entire batch
            dy_dt = torch.zeros_like(y_pred)
            for i in range(batch_size):
                t_i = t_batch[i].view(-1).requires_grad_(True)
                y_i = model(t_i)
                dy_dt_i = grad(outputs=y_i, inputs=t_i, 
                              grad_outputs=torch.ones_like(y_i), 
                              create_graph=True)[0]
                dy_dt[i] = dy_dt_i
            
            # Expected derivatives from the ODE system
            phi_y = torch.stack([vi_problem.ode_system(y) for y in y_pred])
            
            # Compute the MSE loss (Eq. 14)
            loss = torch.mean(torch.sum((dy_dt - phi_y)**2, dim=1))
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Extract the current prediction (at t = T)
            T_tensor = torch.tensor([T], dtype=torch.float32).to(device)
            x_curr = model(T_tensor)
            x_curr = vi_problem.projection_func(x_curr)
            epsilon_curr = vi_problem.vi_error(x_curr)
            
            # Update the best prediction if necessary
            if epsilon_curr < epsilon_best:
                epsilon_best = epsilon_curr
                x_best = x_curr
            
            # Store history
            loss_history.append(loss.item())
            vi_error_history.append(epsilon_best.item())
            
            # Print progress at checkpoints
            if (iter + 1) % checkpoint_interval == 0:
                elapsed_time = time.time() - round_start_time
                print(f"  Iteration {iter+1}/{max_iter_per_round} ({elapsed_time:.2f}s): Loss = {loss.item():.6f}, VI Error = {epsilon_best.item():.6f}")
            
            # Check stopping criteria based on improvement rate
            if ir_threshold is not None and iter > 0:
                ir = (epsilon_old - epsilon_best) / epsilon_old * 100
                if ir < ir_threshold:
                    print(f"  Stopping at iteration {iter + 1}: Improvement rate {ir:.2f}% < threshold {ir_threshold}%")
                    break
        
        round_time = time.time() - round_start_time
        print(f"Completed Round {round_idx + 1} in {round_time:.2f}s: Best VI Error = {epsilon_best.item():.6f}")
        
        # Update epsilon_old for the next round
        epsilon_old = epsilon_best.clone()
    
    # Final statistics
    total_time = time.time() - start_time
    print(f"Algorithm 2 completed in {total_time:.2f} seconds")
    print(f"Final VI Error: {epsilon_best.item():.6f}")
    
    return x_best.cpu(), loss_history, vi_error_history