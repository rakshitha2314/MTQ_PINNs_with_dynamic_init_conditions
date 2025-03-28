import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.autograd import grad

class StateFunction(nn.Module):
    def __init__(self, n_dim, device='cpu'):
        super(StateFunction, self).__init__()
        self.n_dim = n_dim
        self.device = device
        
        # Determine network width based on problem size
        if n_dim <= 30000:
            width = 300
        elif n_dim <= 50000:
            width = 400
        else:
            width = 500
            
        self.net = nn.Sequential(
            nn.Linear(1, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, n_dim)
        ).to(device)
        
        # Initialize initial condition to be used
        self.y0 = None
        
    def set_initial_condition(self, y0):
        """Set the initial condition for the ODE system"""
        if isinstance(y0, np.ndarray):
            self.y0 = torch.tensor(y0, dtype=torch.float32).to(self.device)
        elif isinstance(y0, torch.Tensor):
            self.y0 = y0.to(self.device)
        else:
            raise ValueError("Initial condition must be numpy array or torch tensor")
            
    def forward(self, t):
        """
        Implement the architecture from Eq. (12)
        y_hat(t, y0; w) = y0 + (1 - e^(-t)) * NN(t; w)
        """
        if self.y0 is None:
            raise ValueError("Initial condition has not been set")
            
        t = t.view(-1, 1).to(self.device)
        modifier = (1 - torch.exp(-t))
        nn_output = self.net(t)
        
        return self.y0 + modifier * nn_output

def algorithm1_optimized(vi_problem, y0, time_interval, max_iter=50000, batch_size=16, lr=0.001, 
                        device='cpu', checkpoint_interval=1000):
    """
    Optimized implementation of Algorithm 1 from the paper
    
    Parameters:
    -----------
    vi_problem: object
        The VI problem to solve with methods: 
        - ode_system(y): computes the ODE dynamics
        - vi_error(x): computes the VI error
        - projection_func(x): projects onto the feasible set
    y0: torch.Tensor or numpy.ndarray
        Initial condition
    time_interval: list or tuple
        Time interval [0, T]
    max_iter: int
        Maximum number of iterations
    batch_size: int
        Batch size for training
    lr: float
        Learning rate
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
    
    print(f"Running Algorithm 1 on device: {device}")
    print(f"Problem dimension: {n_dim}")
    print(f"Time interval: {time_interval}")
    print(f"Max iterations: {max_iter}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    
    start_time = time.time()
    
    # Construct the neural network model
    model = StateFunction(n_dim, device=device)
    if isinstance(y0, np.ndarray):
        y0_tensor = torch.tensor(y0, dtype=torch.float32).to(device)
    else:
        y0_tensor = y0.to(device)
    model.set_initial_condition(y0_tensor)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize best error and prediction
    T_tensor = torch.tensor([T], dtype=torch.float32).to(device)
    x_best = model(T_tensor)
    x_best = vi_problem.projection_func(x_best)
    epsilon_best = vi_problem.vi_error(x_best)
    
    # Store history for plotting
    loss_history = []
    vi_error_history = []
    
    # Main training loop
    for iter in range(max_iter):
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
            elapsed_time = time.time() - start_time
            print(f"Iteration {iter+1}/{max_iter} ({elapsed_time:.2f}s): Loss = {loss.item():.6f}, VI Error = {epsilon_best.item():.6f}")
    
    # Final statistics
    total_time = time.time() - start_time
    print(f"Algorithm 1 completed in {total_time:.2f} seconds")
    print(f"Final Loss: {loss_history[-1]:.6f}")
    print(f"Final VI Error: {epsilon_best.item():.6f}")
    
    return x_best.cpu(), loss_history, vi_error_history