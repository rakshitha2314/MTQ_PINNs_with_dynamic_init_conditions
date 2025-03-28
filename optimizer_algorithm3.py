import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Define the neural network model
class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(PINN, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Define the function G(x)
def G(x, a, b, c, d, e):
    G_x = torch.zeros_like(x)
    for i in range(len(x) - 1):
        G_x[i] = a[i] * torch.sin(x[i] + b[i]) - c[i] * torch.log(x[i] + 1 + d[i]) + e[i]
    G_x[-1] = a[-1] * torch.sin(x[-1] + b[-1]) + e[-1]
    return G_x

# Define the projection function onto the feasible set Ω
def project_onto_feasible_set(x, l, h):
    l_tensor = torch.tensor(l, dtype=torch.float32)
    h_tensor = torch.tensor(h, dtype=torch.float32)
    return torch.clamp(x, min=l_tensor, max=h_tensor)

# Define the function to calculate the VI error
def calculate_vi_error(x, x_star, G_x_star):
    return torch.sum((x - x_star) * G_x_star)

# Define the loss function
def loss_function(y_pred, y_true):
    return torch.mean((y_pred - y_true)**2)

# Load data from .npy files
def load_data(problem_instance):
    a = np.load(f'{problem_instance}/a.npy')
    b = np.load(f'{problem_instance}/b.npy')
    c = np.load(f'{problem_instance}/c.npy')
    d = np.load(f'{problem_instance}/d.npy')
    e = np.load(f'{problem_instance}/e.npy')
    l = np.load(f'{problem_instance}/l.npy')
    h = np.load(f'{problem_instance}/h.npy')
    
    return a, b, c, d, e, l, h

# Define the main function for Algorithm 3 with logging of MSE loss and VI error
def alg_3(VI, T, y0, input_dim, hidden_dim, output_dim, num_layers, l, h, a, b, c, d, e, max_iter, lr, batch_size, X_train, y_train):
    # Initialize the neural network model
    model = PINN(input_dim, hidden_dim, output_dim, num_layers)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Create data loaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Set the initial condition
    x_best = torch.tensor(y0, dtype=torch.float32)
    y_hat = model(x_best)
    
    E = lambda y: calculate_vi_error(y, x_best, G(x_best, a, b, c, d, e))
    epsilon_best = E(y_hat)
    
    mse_losses = []
    vi_errors = []
    
    for iter in range(max_iter):
        # Training step
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_hat = model(X_batch)
            loss = loss_function(y_hat, y_batch)
            loss.backward()
            optimizer.step()
        
        # Extract the NN prediction
        x_curr = model(x_best).detach()
        
        # Project x_curr onto the feasible set Ω
        x_curr = project_onto_feasible_set(x_curr, l, h)
        
        # Calculate the VI error
        epsilon_curr = E(x_curr)
        
        # Log the MSE loss and VI error
        mse_losses.append(loss.item())
        vi_errors.append(epsilon_curr.item())
        
        # Update the best prediction if the current error is lower
        if epsilon_curr < epsilon_best:
            epsilon_best = epsilon_curr
            x_best = x_curr
            # Update the initial condition
            y_hat = model(x_best)
    
    return x_best, mse_losses, vi_errors

# Example usage
VI = [0, 1]  # Placeholder for the VI problem definition
T = 100.0  # Time interval [0, 100]
hidden_dims = {'E1': 300, 'E2': 400, 'E3': 500}
num_layers = 5
max_iter = 1000
lr = 0.001
batch_size = 16

# Define problem sizes for each instance
problem_sizes = {'E1': 30000, 'E2': 50000, 'E3': 100000}

# Load data for each problem instance
problem_instances = ['E1', 'E2', 'E3']
results = {}

for instance in problem_instances:
    a, b, c, d, e, l, h = load_data(instance)
    input_dim = problem_sizes[instance]
    output_dim = problem_sizes[instance]
    hidden_dim = hidden_dims[instance]
    y0 = np.ones(input_dim)  # Initial condition as the all-ones vector
    
    # Generate random training data (replace with actual data if available)
    X_train = np.random.rand(100, input_dim)
    y_train = np.random.rand(100, output_dim)
    
    # Solve the VI problem
    x_best, mse_losses, vi_errors = alg_3(VI, T, y0, input_dim, hidden_dim, output_dim, num_layers, l, h, a, b, c, d, e, max_iter, lr, batch_size, X_train, y_train)
    results[instance] = (mse_losses, vi_errors)

# Generate plots
fig, axes = plt.subplots(len(problem_instances), 2, figsize=(12, 18))

for i, instance in enumerate(problem_instances):
    mse_losses, vi_errors = results[instance]
    
    # Plot MSE loss
    axes[i, 0].plot(mse_losses)
    axes[i, 0].set_title(f'{instance} - MSE Loss')
    axes[i, 0].set_xlabel('Epoch')
    axes[i, 0].set_ylabel('MSE Loss')

    # Plot VI error
    axes[i, 1].plot(vi_errors)
    axes[i, 1].set_title(f'{instance} - VI Error')
    axes[i, 1].set_xlabel('Epoch')
    axes[i, 1].set_ylabel('VI Error')

plt.tight_layout()
plt.show()

# Generate tables
for instance in problem_instances:
    mse_losses, vi_errors = results[instance]
    data = {
        'Epoch': list(range(1, max_iter + 1)),
        'MSE Loss': mse_losses,
        'VI Error': vi_errors
    }
    df = pd.DataFrame(data)
    print(f'\nTable for {instance}')
    print(df.to_string(index=False))