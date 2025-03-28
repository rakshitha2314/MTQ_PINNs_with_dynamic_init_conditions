import torch
import numpy as np

class BoxConstrainedVI:
    def __init__(self, n_dim, a, b, c, d, e, l, h, device='cpu'):
        """
        Initialize a box-constrained VI problem
        
        Parameters:
        -----------
        n_dim: int
            Dimension of the problem
        a, b, c, d, e: torch.Tensor or numpy.ndarray
            Problem data for the function G
        l, h: torch.Tensor or numpy.ndarray
            Lower and upper bounds for the box constraints
        device: str
            The device to use ('cpu' or 'cuda')
        """
        self.n_dim = n_dim
        self.device = device
        
        # Convert numpy arrays to torch tensors if necessary
        self.a = torch.tensor(a, dtype=torch.float32).to(device) if isinstance(a, np.ndarray) else a.to(device)
        self.b = torch.tensor(b, dtype=torch.float32).to(device) if isinstance(b, np.ndarray) else b.to(device)
        self.c = torch.tensor(c, dtype=torch.float32).to(device) if isinstance(c, np.ndarray) else c.to(device)
        self.d = torch.tensor(d, dtype=torch.float32).to(device) if isinstance(d, np.ndarray) else d.to(device)
        self.e = torch.tensor(e, dtype=torch.float32).to(device) if isinstance(e, np.ndarray) else e.to(device)
        self.l = torch.tensor(l, dtype=torch.float32).to(device) if isinstance(l, np.ndarray) else l.to(device)
        self.h = torch.tensor(h, dtype=torch.float32).to(device) if isinstance(h, np.ndarray) else h.to(device)
    
    def G_func(self, x):
        """
        Compute G(x) as per Eq. (21)
        """
        x = x.to(self.device)
        
        # Ensure x is a 1D tensor
        if x.dim() == 0:
            x = x.unsqueeze(0)
        elif x.dim() > 1:
            x = x.squeeze()
        
        # Ensure x has the same length as other parameters
        if len(x) != self.n_dim:
            raise ValueError(f"Input x must have length {self.n_dim}, got {len(x)}")
        
        # Prepare result tensor
        result = torch.zeros_like(x)
        
        # Compute for all but the last element
        for i in range(self.n_dim - 1):
            result[i] = (
                self.a[i] * torch.sin(x[i] + self.b[i]) - 
                self.c[i] * torch.log(x[i + 1] + self.d[i]) + 
                self.e[i]
            )
        
        # Last element
        result[-1] = self.a[-1] * torch.sin(x[-1] + self.b[-1]) + self.e[-1]
        
        return result

    def vi_error(self, x):
        """
        Calculate the VI error as defined in Eq. (17)
        E(x_pred) = ||P_Ω(x_pred - G(x_pred)) - x_pred||
        """
        x = x.to(self.device)
        
        # Ensure x is a 1D tensor
        if x.dim() == 0:
            x = x.unsqueeze(0)
        elif x.dim() > 1:
            x = x.squeeze()
        
        G_x = self.G_func(x)
        projected = self.projection_func(x - G_x)
        return torch.norm(projected - x)
    
    def ode_system(self, y):
        """
        Implement the ODE system from Eq. (9-10)
        Φ(y) = -G(P_Ω(y)) + P_Ω(y) - y
        """
        y = y.to(self.device)
        y_proj = self.projection_func(y)
        return -self.G_func(y_proj) + y_proj - y
    
    def projection_func(self, x):
        """
        Project onto the box-constrained feasible set
        """
        x = x.to(self.device)
        return torch.max(torch.min(x, self.h), self.l)
    
    def feasible_set_func(self, x):
        """
        Check if x is in the feasible set
        """
        x = x.to(self.device)
        return torch.all((x >= self.l) & (x <= self.h))
    


class SphericalConstrainedVI:
    def __init__(self, n_dim, beta, a, b, c, d, e, device='cpu'):
        """
        Initialize a spherically-constrained VI problem
        
        Parameters:
        -----------
        n_dim: int
            Dimension of the problem
        beta: float
            Radius of the sphere
        a, b, c, d, e: torch.Tensor or numpy.ndarray
            Problem data for the function G
        device: str
            The device to use ('cpu' or 'cuda')
        """
        self.n_dim = n_dim
        self.beta = beta
        self.device = device
        
        # Convert numpy arrays to torch tensors if necessary
        self.a = torch.tensor(a, dtype=torch.float32).to(device) if isinstance(a, np.ndarray) else a.to(device)
        self.b = torch.tensor(b, dtype=torch.float32).to(device) if isinstance(b, np.ndarray) else b.to(device)
        self.c = torch.tensor(c, dtype=torch.float32).to(device) if isinstance(c, np.ndarray) else c.to(device)
        self.d = torch.tensor(d, dtype=torch.float32).to(device) if isinstance(d, np.ndarray) else d.to(device)
        self.e = torch.tensor(e, dtype=torch.float32).to(device) if isinstance(e, np.ndarray) else e.to(device)
    
    def G_func(self, x):
        """
        Compute G(x) as per Eq. (23)
        """
        x = x.to(self.device)
        
        # Ensure x is a 1D tensor
        if x.dim() == 0:
            x = x.unsqueeze(0)
        elif x.dim() > 1:
            x = x.squeeze()
        
        # Ensure x has the same length as other parameters
        if len(x) != self.n_dim:
            raise ValueError(f"Input x must have length {self.n_dim}, got {len(x)}")
        
        # Prepare result tensor
        result = torch.zeros_like(x)
        
        # Compute for all but the last element
        for i in range(self.n_dim - 1):
            result[i] = (
                self.a[i] * torch.sin(x[i] + self.b[i]) - 
                self.c[i] * torch.cos(x[i + 1] + self.d[i]) + 
                self.e[i]
            )
        
        # Last element
        result[-1] = self.a[-1] * torch.sin(x[-1] + self.b[-1]) + self.e[-1]
        
        return result

    def vi_error(self, x):
        """
        Calculate the VI error as defined in Eq. (17)
        E(x_pred) = ||P_Ω(x_pred - G(x_pred)) - x_pred||
        """
        x = x.to(self.device)
        
        # Ensure x is a 1D tensor
        if x.dim() == 0:
            x = x.unsqueeze(0)
        elif x.dim() > 1:
            x = x.squeeze()
        
        G_x = self.G_func(x)
        projected = self.projection_func(x - G_x)
        return torch.norm(projected - x)
    
    def ode_system(self, y):
        """
        Implement the ODE system from Eq. (9-10)
        Φ(y) = -G(P_Ω(y)) + P_Ω(y) - y
        """
        y = y.to(self.device)
        y_proj = self.projection_func(y)
        return -self.G_func(y_proj) + y_proj - y
    
    def projection_func(self, x):
        """
        Project onto the spherically-constrained feasible set
        """
        x = x.to(self.device)
        norm_x = torch.norm(x)
        if norm_x > self.beta:
            return self.beta * x / norm_x
        return x
    
    def feasible_set_func(self, x):
        """
        Check if x is in the feasible set
        """
        x = x.to(self.device)
        return torch.norm(x) <= self.beta