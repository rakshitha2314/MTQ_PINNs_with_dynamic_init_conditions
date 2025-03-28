import torch
import numpy as np
import requests
import io
import os

def download_file(url, local_filename=None):
    """
    Download a file from a URL and save it locally
    """
    if local_filename is None:
        local_filename = url.split('/')[-1]
    
    # Check if file already exists
    if os.path.exists(local_filename):
        print(f"File '{local_filename}' already exists. Using local copy.")
        return local_filename
    
    print(f"Downloading {url}...")
    response = requests.get(url)
    response.raise_for_status()
    
    with open(local_filename, 'wb') as f:
        f.write(response.content)
    
    print(f"Download completed: {local_filename}")
    return local_filename

def load_vi_data(url, local_filename=None):
    """
    Load VI data from local .npy files
    
    Returns:
    --------
    data: dict
        Dictionary containing problem data (a, b, c, d, e, l, h)
    """
    try:
        # If local_filename is a directory, look for .npy files
        if os.path.isdir(local_filename):
            data = {}
            for param in ['a', 'b', 'c', 'd', 'e', 'l', 'h']:
                param_file = os.path.join(local_filename, f"{param}.npy")
                if os.path.exists(param_file):
                    data[param] = torch.tensor(np.load(param_file), dtype=torch.float32)
                else:
                    print(f"Warning: {param_file} not found")
            
            # Verify we have all necessary parameters
            required_params = ['a', 'b', 'c', 'd', 'e']
            if all(param in data for param in required_params):
                return data
            else:
                print("Missing required parameters")
                return None
        
        # If it's a single file
        elif local_filename and os.path.exists(local_filename):
            data = np.load(local_filename)
            converted_data = {}
            for key in data.files:
                converted_data[key] = torch.tensor(data[key], dtype=torch.float32)
            return converted_data
        
        else:
            print(f"File or directory {local_filename} not found")
            return None
    
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None