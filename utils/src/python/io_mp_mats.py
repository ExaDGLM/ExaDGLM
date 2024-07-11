import numpy as np
import os
import time
from datetime import timedelta
from mpmath import mp, mpf, matrix

# IEEE-754 floating point number
# double precision (64 bits = matissa 52 bits + exponent 11 bits + sign 1 bit)
# mp.prec: binary precision (matissa bits)
# mp.dps: decimal places
# prec ~= 3.33 * dps

DPS = 21  # decimal places
#mp.prec = 70 
mp.dps = DPS


def truncate_to_zero(arr):
    eps = np.finfo(arr.dtype).eps
    NODETOL = arr.max()*eps
    #NODETOL = 1e-15
    arr[np.abs(arr) < NODETOL] = 0
    
    
def write_mp_matrix(N, mat_name, mat_mp, base_dir):
    fpath = f"{base_dir}/mp_mats/N{N}/{mat_name}.txt"
    
    if not os.path.exists(f"{base_dir}/mp_mats/N{N}"):
        os.makedirs(f"{base_dir}/mp_mats/N{N}")
        
    nrow, ncol = mat_mp.rows, mat_mp.cols
    
    with open(fpath, 'w') as f:
        for i in range(nrow):
            for j in range(ncol):
                f.write(f"{mat_mp[i,j]}\n")


def read_mp_matrix(N, nrow, ncol, mat_name, base_dir):
    fpath = f"{base_dir}/mp_mats/N{N}/{mat_name}.txt"
        
    if os.path.exists(fpath):
        with open(fpath, 'r') as f:
            lines = f.readlines()
            assert len(lines) == nrow*ncol
            
            mat_mp = matrix(nrow, ncol)
            
            for ij, line in enumerate(lines):
                i = ij//ncol
                j = ij%ncol
                mat_mp[i,j]= mpf(line)
                
        return mat_mp
    
    else:
        return None
    
    
def convert_mp_matrix_to_numpy(mat_mp, truncate=True):
    nrow, ncol = mat_mp.rows, mat_mp.cols
    
    mat = np.zeros((nrow, ncol), 'f8')
        
    for i in range(nrow):
        for j in range(ncol):
            mat[i,j] = mat_mp[i,j]
            
    if truncate:
        truncate_to_zero(mat)
            
    return mat


def write_bin_matrix(N, mat_name, mat, base_dir):
    fpath = f"{base_dir}/bin_mats/N{N}/{mat_name}.bin"
    
    if not os.path.exists(f"{base_dir}/bin_mats/N{N}"):
        os.makedirs(f"{base_dir}/bin_mats/N{N}")
        
    with open(fpath, 'wb') as f:
         mat.ravel().tofile(f)
                
                
def read_bin_matrix(N, nrow, ncol, mat_name, base_dir):
    fpath = f"{base_dir}/bin_mats/N{N}/{mat_name}.bin"
        
    if os.path.exists(fpath):
        with open(fpath, 'rb') as f:
            mat = np.fromfile(f, dtype='f8').reshape((nrow, ncol))
                
        return mat
    
    else:
        raise FileNotFoundError(f"{fpath} does not exist!")
        

def print_elapsed_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        formatted_time = str(timedelta(seconds=elapsed_time))
        print(f"[{formatted_time}] {func.__name__}")
        return result
    
    return wrapper
