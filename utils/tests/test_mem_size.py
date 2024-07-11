from textwrap import dedent

import path
from mem_size import print_mem_cpp
from mem_size import print_mem_cuda


def test_print_mem_cpp(capsys):
    n_dict = {'nelem': 750, 'N': 6, 'NP': 84, 'NFACE': 4, 'NFP': 28, 'NVAR':5, 'comm_size':2, 'buf_size':100}
    real = {'f8':'double', 'f4':'float'}['f8']
    class_cpp_fpath = path.join(path.cwd, 'sample1.class.cpp')

    with open(class_cpp_fpath, 'r') as f:
        print_mem_cpp(n_dict, real, f.read(), verbose=2)
    
    out, err = capsys.readouterr()
    expect = \
'''{'nelem': 750, 'N': 6, 'NP': 84, 'NFACE': 4, 'NFP': 28, 'NVAR': 5, 'comm_size': 2, 'buf_size': 100}
CPU array sizes
rx             (double, nelem                         ):   6.00 KB
ry             (double, nelem                         ):   6.00 KB
rz             (double, nelem                         ):   6.00 KB
Fscale         (double, nelem*NFACE                   ):  24.00 KB
maxv           (double, (nelem*NFACE + buf_size)      ):  24.80 KB
EtoB           (int   , nelem*NFACE                   ):  12.00 KB
vmapP          (int   , nelem*NFACE*NFP               ): 336.00 KB
k              (double, nelem*NP                      ): 504.00 KB
comm_face_idxs (int   , comm_size*3                   ):  24.00 Bytes
p_u            (double, NVAR*(nelem*NP + buf_size*NFP)):   2.63 MB
p_k            (double, NVAR*nelem*NP                 ):   2.52 MB
total bytes:   6.07 MB
'''    
    assert out == expect
    
    
def test_print_mem_cuda(capsys):
    n_dict = {'nelem': 750, 'N': 6, 'NP': 84, 'NFACE': 4, 'NFP': 28, 'NVAR': 5, 'buf_size': 100}
    real = {'f8':'double', 'f4':'float'}['f8']
    class_cu_fpath = path.join(path.cwd, 'sample2.class.cu')

    with open(class_cu_fpath, 'r') as f:
        print_mem_cuda(n_dict, real, f.read(), verbose=2)
    
    out, err = capsys.readouterr()
    expect = \
'''{'nelem': 750, 'N': 6, 'NP': 84, 'NFACE': 4, 'NFP': 28, 'NVAR': 5, 'buf_size': 100}
GPU array sizes
WDr      (double, NP*NP                         ):  56.45 KB
LIFT     (double, NP*NFACE*NFP                  ):  75.26 KB
rx       (double, nelem                         ):   6.00 KB
vmapP    (int   , nelem*NFACE*NFP               ): 336.00 KB
maxv     (double, (nelem*NFACE + buf_size)      ):  24.80 KB
p_u      (double, NVAR*(nelem*NP + buf_size*NFP)):   2.63 MB
p_ub     (double, NVAR*nelem*NFACE*NFP          ):   3.36 MB
p_fluxLM (double, NVAR*nelem*NFACE*NFP          ):   3.36 MB
total bytes:   9.85 MB
'''    
    assert out == expect 
