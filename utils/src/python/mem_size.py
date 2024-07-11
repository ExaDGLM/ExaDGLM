import numpy as np
import re
import os


def unit_bytes(num):
    for unit in ['Bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1000:
            return num, unit
        num /= 1000
        
        
def print_arr_sizes(n_dict, arr_list, verbose):
    '''
    arr_list: [[vname, sizes, dtype], ...]
          ex) [['LIFT', 'NP*NFACE*FNP', 'double'], ...]
    '''
    #
    # maximum lengths for aligned print
    #
    maxlen_vname = 0
    maxlen_sizes = 0
    maxlen_dtype = 0
    for vname, sizes, dtype in arr_list:                
        maxlen_vname = max(maxlen_vname, len(vname))        
        maxlen_sizes = max(maxlen_sizes, len(sizes))
        maxlen_dtype = max(maxlen_dtype, len(dtype))
    
    #
    # print array sizes
    #
    total = 0
    for vname, sizes, dtype in arr_list:
        #size_str_list = sizes.split('*')
        #size_int_list = [n_dict[s] for s in size_str_list]
        #byte = np.prod(size_int_list)*{'int':4, 'float':4, 'double':8}[dtype]
        byte = eval(sizes, n_dict)*{'int':4, 'float':4, 'double':8}[dtype]
        total += byte        
        
        if verbose >= 2:
            b, unit = unit_bytes(byte)
            print(f"{vname.ljust(maxlen_vname)} ({dtype.ljust(maxlen_dtype)}, {sizes.ljust(maxlen_sizes)}): {b:>6.2f} {unit}")
        
    b, unit = unit_bytes(total)
    if verbose >= 1:
        print(f"total bytes: {b:6.2f} {unit}")
    
    return f"{b:6.2f} {unit}"
    
    
def print_mem_cpp(n_dict, real, cpp_class_code, verbose=1):
    """
    extract approximate memory sizes from the class.cpp file
    parsing vector arrays using regular expression
    
    verbose levels - 0: no output, 1: simple output, 2: detail output
    
    ex)
    input:
        '''
        vector<REAL> rx, ry, rz;
        vector<REAL> Fscale;
        vector<int> EtoB, vmapP;
        vector<vector<REAL>> p_u, p_k;

        rx.resize(nelem);
        ry.resize(nelem);
        rz.resize(nelem);
        Fscale.resize(nelem*NFACE);
        EtoB.resize(nelem*NFACE);
        vmapP.resize(nelem*NFACE*NFP);

        p_u.resize(NVAR);
        p_k.resize(NVAR);

        for (int i=0; i<NVAR; i++) {
            p_u[i]   = vector<REAL>(nelem*NP + buf_size*NFP);
            p_k[i]   = vector<REAL>(nelem*NP, 0);  // zero initialization
        }
        '''
    parse:
        rx:     (REAL, nelem)
        ry:     (REAL, nelem)
        rz:     (REAL, nelem)
        Fscale: (REAL, nelem*NFACE)
        EtoB:   ( int, nelem*NFACE)
        vmapP:  ( int, nelem*NFACE*NFP)
        p_u:    (REAL, NVAR*(nelem*NP + buf_size*NFP))
        p_k:    (REAL, NVAR*nelem*NP)
    """

    if verbose >= 1:
        print(f"{n_dict}")
        print("CPU array sizes")
    
    # dtype
    pattern1 = re.compile(r"vector<(int|REAL)>\s*([\w, ]+)\s*;")  # \w = [a-zA-Z0-9_]
    pattern2 = re.compile(r"vector<vector<(int|REAL)>>\s*([\w, ]+)\s*;")
    dtype_matches = pattern1.findall(cpp_class_code)
    dtype_matches += pattern2.findall(cpp_class_code)

    vdtype_dict = {}
    for dtype, vnames in dtype_matches:
        for vname in vnames.split(','):
            vdtype_dict[vname.strip()] = dtype

    # array size
    pattern3 = re.compile(r"(\S+).resize\(([\w*+ ]+)[,\)]")
    pattern4 = re.compile(r"(\S+)\[i\]\s*=\s*vector<(int|REAL)>\(([\w*+ ]+)[,\)]")
    size_matches = pattern3.findall(cpp_class_code)
    pack_size_matches = pattern4.findall(cpp_class_code)

    vsize_dict = {}  # {vname:sizes, ...}
    for vname, sizes in size_matches:
        if '+' in sizes:
            vsize_dict[vname] = '(' + sizes + ')'
        else:
            vsize_dict[vname] = sizes
    
    for vname, dtype, sizes in pack_size_matches:
        if '+' in  sizes:
            vsize_dict[vname] += '*' + '(' + sizes + ')'
        else:
            vsize_dict[vname] += '*' + sizes

    arr_list = []  # [(vname, sizes, dtype), ...]
    for vname, dtype in vdtype_dict.items():
        arr_list.append((vname, vsize_dict[vname], {'int':'int', 'REAL':real}[dtype]))
    
    return print_arr_sizes(n_dict, arr_list, verbose)    
    
    
def print_mem_cuda(n_dict, real, cpp_class_code, verbose=1):
    """
    extract approximate GPU memory sizes from the class.cu file
    parsing cudaMalloc arrays using regular expression
    
    verbose levels - 0: no output, 1: simple output, 2: detail output
    
    ex)
    input:
        '''
        cudaMalloc(&d_WDr,  NP*NP*sizeof(REAL));
        cudaMalloc(&d_LIFT, NP*NFACE*NFP*sizeof(REAL));
        cudaMalloc(&d_rx,   nelem*sizeof(REAL));
        cudaMalloc(&d_vmapP, nelem*NFACE*NFP*sizeof(int));
        cudaMalloc(&maxv, (nelem*NFACE + buf_size)*sizeof(REAL));

        for (int i=0; i<NVAR; i++) {
            cudaMalloc(&p_u[i],      (nelem*NP + buf_size*NFP)*sizeof(REAL));
            cudaMalloc(&p_ub[i],     nelem*NFACE*NFP*sizeof(REAL));        
            cudaMalloc(&p_fluxLM[i], nelem*NFACE*NFP*sizeof(REAL));
        }
        '''
    parse:
        WDr     : (REAL, NP*NP)
        LIFT    : (REAL, NP*NFACE*NFP)
        rx      : (REAL, nelem)
        vmapP   : ( int, nelem*NFACE*NFP)
        maxv    : (REAL, nelem*NFACE + buf_size)
        p_u     : (REAL, NVAR*(nelem*NP + buf_size*NFP))
        p_ub    : (REAL, NVAR*nelem*NFACE*NFP)
        p_fluxLM: (REAL, NVAR*nelem*NFACE*NFP)
    """

    if verbose >= 1:
        print(f"{n_dict}")
        print("GPU array sizes")
    
    pattern1 = r"cudaMalloc\(&(\w+),\s+([\w*+ ()]+)\*sizeof\((int|REAL)\)\);"
    pattern2 = r"cudaMalloc\(&(p_\w+)\[i\],\s+([\w*+ ()]+)\*sizeof\((int|REAL)\)\);"
    matches1 = re.findall(pattern1, cpp_class_code)
    matches2 = re.findall(pattern2, cpp_class_code)
    
    arr_list = [[vname, sizes, {'int':'int', 'REAL':real}[dtype]] for vname, sizes, dtype in matches1]
    arr_list.extend( [[vname, "NVAR*"+sizes, {'int':'int', 'REAL':real}[dtype]] for vname, sizes, dtype in matches2] )
    return print_arr_sizes(n_dict, arr_list, verbose)
