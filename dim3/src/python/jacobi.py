import numpy as np
import mpmath
from mpmath import mpf, matrix, sqrt, power, gamma, eigh

# IEEE-754 floating point number
# double precision (64 bits = matissa 52 bits + exponent 11 bits + sign 1 bit)
# mp.prec: binary precision (matissa bits)
# mp.dps: decimal places
# prec ~= 3.33 * dps

DPS = 21  # decimal places
#mpmath.mp.prec = 70 
mpmath.mp.dps = DPS


def jacobi_gq(N, a, b):
    '''
    Compute the N-th order Gauss quadrature points x, and weights, w,
    associated with the Jacobi polynomial
    '''
    
    x = matrix(N+1, 1)
    
    if N == 0:
        x[0] = (a-b)/(a+b+2)
        return x
    
    # Form symmetric matrix from recurrence
    J = matrix(N+1, N+1)
    
    for k in range(N+1):
        h1 = 2*k + a + b
        J[k,k] = -0.5*(a**2 - b**2)/(h1 + 2)/h1
        
        if k < N:
            kp = k+1
            J[k,k+1] = 2/(h1+2)*sqrt(kp*((kp+a+b)*(kp+a)*(kp+b))/(h1+1)/(h1+3))   
    
    if a + b < 10*np.finfo(float).eps:
        J[0,0] = 0.0
        
    J[:,:] = J[:,:] + J.transpose()[:,:]

    # Compute quadrature by eigenvalue solve
    eigen_val, eigen_vec = eigh(J)
    x = eigen_val
    
    if N%2 == 0:
        x[N//2] = 0

    return x


def jacobi_gll(N, a=0, b=0):
    '''
    Compute the N-th order Gauss-Lobatto-Legendre quadrature points x,
    associated with the Jacobi polynomial
    '''

    x = matrix(N+1, 1)
    x[0] = -1.0
    x[N] = 1.0
    
    if N > 1:
        x[1:-1] = jacobi_gq(N-2, a+1, b+1)
        
    return x
        
        
def jacobi(N, a, b, x):
    '''
    Evaluate Jacobi Polynomial at points x for order N
    Note : They are normalized to be orthonormal.    
    '''    
    PL = matrix(N+1, len(x))

    # Initial values P_0(x) and P_1(x)
    gamma0 = power(2, a+b+1)/(a+b+1)*gamma(a+1)*gamma(b+1)/gamma(a+b+1)
    PL[0,:] = 1/sqrt(gamma0)    
    if N == 0: return PL[0,:]

    gamma1 = (a+1)*(b+1)/(a+b+3)*gamma0
    for j in range(len(x)):
        PL[1,j] = ((a+b+2)*x[j]/2 + (a-b)/2)/sqrt(gamma1)
    if N == 1: return PL[1,:]

    # Repeat value in recurrence.
    aold = 2/(a+b+2)*sqrt((a+1)*(b+1)/(a+b+3))
    
    # Forward recurrence using the symmetry of the recurrence.
    for i in range(1, N):
        h1 = 2*i+a+b
        anew = 2/(h1+2)*sqrt( (i+1)*(i+1+a+b)*(i+1+a)*(i+1+b)/(h1+1)/(h1+3))
        bnew = -(a*a-b*b)/h1/(h1+2)
        
        for j in range(len(x)):
            PL[i+1,j] = 1/anew*(-aold*PL[i-1,j] + (x[j] - bnew)*PL[i,j])
        
        aold =anew
        
    return PL[N,:]  


def grad_jacobi(N, a, b, x):
    '''
    Evaluate the derivative of the Jacobi polynomial at points r for order N
    '''    
    dP = matrix(len(x), 1)
    
    if N > 0:
        for j in range(len(x)):
            dP[j] = sqrt(N*(N+a+b+1))*jacobi(N-1, a+1, b+1, x)[j]
        
    return dP
