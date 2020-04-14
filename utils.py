import numpy as np

I = np.eye(2, dtype='complex128')
X = np.array([[0, 1], [1, 0]], dtype='complex128')
Y = np.array([[0, -1j], [1j, 0]], dtype='complex128')
Z = np.array([[1, 0], [0, -1]], dtype='complex128')
CX = np.array([[1, 0, 0, 0], 
               [0, 1, 0, 0], 
               [0, 0, 0, 1], 
               [0, 0, 1, 0]], dtype='complex128')
ket_0 = np.array([[1], [0]], dtype='complex128')
ket_1 = np.array([[0], [1]], dtype='complex128')

def NKron(*args):
    """Calculate a Kronecker product over variable number of inputs. Credit 
       to Jarrod McClean."""
    # 1x1 matrix to start with 
    result = np.array([[1.0]]) 
    for op in args:
        result = np.kron(result, op)
    return result

def NDot(*args):
    """Normal matrix multiplication using np.dot(a,b)"""
    if len(args) < 2: 
        raise ValueError("NDot needs at least 2 matrices to multiply")
    result = args[0]
    for op in args[1:]: 
        result = np.dot(result, op)
    return result

def array_eq(a, b, tol):
    """
    Checks if matrix a is equal to matrix b by doing elementwise 
    comparisons, allowing tolerance tol. 
    """
    if a.shape != b.shape: 
        return False 
    for index, x in np.ndenumerate(a):
        if np.abs(x - b[index]) > tol:
            return False 
    return True

def adjoint(u):
    """Returns conjugate transpose of u."""
    return np.transpose(np.conjugate(u))
