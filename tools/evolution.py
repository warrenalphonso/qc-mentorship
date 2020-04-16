import numpy as np
import scipy.linalg
import unittest 

from utils import * 

def time_evo(H, t: float): 
    """
    Returns the solution to the Schrodinger equation for Hamiltonian H: 
    U = exp(-itH/h) where h is Planck's reduced constant. 
    The exponential is a matrix function that takes the exponential of (-itv/h) 
    where v are the eigenvalues of H and multiplies by the outer product of the 
    corresponding eigenvectors. 

    We absorb h into t. I think. 

    Parameters: 
        H : (n, n) matrix
            Hamiltonian matrix. Must be Hermitian. 
        t : float 
            Time to evolve for. 
    Returns: 
        u : u = exp(-itH) as described above. u is a unitary matrix. 
    """
    prc = 1.054571817e-34 # Planck's reduced constant in Joule seconds
    # TODO: Does including prc matter? Can I assume it's folded into t?
    # No, I don't think I can. I should test incorporating prc here. But I 
    # can't just divide it because it's too small and Python won't handle it 
    # well. Instead I need to use logarithms!
    return scipy.linalg.expm(-1j * t * H)

def prod_time_evo(Hs: list, ts: float):
    """
    The product of each Hamiltonian H in Hs evolved for correspoidning time t 
    in ts. This can be used for Trotterization if the ts are all the same. 
    We use this for each step in the Variational Hamiltonian Ansatz (VHA). 

    Parameters: 
        Hs: list of (n, n) matrices (sub-matrices of Hamiltonian)
            A list of the Hamiltonian sub-matrices. The sum of the Hamiltonians 
            in Hs is the actual Hamiltonian of the system. Each H in Hs must 
            be Hermitian. 
        ts: list of ints (times) 
            A list of the times to evolve the sub-matrices in Hs by. 
            Hs[i] is evolved for ts[i] seconds. 
    Returns: 
        u: (n, n) unitary matrix. 
            The product of each Hamiltonian sub-matrix evolved for its 
            corresponding time. Since we're multiplying k unitaries, our result 
            must also be unitary. 
    """
    result = 1 
    for (H, t) in zip(Hs, ts):
        result = NDot(result, time_evo(H, t))
    return result

# TESTS 
tol = 0.005
herms = [np.eye(4), X, Y, Z, I, NKron(X, Y), NKron(Y, Z, Z, I)]

# Test scipy.linalg.expm() to make sure it does what I expect. Use X. 
class TestExpm(unittest.TestCase):
    def test_X(self):
        # Making sure expm doesn't just exponentiate elements 
        expm_X = scipy.linalg.expm(X)
        exp_X = np.exp(X)
        # I worked it out on paper and these should be different 
        self.assertFalse(array_eq(expm_X, exp_X, tol))
        # Make sure expm_X has the correct values 
        e = 2.718281
        corr_expm_X = .5 * np.array([[e + 1/e, e - 1/e], 
                                     [e - 1/e, e + 1/e]], dtype='complex128')
        self.assertTrue(array_eq(expm_X, corr_expm_X, tol))

class TestUnitary(unittest.TestCase):
    def test_eig_mod_1(self):
        # Making sure eigenvalues of time_evo() have modulus 1 since it 
        # returns a unitary
        def check_if_eigs_mod_1(u):
            eig = np.linalg.eig(u)
            mag = np.abs(eig[0])
            self.assertTrue(array_eq(mag, np.ones(mag.shape), tol))

        for H in herms: 
            u = time_evo(H, np.random.uniform(0, 3))
            check_if_eigs_mod_1(u)

    def test_prod_identity(self):
        # U*U = I should be true 
        def adjoint_prod(u):
            prod = NDot(adjoint(u), u)
            inv_prod = NDot(u, adjoint(u))
            self.assertTrue(array_eq(prod, np.eye(prod.shape[0]), tol))
            self.assertTrue(array_eq(inv_prod, np.eye(prod.shape[0]), tol))

        for H in herms: 
            u = time_evo(H, np.random.uniform(0, 1))
            adjoint_prod(u)

# Check that time_evo() is continuous: t=3 * t=1 should equal t=4 for same H. 
class TestContinuous(unittest.TestCase):
    def test_continuous(self):
        for H in herms: 
            u_1 = time_evo(H, 1)
            u_3 = time_evo(H, 3)
            u_4 = time_evo(H, 4)
            self.assertTrue(array_eq(NDot(u_1, u_3), u_4, tol))

# TODO: Test prod_time_evo()

if __name__ == '__main__':
    unittest.main()
