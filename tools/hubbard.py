import numpy as np
import unittest 

from utils import * 
from operators import * 
from evolution import * 

from typing import Callable

class Hubbard2D: 
    """
    The Hubbard Hamiltonian. 
    We define a 2D lattice of fermion sites. Each site on the lattice 
    has two spin-orbitals. This class defines the properties of that site, 
    and then uses those properties to compute the Hamiltonian. 

    Attributes: 
        

    Methods: 
        djfk
    """

    def __init__(self, N_X: int, N_Y: int, 
                 t: float, u: float, mu: float=None, t_ij=None): 
        """
        Most of the computation is done on initialization so that we ensure we 
        only have to do it once. 

        Parameters: 
            N_X : int 
                Width of the 2D lattice. 
            N_Y : int 
                Length of the 2D lattice. 
            t : float 
                Common kinetic energy multiple. Every adjacent movement of a 
                fermion is multiplied by this kinetic energy coefficient. 
                -t is applied. 
            t_ij : [Optional] (N_Y, N_X) array 
                Specific kinetic energy multiplies. For a simple model, this 
                can be omitted for a default value of t_ij[i][j] = 1. 
            u : float 
                Common interaction energy term. This is only applied if BOTH 
                spin-orbitals of a site are occupied. 
                u is applied. 
            mu : float 
                Common chemical potential energy term. 
                -mu is applied. 
        """
        self.N_X = N_X 
        self.N_Y = N_Y 

        self.t = t
        self.t_ij = t_ij or np.ones((N_Y, N_X), dtype='complex128')
        self.u = u 
        self.mu = mu 

        self.lattice = [[ { 
            'i': i, 
            'j': j, 
            'c-up': CreationOperator('up', self.N_Y*i+j, self.N_Y*self.N_X), 
            'c-down': CreationOperator('down', self.N_Y*i+j, self.N_Y*self.N_X), 
            'a-up': AnnihilationOperator('up', self.N_Y*i+j, self.N_Y*self.N_X), 
            'a-down': AnnihilationOperator('down', self.N_Y*i+j, self.N_Y*
                                           self.N_X) 
        } for j in range(self.N_X) ] for i in range(self.N_Y) ]

        self.Hs = [ 
            # Get horizontal, even hopping terms
            self._sum_pair_operators(lambda i, j: 1-j%2, lambda i, j: (i, j+1)), 
            # Get horizontal, odd hopping terms 
            self._sum_pair_operators(lambda i, j: j%2, lambda i, j: (i, j+1)), 
            # Get vertical, even hopping terms 
            self._sum_pair_operators(lambda i, j: 1-i%2, lambda i, j: (i+1, j)), 
            # Get vertical, odd hopping terms 
            self._sum_pair_operators(lambda i, j: i%2, lambda i, j: (i+1, j)), 
            # Get interacting U term 
            self._sum_interacting_term()
        ]

        self.H = sum(self.Hs)

    def _sum_pair_operators(self, filter_ij: Callable[[int, int], bool], 
                            map_to_next_ij: Callable[[int, int], list]):
        """
        This loops over all sites self.lattice[i][j]. For each (i, j), it 
        evaluates filter_ij(i, j). If False, continues to next iteration. 
        If True, it calculates another pair of indices using 
        map_to_next_ij(i, j) -> (k, l). 
        Call site (i, j) -> n and site (k, l) -> m. 
        Then, adds 4 terms to the result: 
            c_nu * a_mu 
            c_mu * a_nu 
            c_nd * a_md 
            c_md * a_nd
        Returns the sum once done iterating. 
        """

        result = 0
        test = 0
        for i in range(self.N_Y):
            for j in range(self.N_X):
                if filter_ij(i, j): 
                    # Get indices for next site
                    k, l = map_to_next_ij(i, j)
                    # Make sure next site isn't out of bounds
                    if k >= self.N_Y or l >= self.N_X: continue

                    site_n = self.lattice[i][j]
                    site_m = self.lattice[k][l]

                    result += NDot(site_n['c-up'].JW, site_m['a-up'].JW)
                    result += NDot(site_m['c-up'].JW, site_n['a-up'].JW)
                    result += NDot(site_n['c-down'].JW, site_m['a-down'].JW)
                    result += NDot(site_m['c-down'].JW, site_n['a-down'].JW)
                    test += 1
        print(test)
        return -self.t * result 

    def _sum_interacting_term(self):
        """Sums up the interaction term."""
        result = 0 
        test = 0
        for i in range(self.N_Y): 
            for j in range(self.N_X): 
                site = self.lattice[i][j]
                result += NDot(site['c-up'].JW, site['a-up'].JW, 
                               site['c-down'].JW, site['a-down'].JW)
                test += 1
        print(test)
        return self.u * result 

if __name__ == '__main__':
    N_X = 2 
    N_Y = 2 
    t = 1 
    u = 2
    mu = 0 
    x = Hubbard2D(N_X, N_Y, t, u, mu)
    print(np.argmin(x.H[10]))
    print(min(x.H[10]))
    print(x.H.shape)
    print(np.linalg.eigh(x.H)[0][0])

