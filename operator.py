import numpy as np
import unittest

from utils import * 

class FermionicOperator: 
    """
    A base class for any type of operator. You shouldn't be instantiating this 
    directly. 

    Attributes: 
        type : str 
            The type of operator: 'creation' or 'annihilation'.
        spin : str 
            Spin of the particle the operator acts on. 
        site : int < num_sites
            The index of the site this operator is acting on. 
            We'll calculate which 
            exact qubit it acts on based on site, spin, and num_sites. 
            For systems with spin, we assign two qubits to each site: on site j, 
            the 'up' spin acts on qubit j, and the 'down' spin acts on qubit 
            num_sites + j 
        num_sites : int 
            The number of sites in a system. 
        dim : int 
            The number of qubits used. 
        JW : 2**dim x 2**dim matrix 
            Jordan-Wigner transformation of operator. The lowest qubit is on 
            the *LEFT* ie |10> = [0, 0, 1, 0]. 
        JW_str : str 
            String representation of Jordan-Wigner encoding. 
    
    Methods: 
        TODO: 
    """
    
    def __init__(self, _type: str, spin: str, site: int, num_sites: int):
        """
        Parameters: 
            _type : str 
                One of ['creation', 'annihilation'].
            spin : str 
                One of ['up', 'down']. 
            site : int 
            num_sites : int 
        """

        types = ['creation', 'annihilation']
        if _type not in types: 
            raise ValueError("_type must be either 'creation' or \
                             'annihilation'!")
        spins = ['up', 'down']
        if spin not in spins: 
            raise ValueError("spin must be either 'up' or 'down'!")

        self.type = _type 
        self.spin = spin 
        self.site = site
        self.num_sites = num_sites 

        if self.spin == 'up': 
            self.qubit = self.site 
        elif self.spin == 'down': 
            self.qubit = self.num_sites + self.site
        else: 
            raise ValueError("spin value must be 'up' or 'down'!")

        self.dim = 2 * num_sites 
        self.JW, self.JW_str = self._gen_JW_mapping()

    def _gen_JW_mapping(self): 
        """
        Generate a Jordan-Wigner representation of a fermionic operator. 

        We let |0> represent no electron in a spin-orbital, and we let |1> 
        represent an electron existing in a spin-orbital. 
        This allows us to define the creation operator as (X-iY)/2 and the 
        annihilation operator as (X+iY)/2. 
        The main problem with this is that the creation and annihilation 
        operators don't anti-commute. To fix this, we prepend the operator 
        with Pauli-Z tensors. Because the Pauli operators anti-commute, this 
        preserves our desired anti-commutation relation. 
        """

        res = 1 
        str_repr = ''

        # Notice order of tensor product: gate on 0th qubit is "leftmost"
        for i in range(self.dim):
            if i < self.qubit: 
                res = NKron(res, Z)
                str_repr += 'Z'
            elif i == self.qubit and self.type == 'creation': 
                res = NKron(res, (X - 1j * Y) / 2)
                str_repr += '-'
            elif i == self.qubit and self.type == 'annihilation':
                res = NKron(res, (X + 1j * Y) / 2)
                str_repr += '+'
            elif i > self.qubit: 
                res = NKron(res, I)
                str_repr += 'I'
            else: 
                raise ValueError("Something's wrong with the JW encoding!")
        return res, str_repr

class CreationOperator(FermionicOperator):
    """Shortcut for a creation operator."""

    def __init__(self, spin: str, site: int, num_sites: int):
        super(CreationOperator, self).__init__('creation', spin, site, 
                                               num_sites)

class AnnihilationOperator(FermionicOperator):
    """Shortcut for an annihilation operator."""

    def __init__(self, spin: str, site: int, num_sites: int):
        super(AnnihilationOperator, self).__init__('annihilation', spin, site, 
                                               num_sites)

# TESTS

class TestCreationOperator(unittest.TestCase): 
    def test_init(self): 
        c_0 = CreationOperator('up', 0, 4)
        self.assertEqual(c_0.type, 'creation')
        c_1 = CreationOperator('up', 1, 3)
        self.assertEqual(c_1.type, 'creation')
        c_2 = CreationOperator('down', 2, 4)
        self.assertEqual(c_2.type, 'creation')
        c_4 = CreationOperator('up', 4, 5)
        self.assertEqual(c_4.type, 'creation')

    def test_attributes(self):
        c_2 = CreationOperator('down', 2, 4)
        self.assertEqual(c_2.spin, 'down')
        self.assertEqual(c_2.site, 2)
        self.assertEqual(c_2.num_sites, 4)
        self.assertEqual(c_2.dim, 8)
        self.assertEqual(c_2.JW.shape, (256, 256))
        self.assertEqual(c_2.JW_str, 'ZZZZZZ-I')

class TestAnnihilationOperator(unittest.TestCase):
    def test_init(self):
        a_1 = AnnihilationOperator('down', 3, 6)
        self.assertEqual(a_1.type, 'annihilation')

class TestAntiCommutationRelations(unittest.TestCase):
    def test_same_site_same_spin_one_dim(self):
        c_0 = CreationOperator('up', 0, 1)
        a_0 = AnnihilationOperator('up', 0, 1)
        anti_comm = NDot(c_0.JW, a_0.JW) + NDot(a_0.JW, c_0.JW)
        i_4 = np.eye(2**c_0.dim)
        self.assertTrue(array_eq(anti_comm, i_4, 0.005))
    
    def test_same_site_diff_spin_one_dim(self):
        c_0 = CreationOperator('up', 0, 1)
        a_0 = AnnihilationOperator('down', 0, 1)
        anti_comm = NDot(c_0.JW, a_0.JW) + NDot(a_0.JW, c_0.JW)
        z_4 = np.zeros((2**c_0.dim, 2**c_0.dim))
        self.assertTrue(array_eq(anti_comm, z_4, 0.005))

    def test_diff_site_same_spin_two_dim(self):
        c_0 = CreationOperator('down', 0, 2)
        a_1 = AnnihilationOperator('down', 1, 2)
        anti_comm = NDot(c_0.JW, a_1.JW) + NDot(a_1.JW, c_0.JW)
        z_16 = np.zeros((2**c_0.dim, 2**c_0.dim))
        self.assertTrue(array_eq(anti_comm, z_16, 0.005))
        
    def test_diff_site_diff_spin_two_dim(self):
        c_0 = CreationOperator('up', 0, 2)
        a_1 = AnnihilationOperator('down', 1, 2)
        anti_comm = NDot(c_0.JW, a_1.JW) + NDot(a_1.JW, c_0.JW)
        z_16 = np.zeros((2**c_0.dim, 2**c_0.dim))
        self.assertTrue(array_eq(anti_comm, z_16, 0.005))

    # TODO: more complex tests for anti-commutator 

    def test_creation_anti(self):
        c_0 = CreationOperator('up', 0, 3)
        c_2 = CreationOperator('down', 2, 3)
        anti_comm = NDot(c_0.JW, c_2.JW) + NDot(c_2.JW, c_0.JW)
        z_64 = np.zeros((2**c_0.dim, 2**c_0.dim))
        self.assertTrue(array_eq(anti_comm, z_64, 0.005))

    def test_annihilation_anti(self):
        a_0 = AnnihilationOperator('down', 0, 2)
        a_1 = AnnihilationOperator('down', 1, 2)
        anti_comm = NDot(a_0.JW, a_1.JW) + NDot(a_1.JW, a_0.JW)
        z_16 = np.zeros((2**a_0.dim, 2**a_0.dim))
        self.assertTrue(array_eq(anti_comm, z_16, 0.005))

if __name__ == '__main__':
    unittest.main()
