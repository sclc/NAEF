""" """
import numpy as np
from scipy.sparse import linalg

class LegendrePolynomial():
    """ Chebyshev Polynomial class"""
    def __init__(self):
        pass

    def basis_generation_with_eigenvalue_shifting_and_scaling_single_vec(self, mat, vec, step_val, max_eigenval, min_eigenval):
        """ step_val >=1"""
        assert step_val>=1, "Need a larger step_val"

        legendre_basis = np.zeros((mat.shape[0], step_val))
        matvec = linalg.aslinearoperator(mat)

        s_alpha = 2.0 / (max_eigenval - min_eigenval)
        s_beta  = - (max_eigenval + min_eigenval) / (max_eigenval - min_eigenval)

        for sIdx in range(1,step_val+1):
            degree = sIdx-1
            if degree == 0:
                legendre_basis[:,degree] = vec[:,0] 

            elif degree == 1:
                legendre_basis[:,degree] = (s_alpha * matvec(vec) + s_beta * vec)[:,0] 
            else:
                fl_degree = float(degree)
                degree_reciprocal = np.reciprocal(fl_degree)
                legendre_basis[:,degree] = degree_reciprocal * (2.*(fl_degree-1.) + 1.) * s_alpha * matvec(legendre_basis[:, degree-1]) \
                        + degree_reciprocal * (2.*(fl_degree-1.) + 1.) * s_beta * legendre_basis[:,degree-1] \
                        - degree_reciprocal *(fl_degree-1.) * legendre_basis[:,degree-2]

        return legendre_basis


    def basis_generation_with_eigenvalue_shifting_and_scaling_block_vecs(self, mat, blockvec, step_val, max_eigenval, min_eigenval):
        """ step_val >=1"""
        assert step_val>=1, "Need a larger step_val"

        block_size = blockvec.shape[1]
        #print("block_size ", block_size)
        legendre_basis = np.zeros((mat.shape[0], block_size*step_val))
        #print("basis ", chebyshev_basis.shape[0], " , ",chebyshev_basis.shape[1])
        op_linalg_A = linalg.aslinearoperator(mat)

        s_alpha = 2.0 / (max_eigenval - min_eigenval)
        s_beta  = - (max_eigenval + min_eigenval) / (max_eigenval - min_eigenval)

        for sIdx in range(1,step_val+1):
            degree = sIdx-1
            if degree == 0:
                legendre_basis[:,0:block_size] = blockvec[:,0:block_size]

            elif degree == 1:
                legendre_basis[:,block_size:2*block_size] = (s_alpha * op_linalg_A.matmat(blockvec) + s_beta * blockvec)[:,0:block_size]
            else:
                fl_degree = float(degree)
                degree_reciprocal = np.reciprocal(fl_degree)
                legendre_basis[:,degree*block_size:(degree+1)*block_size] = degree_reciprocal * (2.*(fl_degree-1.) + 1.) * s_alpha * op_linalg_A.matmat(legendre_basis[:, (degree-1)*block_size:degree*block_size]) \
                        + degree_reciprocal * (2.*(fl_degree-1.) + 1.) * s_beta * legendre_basis[:,(degree-1)*block_size:degree*block_size] \
                        - degree_reciprocal *(fl_degree-1.) * legendre_basis[:,(degree-2)*block_size:(degree-1)*block_size]

        return legendre_basis




