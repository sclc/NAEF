""" 
Experiment Diary 2016-05-07
"""
import sys
from scipy import io
import numpy as np
from scipy.sparse.linalg import *
sys.path.append("../src/") 
from worker import Worker
from native_conjugate_gradient import NativeConjugateGradient
from native_conjugate_gradient import NativeBlockConjugateGradient
from gerschgorin_circle_theorem import GerschgorinCircleTheoremEigenvalueEstimator 
from chebyshev_polynomial import ChebyshevPolynomial
from chebyshev_basis_cacg import CBCG
from chebyshev_basis_cacg import BCBCG

class WorkerIterativeLinearSystemSolverCG_Exp_160507_A(Worker):
    """ Description: Experiment A
        Numerical Method: Naive Conjugate Gradient
        tol:
        max_iteration:
        matrix:
        Reference:
            1. 
    """
    def __init__(self, mat_path):
        """ """
        #print ("WorkerIterativeLinearSystemSolver works good")
        Worker.__init__(self)
        self._hist_list = []

        if mat_path == "":
            """ Need to generatre matrix """
            print("calling self._matrix_generation")
            #self._mat = self._matrix_generation()
        else:
            self._mat_coo = io.mmread(mat_path)
            self._mat = self._mat_coo.tocsr()
            self._mat_info = io.mminfo(mat_path)
            print("Done reading matrix {}, Row:{}, Col:{}".format( mat_path, self._mat.shape[0], self._mat.shape[1]))
            print("mminfo:{}".format(self._mat_info))
            #print("matrix:{}".format(type(self._mat)))
            #print("matrix:{}".format(dir(self._mat)))
            if self._mat.getformat() == "csr":
                print("Yeah, it is CSR")


    def _matrix_generator(self):
        """ generation of matrix """
        print("_matrix_generator")


    def _setup_testbed(self, block_size):
        """ this can considered as a basic experiment input descripting """
        self._B = np.random.random( ( self._mat.shape[0],block_size) )
        np.savetxt("/home/scl/tmp/rhs.csv",self._B, delimiter=",")
        #self._B = np.ones( ( self._mat.shape[0],6) )
        self._X = np.ones ( (self._mat.shape[1],block_size) )
        #self._X = np.zeros ( (self._mat.shape[1],1) ) 

    def _setup_numerical_algorithm(self,tol, maxiter, step_val):
        """ After a linear solver or other numerical methods loaded
            we need to setup the basic prarm for the algorithm
        """
        self._tol = tol
        self._maxiter = maxiter
        self._step_val = step_val

    def conduct_experiments(self, block_size, tol, maxiter, step_val):
        """ function to condution the experiment """
        print("to conduct the experient")
        self._setup_testbed(block_size)
        self._setup_numerical_algorithm(tol,maxiter,step_val)
        #self._debug_gerschgorin()
        #self._debug_chebyshev_polynomial_basis_generator()
        #print ("before:{}".format(np.inner(self._X[:,0], self._X[:,0])))
        self._debug_cbcg()
        #self._debug_bcg()
        #self._debug_bcbcg()
        print("Experiments done")

    def _debug_gerschgorin (self):
        eigen_estimator = GerschgorinCircleTheoremEigenvalueEstimator()
        max_eigen, min_eigen = eigen_estimator.csr_mat_extreme_eigenvalue_estimation(self._mat)
        print("max:{}, min:{}".format(max_eigen,min_eigen))

    def _debug_chebyshev_polynomial_basis_generator(self):
        step_val=5
        cheby_generator = ChebyshevPolynomial()
        eigen_estimator = GerschgorinCircleTheoremEigenvalueEstimator()
        max_eigen, min_eigen = eigen_estimator.csr_mat_extreme_eigenvalue_estimation(self._mat)
        print("max:{}, min:{}\n".format(max_eigen,min_eigen))
        res = cheby_generator.basis_generation_with_eigenvalue_shifting_and_scaling_single_vec(\
                self._mat, self._B, step_val, max_eigen, min_eigen)
        print(res)

    def _debug_cbcg(self):
        cbcg_solver_obj = CBCG() 
        self._final_x, self._final_r, self._residual_hist = \
               cbcg_solver_obj.cbcg_solver(self._mat, self._B, self._X, self._step_val, self._tol, self._maxiter)
        print(self._residual_hist)

    def _debug_bcg(self):
        bcg_solver_obj = NativeBlockConjugateGradient(self._mat, self._X, self._B, self._tol, self._maxiter)
        self._final_X, self._final_R, self._residual_hist = bcg_solver_obj.bcg_variant_one_run(0) 
        print(self._residual_hist)
        #bcg_solver_obj.bcg_variant_one_run(0) 

    def _debug_bcbcg(self):
        bcbcg_solver_obj = BCBCG() 
        self._final_X, self._final_R, self._residual_hist = \
               bcbcg_solver_obj.bcbcg_solver(self._mat, self._B, self._X, self._step_val, self._tol, self._maxiter,1)
        print(self._residual_hist)






def main ():
# main function for today's experiments 
    #mat_path = "/home/scl/MStore/nasa2146/nasa2146.mtx"
    mat_path = "/home/scl/MStore/crystm01/crystm01.mtx"
    #mat_path = "/home/scl/MStore/ex13/ex13.mtx"
    #mat_path = "/home/scl/MStore/LFAT5/LFAT5.mtx"
    block_size = 1
    tol = 1e-10
    maxiter = 3
    step_val =5 

    linear_system_solver_worker_test = WorkerIterativeLinearSystemSolverCG_Exp_160507_A(mat_path)
    linear_system_solver_worker_test.conduct_experiments(block_size,tol,maxiter, step_val)
    #linear_system_solver_worker_test.debug_NativeConjugateGradient()



if __name__ == "__main__":
    """ call main funtion for testing """
    main()
