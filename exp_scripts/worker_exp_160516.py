""" 
Experiment Diary 2016-05-16
"""
import sys
import math
import matplotlib.pyplot as plt
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
from presenter import Presenter
from power_iteration import PowerIteration

class WorkerIterativeLinearSystemSolverCG_Exp_160516_A(Worker):
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
        #print ("before:{}".format(np.inner(self._X[:,0], self._X[:,0])))
        #self._bcbcg_exp()
        #self._db_presenter_a()
        self._db_power_iteration()
        print("Experiments done")

    def _bcbcg_exp(self):
        bcbcg_solver_obj = BCBCG() 
        step_val_a = 3
        step_val_b = 5
        self._final_X_a, self._final_R_a, self._residual_hist_a = \
               bcbcg_solver_obj.bcbcg_solver(self._mat, self._B, self._X, step_val_a, self._tol, self._maxiter,0)
        self._final_X_b, self._final_R_b, self._residual_hist_b = \
               bcbcg_solver_obj.bcbcg_solver(self._mat, self._B, self._X, step_val_b, self._tol, self._maxiter,0)

    def _db_presenter_a(self):
        plot_worker = Presenter()
        residual_list = [self._residual_hist_a]
        residual_list.append(self._residual_hist_b)
        legend_list = ["bcbcg_s3", "bcbcg_s5"]
        color_list = ["r", "k"]
        # latex style notation
        #plot_worker.instant_plot_y_log10(residual_list, "crystm01 $x_1$")
        #plot_worker.instant_plot_y_log10(residual_list, "crystm01", "#iteration", "$\\frac{||x_1||}{||b_1||}$", legend_list, color_list)
        plot_worker.instant_plot_y(residual_list, "crystm01", "#iteration", "$\\frac{||x_1||}{||b_1||}$", legend_list, color_list)
    
    def _db_power_iteration(self):
        pi_worker = PowerIteration();
        tol = 1e-4
        maxiters = 1000
        self._v_eigen, self._lambda_eigen, self._lambda_eigen_list = pi_worker.naive_power_iteration (self._mat, self._X, maxiters, tol)
        print ("lambda ", self._lambda_eigen)
        plot_worker = Presenter()
        ratio_list = [self._lambda_eigen_list]
        legend_list = ["naive power iteration"]
        color_list = ["k"]
        plot_worker.instant_plot_y_log10(ratio_list, "crystm01" , "#iteration", "$\\frac{\\lambda_{old} - \\lambda_{new}}{\\lambda_i{old} }$", legend_list, color_list)



def main ():
# main function for today's experiments 
    #mat_path = "/home/scl/MStore/nasa2146/nasa2146.mtx"
    #mat_path = "/home/scl/MStore/crystm01/crystm01.mtx"
    mat_path = "/home/scl/MStore/ex13/ex13.mtx"
    #mat_path = "/home/scl/MStore/LFAT5/LFAT5.mtx"
    block_size = 1
    tol = 1e-10
    maxiter = 1000
    step_val =2 

    linear_system_solver_worker_test = WorkerIterativeLinearSystemSolverCG_Exp_160516_A(mat_path)
    linear_system_solver_worker_test.conduct_experiments(block_size,tol,maxiter, step_val)
    #linear_system_solver_worker_test.debug_NativeConjugateGradient()



if __name__ == "__main__":
    """ call main funtion for testing """
    main()
