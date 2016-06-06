""" 
Experiment Diary 2016-06-05 for ISC 2016
"""
import sys
import math
import matplotlib.pyplot as plt
from scipy import io
import numpy as np
from scipy.sparse.linalg import *
from scipy.sparse import *
sys.path.append("../src/") 
from worker import Worker
from native_conjugate_gradient import NativeConjugateGradient
from native_conjugate_gradient import NativeBlockConjugateGradient
from gerschgorin_circle_theorem import GerschgorinCircleTheoremEigenvalueEstimator 
from chebyshev_polynomial import ChebyshevPolynomial
from chebyshev_basis_cacg import CBCG
from legendre_basis_cacg import LBCG
from legendre_basis_cacg import BLBCG
from chebyshev_basis_cacg import BCBCG
from presenter import Presenter
from power_iteration import PowerIteration

class WorkerIterativeLinearSystemSolverCG_Exp_160605_isc16(Worker):
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
        #self._SB = np.random.random( ( self._mat.shape[0],1) )
        #self._BB = np.random.random( ( self._mat.shape[0],block_size) )
        #np.savetxt("/home/scl/tmp/rhs.csv",self._B, delimiter=",")
        #self._B = np.ones( ( self._mat.shape[0],6) )
        #self._SX = np.ones ( (self._mat.shape[1],1) )
        #self._BX = np.ones ( (self._mat.shape[1],block_size) )
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
        #self._cg_bcg_bcbcg_least_square_exp()
        #self._cg_bcg_blbcg_least_square_exp()
        #self._bcbcg_blbcg_least_square_exp()
        #self._bcbcg_blbcg_least_square_exp_b()
        #self._db_bcg_least_square()
        self._db_bcbcg_eigen_param()
        print("Experiments done")

    def _db_bcg_least_square (self):
        """ """
        m = 32
        self._BB_m  = np.random.random( ( self._mat.shape[0],m) )
        self._BX_m  = np.ones ( (self._mat.shape[1],m) )

        bcg_solver_obj = NativeBlockConjugateGradient(self._mat, self._BX_m, self._BB_m, self._tol, self._maxiter)
        self._final_X_bcg, self._final_R_bcg, self._residual_hist_bcg = bcg_solver_obj.bcg_variant_one_run(0)

        bcg_solver_obj = NativeBlockConjugateGradient(self._mat, self._BX_m, self._BB_m, self._tol, self._maxiter)
        self._final_X_bcg_lstsq, self._final_R_bcg_lstsq, self._residual_hist_bcg_lstsq = bcg_solver_obj.bcg_variant_lstsq_run(0)

        plot_worker = Presenter()
        residual_list = [self._residual_hist_bcg, self._residual_hist_bcg_lstsq]
        legend_list = ["bcg","bcg_lstsq"]
        color_list = ["r","k"]
        plot_worker.instant_plot_y_log10(residual_list, "test", "#iteration", "$\\mathbf{log_{10}\\frac{||x_1||}{||b_1||}}$", legend_list, color_list)

    def _lbcg_least_square_exp (self):
        """ """
        lbcg_solver_obj = LBCG()
        self._final_x_a, self._final_r_a, self._residual_hist_a = \
                 lbcg_solver_obj.lbcg_solver_least_square(self._mat, self._SB, self._SX, 8, self._tol, self._maxiter)
        self._final_x_b, self._final_r_b, self._residual_hist_b = \
                 lbcg_solver_obj.lbcg_solver_least_square(self._mat, self._SB, self._SX, 18, self._tol, self._maxiter)

        cbcg_solver_obj = CBCG() 
        self._final_x_c, self._final_r_c, self._residual_hist_c = \
               cbcg_solver_obj.cbcg_solver_least_square(self._mat, self._SB, self._SX, 8, self._tol, self._maxiter)
        self._final_x_d, self._final_r_d, self._residual_hist_d = \
               cbcg_solver_obj.cbcg_solver_least_square(self._mat, self._SB, self._SX, 18, self._tol, self._maxiter)

        plot_worker = Presenter()
        residual_list = [self._residual_hist_a, self._residual_hist_b, self._residual_hist_c, self._residual_hist_d ]
        legend_list = ["lbcg_lstsq_s8","lbcg_lstsq_s18" ,"cbcg_lstsq_s8", "cbcg_lstsq_s18" ]
        color_list = ["r","k", "b","y"]
        plot_worker.instant_plot_y_log10(residual_list, "test", "#iteration", "$\\mathbf{log_{10}\\frac{||x_1||}{||b_1||}}$", legend_list, color_list)

    def _cg_bcg_bcbcg_least_square_exp(self):
        """ """
        self._BB_1  = np.random.random( ( self._mat.shape[0],1) )
        self._BX_1  = np.ones ( (self._mat.shape[1],1) )
        self._BB_6  = np.random.random( ( self._mat.shape[0],6) )
        self._BX_6  = np.ones ( (self._mat.shape[1],6) )
        self._BB_12 = np.random.random( ( self._mat.shape[0],12) )
        self._BX_12 = np.ones ( (self._mat.shape[1],12) )

        #line 1
        bcg_solver_obj = NativeBlockConjugateGradient(self._mat, self._BX_1, self._BB_1, self._tol, self._maxiter)
        self._final_X_cg, self._final_R_cg, self._residual_hist_cg = bcg_solver_obj.bcg_variant_lstsq_run(0)

        #line 2
        bcg_solver_obj = NativeBlockConjugateGradient(self._mat, self._BX_12, self._BB_12, self._tol, self._maxiter)
        self._final_X_bcg_m12, self._final_R_bcg_m12, self._residual_hist_bcg_m12 = bcg_solver_obj.bcg_variant_lstsq_run(0)

        #line 3
        bcbcg_solver_obj = BCBCG()
        self._final_x_bcbcg_m1s2, self._final_r_bcbcg_m1s2, self._residual_hist_bcbcg_m1s2 = \
                bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB_1, self._BX_1, 2, self._tol, self._maxiter, 0)

        #line 4
        #bcbcg_solver_obj = BCBCG()
        #self._final_x_bcbcg_m1s6, self._final_r_bcbcg_m1s6, self._residual_hist_bcbcg_m1s6 = \
        #        bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB_1, self._BX_1, 6, self._tol, self._maxiter, 0)
        bcbcg_solver_obj = BCBCG()
        self._final_x_bcbcg_m1s6, self._final_r_bcbcg_m1s6, self._residual_hist_bcbcg_m1s6 = \
                bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB_1, self._BX_1, 8, self._tol, self._maxiter, 0)

        #line 5
        bcbcg_solver_obj = BCBCG()
        self._final_x_bcbcg_m6s2, self._final_r_bcbcg_m6s2, self._residual_hist_bcbcg_m6s2 = \
                bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB_6, self._BX_6, 2, self._tol, self._maxiter, 0)

        #line 6
        #bcbcg_solver_obj = BCBCG()
        #self._final_x_bcbcg_m6s6, self._final_r_bcbcg_m6s6, self._residual_hist_bcbcg_m6s6 = \
        #        bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB_6, self._BX_6, 6, self._tol, self._maxiter, 0)
        bcbcg_solver_obj = BCBCG()
        self._final_x_bcbcg_m6s6, self._final_r_bcbcg_m6s6, self._residual_hist_bcbcg_m6s6 = \
                bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB_6, self._BX_6, 8, self._tol, self._maxiter, 0)

        plot_worker = Presenter()
        residual_list = [self._residual_hist_cg, self._residual_hist_bcg_m12,  \
                         self._residual_hist_bcbcg_m1s2, self._residual_hist_bcbcg_m1s6, \
                         self._residual_hist_bcbcg_m6s2, self._residual_hist_bcbcg_m6s6 ]

        legend_list = ["cg","bcg_m12", "bcbcg_m1s2", "bcbcg_m1s6", "bcbcg_m6s2", "bcbcg_m6s6"]
        color_list = ["r","k","b","y","m","g"]
        plot_worker.instant_plot_y_log10(residual_list, "test", "#iteration", "$\\mathbf{log_{10}\\frac{||x_1||}{||b_1||}}$", legend_list, color_list)


    def _cg_bcg_blbcg_least_square_exp(self):
        """ """
        self._BB_1  = np.random.random( ( self._mat.shape[0],1) )
        self._BX_1  = np.ones ( (self._mat.shape[1],1) )
        self._BB_6  = np.random.random( ( self._mat.shape[0],6) )
        self._BX_6  = np.ones ( (self._mat.shape[1],6) )
        self._BB_12 = np.random.random( ( self._mat.shape[0],12) )
        self._BX_12 = np.ones ( (self._mat.shape[1],12) )

        #line 1
        bcg_solver_obj = NativeBlockConjugateGradient(self._mat, self._BX_1, self._BB_1, self._tol, self._maxiter)
        self._final_X_cg, self._final_R_cg, self._residual_hist_cg = bcg_solver_obj.bcg_variant_lstsq_run(0)

        #line 2
        bcg_solver_obj = NativeBlockConjugateGradient(self._mat, self._BX_12, self._BB_12, self._tol, self._maxiter)
        self._final_X_bcg_m12, self._final_R_bcg_m12, self._residual_hist_bcg_m12 = bcg_solver_obj.bcg_variant_lstsq_run(0)

        #line 3
        blbcg_solver_obj = BLBCG()
        self._final_x_blbcg_m1s2, self._final_r_blbcg_m1s2, self._residual_hist_blbcg_m1s2 = \
                blbcg_solver_obj.blbcg_solver_least_square(self._mat, self._BB_1, self._BX_1, 2, self._tol, self._maxiter, 0)

        #line 4
        blbcg_solver_obj = BLBCG()
        self._final_x_blbcg_m1s6, self._final_r_blbcg_m1s6, self._residual_hist_blbcg_m1s6 = \
                blbcg_solver_obj.blbcg_solver_least_square(self._mat, self._BB_1, self._BX_1, 6, self._tol, self._maxiter, 0)

        #line 5
        blbcg_solver_obj = BLBCG()
        self._final_x_blbcg_m6s2, self._final_r_blbcg_m6s2, self._residual_hist_blbcg_m6s2 = \
                blbcg_solver_obj.blbcg_solver_least_square(self._mat, self._BB_6, self._BX_6, 2, self._tol, self._maxiter, 0)

        #line 6
        blbcg_solver_obj = BLBCG()
        self._final_x_blbcg_m6s6, self._final_r_blbcg_m6s6, self._residual_hist_blbcg_m6s6 = \
                blbcg_solver_obj.blbcg_solver_least_square(self._mat, self._BB_6, self._BX_6, 6, self._tol, self._maxiter, 0)

        plot_worker = Presenter()
        residual_list = [self._residual_hist_cg, self._residual_hist_bcg_m12,  \
                         self._residual_hist_blbcg_m1s2, self._residual_hist_blbcg_m1s6, \
                         self._residual_hist_blbcg_m6s2, self._residual_hist_blbcg_m6s6 ]

        legend_list = ["cg","bcg_m12", "blbcg_m1s2", "blbcg_m1s6", "blbcg_m6s2", "blbcg_m6s6"]
        color_list = ["r","k","b","y","m","g"]
        plot_worker.instant_plot_y_log10(residual_list, "test", "#iteration", "$\\mathbf{log_{10}\\frac{||x_1||}{||b_1||}}$", legend_list, color_list)

    def _bcbcg_blbcg_least_square_exp(self):
        """ """
        self._BB_1  = np.random.random( ( self._mat.shape[0],1) )
        self._BX_1  = np.ones ( (self._mat.shape[1],1) )
        self._BB_6  = np.random.random( ( self._mat.shape[0],6) )
        self._BX_6  = np.ones ( (self._mat.shape[1],6) )

        #line 1
        blbcg_solver_obj = BLBCG()
        self._final_x_blbcg_m1s2, self._final_r_blbcg_m1s2, self._residual_hist_blbcg_m1s2 = \
                blbcg_solver_obj.blbcg_solver_least_square(self._mat, self._BB_1, self._BX_1, 2, self._tol, self._maxiter, 0)

        #line 2
        blbcg_solver_obj = BLBCG()
        self._final_x_blbcg_m1s6, self._final_r_blbcg_m1s6, self._residual_hist_blbcg_m1s6 = \
                blbcg_solver_obj.blbcg_solver_least_square(self._mat, self._BB_1, self._BX_1, 6, self._tol, self._maxiter, 0)

        #line 3
        blbcg_solver_obj = BLBCG()
        self._final_x_blbcg_m6s2, self._final_r_blbcg_m6s2, self._residual_hist_blbcg_m6s2 = \
                blbcg_solver_obj.blbcg_solver_least_square(self._mat, self._BB_6, self._BX_6, 2, self._tol, self._maxiter, 0)

        #line 4
        blbcg_solver_obj = BLBCG()
        self._final_x_blbcg_m6s6, self._final_r_blbcg_m6s6, self._residual_hist_blbcg_m6s6 = \
                blbcg_solver_obj.blbcg_solver_least_square(self._mat, self._BB_6, self._BX_6, 6, self._tol, self._maxiter, 0)

        #line 5
        bcbcg_solver_obj = BCBCG()
        self._final_x_bcbcg_m1s2, self._final_r_bcbcg_m1s2, self._residual_hist_bcbcg_m1s2 = \
                bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB_1, self._BX_1, 2, self._tol, self._maxiter, 0)

        #line 6
        bcbcg_solver_obj = BCBCG()
        self._final_x_bcbcg_m1s6, self._final_r_bcbcg_m1s6, self._residual_hist_bcbcg_m1s6 = \
                bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB_1, self._BX_1, 6, self._tol, self._maxiter, 0)

        #line 7
        bcbcg_solver_obj = BCBCG()
        self._final_x_bcbcg_m6s2, self._final_r_bcbcg_m6s2, self._residual_hist_bcbcg_m6s2 = \
                bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB_6, self._BX_6, 2, self._tol, self._maxiter, 0)

        #line 8
        bcbcg_solver_obj = BCBCG()
        self._final_x_bcbcg_m6s6, self._final_r_bcbcg_m6s6, self._residual_hist_bcbcg_m6s6 = \
                bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB_6, self._BX_6, 6, self._tol, self._maxiter, 0)

        plot_worker = Presenter()
        residual_list = [self._residual_hist_blbcg_m1s2, self._residual_hist_blbcg_m1s6, \
                         self._residual_hist_blbcg_m6s2, self._residual_hist_blbcg_m6s6, \
                         self._residual_hist_bcbcg_m1s2, self._residual_hist_bcbcg_m1s6, \
                         self._residual_hist_bcbcg_m6s2, self._residual_hist_bcbcg_m6s6 ]

        legend_list = ["blbcg_m1s2", "blbcg_m1s6", "blbcg_m6s2", "blbcg_m6s6", "bcbcg_m1s2", "bcbcg_m1s6", "bcbcg_m6s2", "bcbcg_m6s6"]
        color_list = ["r","k","b","y","m","g", "m", "0.5"]
        #plot_worker.instant_plot_y_log10(residual_list, "Chem97ZtZ", "#iteration", "$\\frac{||x_1||}{||b_1||}$", legend_list, color_list)
        plot_worker.instant_plot_y_log10(residual_list, "test", "#iteration", "$\\mathbf{log_{10}\\frac{||x_1||}{||b_1||}}$", legend_list, color_list)

    def _bcbcg_blbcg_least_square_exp_b(self):
        """ """
        self._BB_6  = np.random.random( ( self._mat.shape[0],6) )
        self._BX_6  = np.ones ( (self._mat.shape[1],6) )


        #line 1
        blbcg_solver_obj = BLBCG()
        self._final_x_blbcg_m6s6, self._final_r_blbcg_m6s6, self._residual_hist_blbcg_m6s6 = \
                blbcg_solver_obj.blbcg_solver_least_square(self._mat, self._BB_6, self._BX_6, 6, self._tol, self._maxiter, 0)
        #line 2
        blbcg_solver_obj = BLBCG()
        self._final_x_blbcg_m6s12, self._final_r_blbcg_m6s12, self._residual_hist_blbcg_m6s12 = \
                blbcg_solver_obj.blbcg_solver_least_square(self._mat, self._BB_6, self._BX_6, 12, self._tol, self._maxiter, 0)


        #line 3
        bcbcg_solver_obj = BCBCG()
        self._final_x_bcbcg_m6s6, self._final_r_bcbcg_m6s6, self._residual_hist_bcbcg_m6s6 = \
                bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB_6, self._BX_6, 6, self._tol, self._maxiter, 0)
        #line 4
        bcbcg_solver_obj = BCBCG()
        self._final_x_bcbcg_m6s12, self._final_r_bcbcg_m6s12, self._residual_hist_bcbcg_m6s12 = \
                bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB_6, self._BX_6, 12, self._tol, self._maxiter, 0)

        plot_worker = Presenter()
        residual_list = [self._residual_hist_blbcg_m6s6, self._residual_hist_blbcg_m6s12, \
                         self._residual_hist_bcbcg_m6s6, self._residual_hist_bcbcg_m6s12 ]

        legend_list = ["blbcg_m6s6", "blbcg_m6s12", "bcbcg_m6s6", "bcbcg_m6s12"]
        color_list = ["r","k","b","y"]
        #plot_worker.instant_plot_y_log10(residual_list, "Chem97ZtZ", "#iteration", "$\\frac{||x_1||}{||b_1||}$", legend_list, color_list)
        plot_worker.instant_plot_y_log10(residual_list, "test", "#iteration", "$\\mathbf{log_{10}\\frac{||x_1||}{||b_1||}}$", legend_list, color_list)

    def _db_bcbcg_eigen_param(self):
        """ """
        self._BB_6  = np.random.random( ( self._mat.shape[0],6) )
        self._BX_6  = np.ones ( (self._mat.shape[1],6) )

        gerschgorin_estimator = GerschgorinCircleTheoremEigenvalueEstimator()
        max_eigenvalue, min_eigenvalue = gerschgorin_estimator.csr_mat_extreme_eigenvalue_estimation(self._mat)
        print("################", "max:",max_eigenvalue, " , min:", min_eigenvalue)

        bcbcg_solver_obj = BCBCG()
        self._final_x_bcbcg_eigenparam_m6s6, self._final_r_bcbcg_eigenparam_m6s6, self._residual_hist_bcbcg_eigenparam_m6s6 = \
                bcbcg_solver_obj.bcbcg_solver_least_square_eigen_param(self._mat, self._BB_6, self._BX_6, 6, self._tol, self._maxiter, 0, max_eigenvalue, min_eigenvalue)

        bcbcg_solver_obj = BCBCG()
        self._final_x_bcbcg_m6s6, self._final_r_bcbcg_m6s6, self._residual_hist_bcbcg_m6s6 = \
                bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB_6, self._BX_6, 6, self._tol, self._maxiter, 0)

        plot_worker = Presenter()
        residual_list = [self._residual_hist_bcbcg_eigenparam_m6s6, self._residual_hist_bcbcg_m6s6]

        legend_list = ["bcbcg_eigenparam_m6s6", "bcbcg_m6s6"]
        color_list = ["r","k"]
        plot_worker.instant_plot_y_log10(residual_list, "test", "#iteration", "$\\mathbf{log_{10}\\frac{||x_1||}{||b_1||}}$", legend_list, color_list)


def main ():
# main function for today's experiments 
    #bad
    #mat_path = "/home/scl/MStore/vanbody/vanbody.mtx"
    #mat_path = "/home/scl/MStore/olafu/olafu.mtx"
    #mat_path = "/home/scl/MStore/raefsky4/raefsky4.mtx"
    #mat_path = "/home/scl/MStore/smt/smt.mtx"
    #mat_path = "/home/scl/MStore/bcsstk36/bcsstk36.mtx"
    #mat_path = "/home/scl/MStore/pdb1HYS/pdb1HYS.mtx"
    #mat_path = "/home/scl/MStore/ship_001/ship_001.mtx"

    # not so good
    #mat_path = "/home/scl/MStore/Dubcova1/Dubcova1.mtx"
    #mat_path = "/home/scl/MStore/bcsstk17/bcsstk17.mtx"
    #mat_path = "/home/scl/MStore/wathen100/wathen100.mtx"

    #mat_path = "/home/scl/MStore/nasa2146/nasa2146.mtx"
    #mat_path = "/home/scl/MStore/crystm01/crystm01.mtx"
    #mat_path = "/home/scl/MStore/ex13/ex13.mtx"
    #mat_path = "/home/scl/MStore/LFAT5/LFAT5.mtx"

    #good
    #mat_path = "/home/scl/MStore/bodyy6/bodyy6.mtx"
    #mat_path = "/home/scl/MStore/crystm02/crystm02.mtx"

    #isc16
    mat_path = "/home/scl/MStore/Chem97ZtZ/Chem97ZtZ.mtx"
    #mat_path = "/home/scl/MStore/bodyy6/bodyy6.mtx"
    #mat_path = "/home/scl/MStore/wathen100/wathen100.mtx"

    block_size = 4 
    tol = 1e-12
    maxiter = 1500
    step_val =64

    linear_system_solver_worker_test = WorkerIterativeLinearSystemSolverCG_Exp_160605_isc16(mat_path)
    linear_system_solver_worker_test.conduct_experiments(block_size,tol,maxiter, step_val)
    #linear_system_solver_worker_test.chebyshev_poly_exp_a(0,6)
    #linear_system_solver_worker_test.legendre_poly_exp_a(0,6)
    #linear_system_solver_worker_test.debug_NativeConjugateGradient()



if __name__ == "__main__":
    """ call main funtion for testing """
    main()
