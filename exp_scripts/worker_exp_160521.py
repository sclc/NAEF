""" 
Experiment Diary 2016-05-18
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
from legendre_basis_cacg import LBCG
from legendre_basis_cacg import BLBCG
from chebyshev_basis_cacg import BCBCG
from presenter import Presenter
from power_iteration import PowerIteration

class WorkerIterativeLinearSystemSolverCG_Exp_160521(Worker):
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
        self._SB = np.random.random( ( self._mat.shape[0],1) )
        self._BB = np.random.random( ( self._mat.shape[0],block_size) )
        #np.savetxt("/home/scl/tmp/rhs.csv",self._B, delimiter=",")
        #self._B = np.ones( ( self._mat.shape[0],6) )
        self._SX = np.ones ( (self._mat.shape[1],1) )
        self._BX = np.ones ( (self._mat.shape[1],block_size) )
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
        #self._db_power_iteration()
        #self._db_lbcg_exp()
        #self._db_blbcg_exp()
        #self. _numpy_lstsq_test()
        #self._db_cbcg_lstsq()
        #self._db_bcbcg_lstsq()
        #self._db_lbcg_least_square_exp()
        #self._db_blbcg_least_square_exp()
        self._blbcg_least_square_exp()
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


    def legendre_poly_exp_a(self, order_lo, order_hi):
        """ """

        x= np.linspace(-1.1,1.1,41)
        order_controller = np.zeros(order_hi+1)
        y_list = []

        plot_worker = Presenter()
        legend_list = []
        color_list = []

        for order_idx in range(order_lo, order_hi+1):
            order_controller[order_idx] = 1
            legp = np.polynomial.legendre.Legendre( order_controller )
            legcoef = np.polynomial.legendre.leg2poly(legp.coef )
            poly = np.polynomial.Polynomial(legcoef)
            y_list.append( poly(x) )
            print(order_idx, " ", poly(x))
            legend_list.append( "order_"+str(order_idx) )
            color_list.append("k")

            order_controller[order_idx] = 0 

        plot_worker.instant_plot_unified_x_axis(x, y_list, "Legendre Poly" , "x", "y", legend_list, color_list)

    def _db_lbcg_exp (self):
        """ """
        lbcg_solver_obj = LBCG()
        self._final_x_a, self._final_r_a, self._residual_hist_a = \
                 lbcg_solver_obj.lbcg_solver(self._mat, self._B, self._X, 8, self._tol, self._maxiter)
        self._final_x_b, self._final_r_b, self._residual_hist_b = \
                 lbcg_solver_obj.lbcg_solver(self._mat, self._B, self._X, 16, self._tol, self._maxiter)

        cbcg_solver_obj = CBCG() 
        self._final_x_c, self._final_r_c, self._residual_hist_c = \
               cbcg_solver_obj.cbcg_solver(self._mat, self._B, self._X, 16, self._tol, self._maxiter)

        plot_worker = Presenter()
        residual_list = [self._residual_hist_a, self._residual_hist_b, self._residual_hist_c]
        legend_list = ["lbcg_s8","lbcg_s16", "cbcg_s16"]
        color_list = ["r","k", "b"]
        #plot_worker.instant_plot_y_log10(residual_list, "crystm01", "#iteration", "$\\frac{||x_1||}{||b_1||}$", legend_list, color_list)
        plot_worker.instant_plot_y_log10(residual_list, "wathen100", "#iteration", "$\\frac{||x_1||}{||b_1||}$", legend_list, color_list)

    def _db_blbcg_exp(self):
        """ """
        lbcg_solver_obj = LBCG()
        self._final_x_a, self._final_r_a, self._residual_hist_a = \
                 lbcg_solver_obj.lbcg_solver(self._mat, self._SB, self._SX, 8, self._tol, self._maxiter)

        blbcg_solver_obj = BLBCG()
        self._final_x_b, self._final_r_b, self._residual_hist_b = \
                 blbcg_solver_obj.blbcg_solver(self._mat, self._BB, self._BX, 8, self._tol, self._maxiter, 0)

        bcbcg_solver_obj = BCBCG()
        self._final_x_c, self._final_r_c, self._residual_hist_c = \
                 bcbcg_solver_obj.bcbcg_solver(self._mat, self._BB, self._BX, 8, self._tol, self._maxiter, 0)

        plot_worker = Presenter()
        residual_list = [self._residual_hist_a, self._residual_hist_b, self._residual_hist_c]
        legend_list = ["lbcg_s8","blbcg_s8b10", "bcbcg_s8b10"]
        color_list = ["r","k", "b"]
        plot_worker.instant_plot_y_log10(residual_list, "bodyy6", "#iteration", "$\\frac{||x_1||}{||b_1||}$", legend_list, color_list)

    def _numpy_lstsq_test (self):
        """ """
        self._small_mat = np.random.random( ( 5,5) )
        self._small_rhs = np.random.random( ( 5,3) )
        self._lstsq_res = np.linalg.lstsq(self._small_mat, self._small_rhs)
        print (self._small_mat)
        print("")
        print(self._small_rhs)
        print("")
        print(self._lstsq_res)
        print("")
        print(np.matmul(self._small_mat, self._lstsq_res[0]))
        #print(type(self._small_mat), "", type(self._lstsq_res))

    def _db_cbcg_lstsq (self):
        cbcg_solver_obj = CBCG()
        self._final_x_a, self._final_r_a, self._residual_hist_a = \
               cbcg_solver_obj.cbcg_solver_least_square(self._mat, self._SB, self._SX, self._step_val, self._tol, self._maxiter)

        self._final_x_b, self._final_r_b, self._residual_hist_b = \
               cbcg_solver_obj.cbcg_solver_least_square(self._mat, self._SB, self._SX, self._step_val, self._tol, self._maxiter)

        plot_worker = Presenter()
        residual_list = [self._residual_hist_a, self._residual_hist_b]
        legend_list = ["cbcg_s2_lstsq","blbcg_s2"]
        color_list = ["r","k"]
        plot_worker.instant_plot_y_log10(residual_list, "bodyy6", "#iteration", "$\\frac{||x_1||}{||b_1||}$", legend_list, color_list)

    def _db_bcbcg_lstsq (self):
        """ """
        bcbcg_solver_obj = BCBCG()
        self._final_X_a, self._final_R_a, self._residual_hist_a = \
               bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB, self._BX, self._step_val, self._tol, self._maxiter,0)
        self._final_X_b, self._final_R_b, self._residual_hist_b = \
               bcbcg_solver_obj.bcbcg_solver(self._mat, self._BB, self._BX, self._step_val, self._tol, self._maxiter,0)

        plot_worker = Presenter()
        residual_list = [self._residual_hist_a, self._residual_hist_b]
        legend_list = ["bcbcg_s20b4_lstsq","bcbcg_s20b4"]
        color_list = ["r","k"]
        plot_worker.instant_plot_y_log10(residual_list, "crystm02", "#iteration", "$\\frac{||x_1||}{||b_1||}$", legend_list, color_list)

    def _db_lbcg_least_square_exp (self):
        """ """
        lbcg_solver_obj = LBCG()
        self._final_x_a, self._final_r_a, self._residual_hist_a = \
                 lbcg_solver_obj.lbcg_solver_least_square(self._mat, self._SB, self._SX, self._step_val, self._tol, self._maxiter)

        cbcg_solver_obj = CBCG() 
        self._final_x_b, self._final_r_b, self._residual_hist_b = \
               cbcg_solver_obj.cbcg_solver(self._mat, self._SB, self._SX, self._step_val, self._tol, self._maxiter)

        plot_worker = Presenter()
        residual_list = [self._residual_hist_a, self._residual_hist_b]
        legend_list = ["lbcg_lstsq_s2", "cbcg_s2"]
        color_list = ["r","k"]
        #plot_worker.instant_plot_y_log10(residual_list, "crystm01", "#iteration", "$\\frac{||x_1||}{||b_1||}$", legend_list, color_list)
        plot_worker.instant_plot_y_log10(residual_list, "wathen100", "#iteration", "$\\frac{||x_1||}{||b_1||}$", legend_list, color_list)

    def _db_blbcg_least_square_exp(self):
        """ """
        blbcg_solver_obj = BLBCG()
        self._final_x_a, self._final_r_a, self._residual_hist_a = \
                 blbcg_solver_obj.blbcg_solver_least_square(self._mat, self._BB, self._BX, self._step_val, self._tol, self._maxiter, 0)

        bcbcg_solver_obj = BCBCG()
        self._final_x_b, self._final_r_b, self._residual_hist_b = \
                 bcbcg_solver_obj.bcbcg_solver(self._mat, self._BB, self._BX, self._step_val, self._tol, self._maxiter, 0)

        plot_worker = Presenter()
        residual_list = [self._residual_hist_a, self._residual_hist_b]
        legend_list = ["blbcg_s2b3_lstsq","blbcg_s2b3"]
        color_list = ["r","k"]
        plot_worker.instant_plot_y_log10(residual_list, "crystm01", "#iteration", "$\\frac{||x_1||}{||b_1||}$", legend_list, color_list)

    def _blbcg_least_square_exp(self):
        """ """
        blbcg_solver_obj = BLBCG()
        self._final_x_a, self._final_r_a, self._residual_hist_a = \
                 blbcg_solver_obj.blbcg_solver_least_square(self._mat, self._BB, self._BX, self._step_val, self._tol, self._maxiter, 0)

        bcbcg_solver_obj = BCBCG()
        self._final_x_b, self._final_r_b, self._residual_hist_b = \
                 bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB, self._BX, self._step_val, self._tol, self._maxiter, 0)

        plot_worker = Presenter()
        residual_list = [self._residual_hist_a, self._residual_hist_b]
        legend_list = ["blbcg_s64b4_lstsq","bcbcg_s64b4_lstsq"]
        color_list = ["r","k"]
        #plot_worker.instant_plot_y_log10(residual_list, "crystm01", "#iteration", "$\\frac{||x_1||}{||b_1||}$", legend_list, color_list)
        plot_worker.instant_plot_y_log10(residual_list, "bodyy6", "#iteration", "$\\frac{||x_1||}{||b_1||}$", legend_list, color_list)

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
    mat_path = "/home/scl/MStore/bodyy6/bodyy6.mtx"
    #mat_path = "/home/scl/MStore/crystm02/crystm02.mtx"


    block_size = 4 
    tol = 1e-12
    maxiter = 500
    step_val =64

    linear_system_solver_worker_test = WorkerIterativeLinearSystemSolverCG_Exp_160521(mat_path)
    linear_system_solver_worker_test.conduct_experiments(block_size,tol,maxiter, step_val)
    #linear_system_solver_worker_test.chebyshev_poly_exp_a(0,6)
    #linear_system_solver_worker_test.legendre_poly_exp_a(0,6)
    #linear_system_solver_worker_test.debug_NativeConjugateGradient()



if __name__ == "__main__":
    """ call main funtion for testing """
    main()
