""" 
Experiment Diary 2016-07-14
"""
import sys
import math
import matplotlib.pyplot as plt
from scipy import io
import numpy as np
from scipy.sparse.linalg import *
from scipy import sparse
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
from mat_generator import MatGenerator
from scipy.sparse import dia_matrix
from scipy.sparse import eye 

class WorkerIterativeLinearSystemSolverCG_Exp_160714(Worker):
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

        if mat_path == "NeedMatGeneration":
            """ Need to generatre matrix """
            print("please call obj.matrix_generation(dim ,left_semi_bw, right_semi_bw, val_min, val_max)")
        else:
            self._mat_coo = io.mmread(mat_path)
            self._mat = self._mat_coo.tocsr()
            self._mat_info = io.mminfo(mat_path)
            print("Done reading matrix {}, Row:{}, Col:{}".format( mat_path, self._mat.shape[0], self._mat.shape[1]))
            print("mminfo:{}".format(self._mat_info))
            if self._mat.getformat() == "csr":
                print("Yeah, it is CSR")

    def mat_generation(self, dim ,left_semi_bw, right_semi_bw, val_min, val_max, factor_diag):
        """ """ 
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print ("going to generate a dia matrix of dim:{}, left_semi_bw:{} ,right_semi_bw:{}, val_min:{}, val_max:{}".format( \
            dim, left_semi_bw, right_semi_bw, val_min, val_max) )
        mat_generator = MatGenerator()
        #self._mat_dia = mat_generator.dia_mat_band_random_square(dim, left_semi_bw, right_semi_bw, val_min, val_max)
        #self._mat_dia = mat_generator.dia_mat_band_random_square_PDpossible(dim, left_semi_bw, right_semi_bw, val_min, val_max, factor_diag)

        semi_bw = left_semi_bw
        self._mat_dia = mat_generator.dia_mat_band_random_square_sym_PDpossible(dim, semi_bw, val_min, val_max, factor_diag)
        #print(self._mat_dia.todense())
        self._mat = self._mat_dia.tocsr()

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

        self._mat_eigenV_max = eigs(self._mat,k=1, tol=1e-3)[0][0] 
        print(self._mat_eigenV_max,"  ", self._mat_eigenV_max.real, "  ", np.absolute(self._mat_eigenV_max))
        assert np.absolute(self._mat_eigenV_max) == self._mat_eigenV_max.real

        self._mat_eigenV_min = eigs(self._mat,k=1, which="SM",tol=1e-3)[0][0]
        print(self._mat_eigenV_min,"  ", self._mat_eigenV_min.real, "  ", np.absolute(self._mat_eigenV_min))
        assert np.absolute(self._mat_eigenV_min) == self._mat_eigenV_min.real

        #print(np.linalg.eig(self._mat.todense()))
        self._cg_bcg_bcbcg_blcg_exp(block_size)


        if (self._mat_eigenV_max/self._mat_eigenV_min) < 400. :
            self._mat = self._mat - 0.999*self._mat_eigenV_min.real*eye(self._mat.shape[0])

        self._mat_eigenV_max = eigs(self._mat,k=1, tol=1e-3)[0][0] 
        print(self._mat_eigenV_max,"  ", self._mat_eigenV_max.real, "  ", np.absolute(self._mat_eigenV_max))
        assert np.absolute(self._mat_eigenV_max) == self._mat_eigenV_max.real

        self._mat_eigenV_min = eigs(self._mat,k=1, which="SM",tol=1e-3)[0][0]
        print(self._mat_eigenV_min,"  ", self._mat_eigenV_min.real, "  ", np.absolute(self._mat_eigenV_min))
        assert np.absolute(self._mat_eigenV_min) == self._mat_eigenV_min.real

        #print(np.linalg.eig(self._mat.todense()))
        self._cg_bcg_bcbcg_blcg_exp(block_size)

        #self._cg_bcg_blbcg_least_square_exp()
        #self._bcbcg_blbcg_least_square_exp()

        #isc16 figure 2
        #self._bcbcg_blbcg_least_square_exp_b()

        #self._db_bcg_least_square()
        #self._db_bcbcg_eigen_param()
        #self._db_usage_scipy_eig()
        #self._diff_eigen_estimation_test_b()

        #isc16 figure 3
        #self._diff_eigen_estimation_test_c()

        #self._db_power_iteration_with_shifting_acc1()
        print("Experiments done")


    def _cg_bcg_exp(self, block_size):
        """ """
        print("_cg_bcg_bcbcg_least_square_exp starting, ... ")
        self._BB_1  = np.random.random( ( self._mat.shape[0],1) )
        self._BX_1  = np.ones ( (self._mat.shape[1],1) )
        self._BB_X = np.random.random( ( self._mat.shape[0],block_size) )
        self._BX_X = np.ones ( (self._mat.shape[1],block_size) )

        #line 1
        bcg_solver_obj = NativeBlockConjugateGradient(self._mat, self._BX_1, self._BB_1, self._tol, self._maxiter)
        self._final_X_cg, self._final_R_cg, self._residual_hist_cg = bcg_solver_obj.bcg_variant_lstsq_run(0)

        ##line 2
        bcg_solver_obj = NativeBlockConjugateGradient(self._mat, self._BX_X, self._BB_X, self._tol, self._maxiter)
        self._final_X_bcg_mX, self._final_R_bcg_mX, self._residual_hist_bcg_mX = bcg_solver_obj.bcg_variant_lstsq_run(0)


        plot_worker = Presenter()
        residual_list = [self._residual_hist_cg, self._residual_hist_bcg_mX]

        legend_list = ["cg","bcg_m"+str(block_size)]
        color_list = ["r","k"]
        plot_worker.instant_plot_y_log10(residual_list, "MG", "#iteration", "$\\mathbf{log_{10}\\frac{||r_1||}{||b_1||}}$", legend_list, color_list)

    def _cg_bcg_bcbcg_blcg_exp(self, block_size):
        """ """
        print("_cg_bcg_bcbcg_least_square_exp starting, ... ")
        self._BB_1  = np.random.random( ( self._mat.shape[0],1) )
        self._BX_1  = np.ones ( (self._mat.shape[1],1) )
        self._BB_X = np.random.random( ( self._mat.shape[0],block_size) )
        self._BX_X = np.ones ( (self._mat.shape[1],block_size) )

        #line 1
        bcg_solver_obj = NativeBlockConjugateGradient(self._mat, self._BX_1, self._BB_1, self._tol, self._maxiter)
        self._final_X_cg, self._final_R_cg, self._residual_hist_cg = bcg_solver_obj.bcg_variant_lstsq_run(0)

        ##line 2
        bcg_solver_obj = NativeBlockConjugateGradient(self._mat, self._BX_X, self._BB_X, self._tol, self._maxiter)
        self._final_X_bcg_mX, self._final_R_bcg_mX, self._residual_hist_bcg_mX = bcg_solver_obj.bcg_variant_lstsq_run(0)


        #line 3
        bcbcg_solver_obj = BCBCG()
        self._final_x_bcbcg_m1sY, self._final_r_bcbcg_m1sY, self._residual_hist_bcbcg_m1sY = \
                bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB_1, self._BX_1, self._step_val, self._tol, self._maxiter, 0)

        #line 4
        bcbcg_solver_obj = BCBCG()
        self._final_x_bcbcg_mXsY, self._final_r_bcbcg_mXsY, self._residual_hist_bcbcg_mXsY = \
                bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB_X, self._BX_X, self._step_val, self._tol, self._maxiter, 0)

        #line 5
        blbcg_solver_obj = BLBCG()
        self._final_x_blbcg_mXsY, self._final_r_blbcg_mXsY, self._residual_hist_blbcg_mXsY = \
                blbcg_solver_obj.blbcg_solver_least_square(self._mat, self._BB_X, self._BX_X, self._step_val, self._tol, self._maxiter, 0)
        
        plot_worker = Presenter()
        residual_list = [self._residual_hist_cg, self._residual_hist_bcg_mX, self._residual_hist_bcbcg_m1sY, \
                self._residual_hist_bcbcg_mXsY, self._residual_hist_blbcg_mXsY]

        legend_list = ["cg","bcg_m"+str(block_size), "bcbcg_m1_s"+str(self._step_val), \
                "bcbcg_m"+str(block_size)+"s"+str(self._step_val), "blbcg_m"+str(block_size)+"s"+str(self._step_val)]
        color_list = ["r","k","b","g","m"]
        plot_worker.instant_plot_y_log10(residual_list, "MG", "#iteration", "$\\mathbf{log_{10}\\frac{||r_1||}{||b_1||}}$", legend_list, color_list)

    def _cg_bcg_bcbcg_least_square_exp(self):
        """ """
        print("_cg_bcg_bcbcg_least_square_exp starting, ... ")
        self._BB_1  = np.random.random( ( self._mat.shape[0],1) )

        self._BX_1  = np.ones ( (self._mat.shape[1],1) )
        self._BB_4  = np.random.random( ( self._mat.shape[0],4) )
        self._BX_4  = np.ones ( (self._mat.shape[1],4) )
        self._BB_12 = np.random.random( ( self._mat.shape[0],12) )
        self._BX_12 = np.ones ( (self._mat.shape[1],12) )

        #line 1
        bcg_solver_obj = NativeBlockConjugateGradient(self._mat, self._BX_1, self._BB_1, self._tol, self._maxiter)
        self._final_X_cg, self._final_R_cg, self._residual_hist_cg = bcg_solver_obj.bcg_variant_lstsq_run(0)

        ##line 2
        bcg_solver_obj = NativeBlockConjugateGradient(self._mat, self._BX_12, self._BB_12, self._tol, self._maxiter)
        self._final_X_bcg_m12, self._final_R_bcg_m12, self._residual_hist_bcg_m12 = bcg_solver_obj.bcg_variant_lstsq_run(0)

        ##line 3
        #bcbcg_solver_obj = BCBCG()
        #self._final_x_bcbcg_m1s2, self._final_r_bcbcg_m1s2, self._residual_hist_bcbcg_m1s2 = \
        #        bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB_1, self._BX_1, 2, self._tol, self._maxiter, 0)

        ##line 4
        #bcbcg_solver_obj = BCBCG()
        #self._final_x_bcbcg_m1s8, self._final_r_bcbcg_m1s8, self._residual_hist_bcbcg_m1s8 = \
        #        bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB_1, self._BX_1, 8, self._tol, self._maxiter, 0)

        ##line 5
        #bcbcg_solver_obj = BCBCG()
        #self._final_x_bcbcg_m4s2, self._final_r_bcbcg_m4s2, self._residual_hist_bcbcg_m4s2 = \
        #        bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB_4, self._BX_4, 2, self._tol, self._maxiter, 0)

        ##line 6
        #bcbcg_solver_obj = BCBCG()
        #self._final_x_bcbcg_m4s8, self._final_r_bcbcg_m4s8, self._residual_hist_bcbcg_m4s8 = \
        #        bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB_4, self._BX_4, 8, self._tol, self._maxiter, 0)

        plot_worker = Presenter()
        #residual_list = [self._residual_hist_cg, self._residual_hist_bcg_m12,  \
        #                 self._residual_hist_bcbcg_m1s2, self._residual_hist_bcbcg_m1s8, \
        #                 self._residual_hist_bcbcg_m4s2, self._residual_hist_bcbcg_m4s8 ]
        residual_list = [self._residual_hist_cg, self._residual_hist_bcg_m12]

        #legend_list = ["cg","bcg_m12", "bcbcg_m1s2", "bcbcg_m1s8", "bcbcg_m4s2", "bcbcg_m4s8"]
        legend_list = ["cg","bcg_m12"]
        #color_list = ["r","k","b","y","m","g"]
        color_list = ["r","k"]
        #plot_worker.instant_plot_y_log10(residual_list, "test", "#iteration", "$\\mathbf{log_{10}\\frac{||x_1||}{||b_1||}}$", legend_list, color_list)
        #plot_worker.instant_plot_y_log10(residual_list, "wathen100(dim=30,401, nnz=471,601, cond=5816.01 )", "#iteration", "$\\mathbf{log_{10}\\frac{||x_1||}{||b_1||}}$", legend_list, color_list)
        plot_worker.instant_plot_y_log10(residual_list, "MG", "#iteration", "$\\mathbf{log_{10}\\frac{||x_1||}{||b_1||}}$", legend_list, color_list)


    def _bcbcg_blbcg_least_square_exp_b(self):
        """ figure 2"""
        print("_bcbcg_blbcg_least_square_exp_b starting ... ")

        m=3
        self._BB  = np.random.random( ( self._mat.shape[0],m) )
        self._BX  = np.ones ( (self._mat.shape[1],m) )


        #line 1
        blbcg_solver_obj = BLBCG()
        self._final_x_blbcg_a, self._final_r_blbcg_a, self._residual_hist_blbcg_a = \
                blbcg_solver_obj.blbcg_solver_least_square(self._mat, self._BB, self._BX, 16, self._tol, self._maxiter, 0)
        #line 2
        blbcg_solver_obj = BLBCG()
        self._final_x_blbcg_b, self._final_r_blbcg_b, self._residual_hist_blbcg_b = \
                blbcg_solver_obj.blbcg_solver_least_square(self._mat, self._BB, self._BX, 32, self._tol, self._maxiter, 0)

        #line addition 
        #blbcg_solver_obj = BLBCG()
        #self._final_x_blbcg_c, self._final_r_blbcg_c, self._residual_hist_blbcg_c = \
        #        blbcg_solver_obj.blbcg_solver_least_square(self._mat, self._BB_4, self._BX_4, 32, self._tol, self._maxiter, 0)

        #line 3
        bcbcg_solver_obj = BCBCG()
        self._final_x_bcbcg_a, self._final_r_bcbcg_a, self._residual_hist_bcbcg_a = \
                bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB, self._BX, 16, self._tol, self._maxiter, 0)
        #line 4
        bcbcg_solver_obj = BCBCG()
        self._final_x_bcbcg_b, self._final_r_bcbcg_b, self._residual_hist_bcbcg_b = \
                bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB, self._BX, 32, self._tol, self._maxiter, 0)

        #line addition
        #bcbcg_solver_obj = BCBCG()
        #self._final_x_bcbcg_c, self._final_r_bcbcg_c, self._residual_hist_bcbcg_c = \
        #        bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB_4, self._BX_4, 32, self._tol, self._maxiter, 0)


        plot_worker = Presenter()
        #residual_list = [self._residual_hist_blbcg_a, self._residual_hist_blbcg_b, self._residual_hist_blbcg_c, \
        #                 self._residual_hist_bcbcg_a, self._residual_hist_bcbcg_b, self._residual_hist_bcbcg_c  ]
        residual_list = [self._residual_hist_blbcg_a, self._residual_hist_blbcg_b,  \
                         self._residual_hist_bcbcg_a, self._residual_hist_bcbcg_b  ]

        #legend_list = ["blbcg_m4s4", "blbcg_m4s8", "blbcg_m4s12", "bcbcg_m4s4", "bcbcg_m4s8", "bcbcg_m4s12"]
        legend_list = ["blbcg_m3s16", "blbcg_m3s32", "bcbcg_m3s16", "bcbcg_m3s32"]

        #color_list = ["r","k","b","y","g","m"]
        color_list = ["r","k","b","g"]
        #plot_worker.instant_plot_y_log10(residual_list, "test", "#iteration", "$\\mathbf{log_{10}\\frac{||x_1||}{||b_1||}}$", legend_list, color_list)
        plot_worker.instant_plot_y_log10(residual_list, "bodyy6", "#iteration", "$\\mathbf{log_{10}\\frac{||x_1||}{||b_1||}}$", legend_list, color_list)


    def _diff_eigen_estimation_test_c(self):
        """ """
        print("_diff_eigen_estimation_test_c starting ...")
        gerschgorin_estimator = GerschgorinCircleTheoremEigenvalueEstimator()
        gerschgorin_max_eigenvalue, gerschgorin_min_eigenvalue = gerschgorin_estimator.csr_mat_extreme_eigenvalue_estimation(self._mat)
        print("################Gershchogrin theorem", "min:",gerschgorin_max_eigenvalue, " , old min:", gerschgorin_min_eigenvalue)
        if gerschgorin_min_eigenvalue< 0.:
            gerschgorin_min_eigenvalue = 0.
        print("################Gershchogrin theorem", "min:",gerschgorin_max_eigenvalue, " , new min:", gerschgorin_min_eigenvalue)

        eigs_vals, eigs_vecs = eigs(self._mat, k=6, which="LM")
        print("eigs_vals max", eigs_vals)
        #eigs_vals, eigs_vecs = eigs(self._mat, k=6, tol=1e-3,which="SR")
        #print("eigs_vals min", eigs_vals)

        power_method_solver = PowerIteration()
        self._init_eigen_vec  = np.random.random( ( self._mat.shape[0],1) )
        pm_maxiters = 500
        pm_tol = 1e-6
        pm_max_eigen_vec, pm_max_eigen_val, pm_max_eigen_list = power_method_solver.naive_power_iteration (self._mat, self._init_eigen_vec, pm_maxiters, pm_tol)
        print("################Power method max:",pm_max_eigen_val, " , iteration:", len(pm_max_eigen_list))
        #self._init_eigen_vec  = np.random.random( ( self._mat.shape[0],1) )
        #pm_min_eigen_vec, pm_min_eigen_val, pm_min_eigen_list = power_method_solver.power_iteration_with_shifting_acc1 (self._mat, self._init_eigen_vec, pm_max_eigen_val, pm_maxiters, pm_tol)
        #pm_min_eigen_val = pm_min_eigen_val + pm_max_eigen_val
        #print("################Power method min:",pm_min_eigen_val, " , iteration:", len(pm_min_eigen_list))


        ##
        eigen_repo = {"gerschgorin":(gerschgorin_max_eigenvalue, gerschgorin_min_eigenvalue),\
                      "mix_method":(pm_max_eigen_val,gerschgorin_min_eigenvalue) \
                     }
        print(eigen_repo["gerschgorin"], " , ", eigen_repo["mix_method"])

        self._BB  = np.random.random( ( self._mat.shape[0],3) )
        self._BX  = np.ones ( (self._mat.shape[1],3) )

        step_val = 32 

        #line 2
        bcbcg_solver_obj = BCBCG()
        self._final_gerschgorin_x_bcbcg_a, self._final_gerschgorin_r_bcbcg_a, self._gerschgorin_residual_hist_bcbcg_a = \
                bcbcg_solver_obj.bcbcg_solver_least_square_eigen_param(self._mat, self._BB, self._BX, step_val, self._tol, self._maxiter, 0, eigen_repo["gerschgorin"][0], eigen_repo["gerschgorin"][1])

        #line 4
        bcbcg_solver_obj = BCBCG()
        self._final_mix_x_bcbcg_b, self._final_mix_r_bcbcg_b, self._mix_residual_hist_bcbcg_b = \
                bcbcg_solver_obj.bcbcg_solver_least_square_eigen_param(self._mat, self._BB, self._BX, step_val, self._tol, self._maxiter, 0, eigen_repo["mix_method"][0], eigen_repo["mix_method"][1])

        #addition
        blbcg_solver_obj = BLBCG()
        self._final_gerschgorin_x_blbcg_a, self._final_gerschgorin_r_blbcg_a, self._gerschgorin_residual_hist_blbcg_a = \
                blbcg_solver_obj.blbcg_solver_least_square_eigen_param(self._mat, self._BB, self._BX, step_val, self._tol, self._maxiter, 0, eigen_repo["gerschgorin"][0], eigen_repo["gerschgorin"][1])

        blbcg_solver_obj = BLBCG()
        self._final_mix_x_blbcg_b, self._final_mix_r_blbcg_b, self._mix_residual_hist_blbcg_b = \
                blbcg_solver_obj.blbcg_solver_least_square_eigen_param(self._mat, self._BB, self._BX, step_val, self._tol, self._maxiter, 0, eigen_repo["mix_method"][0], eigen_repo["mix_method"][1])

        plot_worker = Presenter()
        residual_list = [self._gerschgorin_residual_hist_bcbcg_a, self._mix_residual_hist_bcbcg_b , \
                         self._gerschgorin_residual_hist_blbcg_a, self._mix_residual_hist_blbcg_b ]

        legend_list = ["G_bcbcg_m3s32","M_bcbcg_m3s32" , "G_blbcg_m3s32","M_blbcg_m3s32"]
        #legend_list = ["G_bcbcg_m3s8","M_bcbcg_m3s8" , "G_blbcg_m3s8","M_blbcg_m3s8"]
        color_list = ["r","k","b","g"]
        #plot_worker.instant_plot_y_log10(residual_list, "test", "#iteration", "$\\mathbf{log_{10}\\frac{||x_1||}{||b_1||}}$", legend_list, color_list)
        #plot_worker.instant_plot_y_log10(residual_list, "wathen100", "#iteration", "$\\mathbf{log_{10}\\frac{||x_1||}{||b_1||}}$", legend_list, color_list)
        plot_worker.instant_plot_y_log10(residual_list, "bodyy6", "#iteration", "$\\mathbf{log_{10}\\frac{||x_1||}{||b_1||}}$", legend_list, color_list)


def main ():
# main function for today's experiments 
    #small matrix for debuging
    #mat_path = "/home/scl/MStore/mesh1e1/mesh1e1.mtx"

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
    #mat_path = "/home/scl/MStore/Chem97ZtZ/Chem97ZtZ.mtx"

    #isc16
    #mat_path = "/home/scl/MStore/bodyy6/bodyy6.mtx"
    #mat_path = "/home/scl/MStore/wathen100/wathen100.mtx"

    mat_path = "/home/scl/MStore/Chem97ZtZ/Chem97ZtZ.mtx"

    block_size = 10
    tol = 1e-10
    maxiter = 1000
    step_val =5

    dim = 10000
    left_semi_band ,right_semi_band = 1,1
    #big change , big cond
    #valmin = 0.
    valmin = 0.
    valmax = 100.
    #factor_diag = float(dim) 
    factor_diag =  1.44 * valmax 

    #linear_system_solver_worker_test = WorkerIterativeLinearSystemSolverCG_Exp_160712(mat_path)
    #linear_system_solver_worker_test.conduct_experiments(block_size,tol,maxiter, step_val)

    linear_system_solver_worker = WorkerIterativeLinearSystemSolverCG_Exp_160714("NeedMatGeneration")
    #linear_system_solver_worker.mat_generation( dim, left_semi_band , right_semi_band, valmin, valmax)
    linear_system_solver_worker.mat_generation( dim, left_semi_band , right_semi_band, valmin, valmax, factor_diag)
    linear_system_solver_worker.conduct_experiments(block_size,tol,maxiter, step_val)


if __name__ == "__main__":
    """ call main funtion for testing """
    main()
