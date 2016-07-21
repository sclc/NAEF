""" 
Experiment Diary 2016-06-14
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

class WorkerIterativeLinearSystemSolverCG_Exp_160606_isc16(Worker):
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

        #isc16 figure 1
        #self._cg_bcg_bcbcg_least_square_exp()

        #self._cg_bcg_blbcg_least_square_exp()
        #self._bcbcg_blbcg_least_square_exp()

        #isc16 figure 2
        #self._bcbcg_blbcg_least_square_exp_b()

        #self._db_bcg_least_square()
        #self._db_bcbcg_eigen_param()
        #self._db_usage_scipy_eig()
        #self._diff_eigen_estimation_test_b()

        #isc16 figure 3
        self._diff_eigen_estimation_test_c()

        #self._db_power_iteration_with_shifting_acc1()
        print("Experiments done")

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

        #line 2
        bcg_solver_obj = NativeBlockConjugateGradient(self._mat, self._BX_12, self._BB_12, self._tol, self._maxiter)
        self._final_X_bcg_m12, self._final_R_bcg_m12, self._residual_hist_bcg_m12 = bcg_solver_obj.bcg_variant_lstsq_run(0)

        #line 3
        bcbcg_solver_obj = BCBCG()
        self._final_x_bcbcg_m1s2, self._final_r_bcbcg_m1s2, self._residual_hist_bcbcg_m1s2 = \
                bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB_1, self._BX_1, 2, self._tol, self._maxiter, 0)

        #line 4
        bcbcg_solver_obj = BCBCG()
        self._final_x_bcbcg_m1s8, self._final_r_bcbcg_m1s8, self._residual_hist_bcbcg_m1s8 = \
                bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB_1, self._BX_1, 8, self._tol, self._maxiter, 0)

        #line 5
        bcbcg_solver_obj = BCBCG()
        self._final_x_bcbcg_m4s2, self._final_r_bcbcg_m4s2, self._residual_hist_bcbcg_m4s2 = \
                bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB_4, self._BX_4, 2, self._tol, self._maxiter, 0)

        #line 6
        bcbcg_solver_obj = BCBCG()
        self._final_x_bcbcg_m4s8, self._final_r_bcbcg_m4s8, self._residual_hist_bcbcg_m4s8 = \
                bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB_4, self._BX_4, 8, self._tol, self._maxiter, 0)

        plot_worker = Presenter()
        residual_list = [self._residual_hist_cg, self._residual_hist_bcg_m12,  \
                         self._residual_hist_bcbcg_m1s2, self._residual_hist_bcbcg_m1s8, \
                         self._residual_hist_bcbcg_m4s2, self._residual_hist_bcbcg_m4s8 ]

        legend_list = ["cg","bcg_m12", "bcbcg_m1s2", "bcbcg_m1s8", "bcbcg_m4s2", "bcbcg_m4s8"]
        color_list = ["r","k","b","y","m","g"]
        #plot_worker.instant_plot_y_log10(residual_list, "test", "#iteration", "$\\mathbf{log_{10}\\frac{||x_1||}{||b_1||}}$", legend_list, color_list)
        #plot_worker.instant_plot_y_log10(residual_list, "wathen100(dim=30,401, nnz=471,601, cond=5816.01 )", "#iteration", "$\\mathbf{log_{10}\\frac{||x_1||}{||b_1||}}$", legend_list, color_list)
        plot_worker.instant_plot_y_log10(residual_list, "wathen100", "#iteration", "$\\mathbf{log_{10}\\frac{||x_1||}{||b_1||}}$", legend_list, color_list)


    #def _cg_bcg_blbcg_least_square_exp(self):
    #    """ """
    #    print("_cg_bcg_blbcg_least_square_exp starting, ... ")

    #    self._BB_1  = np.random.random( ( self._mat.shape[0],1) )
    #    self._BX_1  = np.ones ( (self._mat.shape[1],1) )
    #    self._BB_6  = np.random.random( ( self._mat.shape[0],6) )
    #    self._BX_6  = np.ones ( (self._mat.shape[1],6) )
    #    self._BB_12 = np.random.random( ( self._mat.shape[0],12) )
    #    self._BX_12 = np.ones ( (self._mat.shape[1],12) )

    #    #line 1
    #    bcg_solver_obj = NativeBlockConjugateGradient(self._mat, self._BX_1, self._BB_1, self._tol, self._maxiter)
    #    self._final_X_cg, self._final_R_cg, self._residual_hist_cg = bcg_solver_obj.bcg_variant_lstsq_run(0)

    #    #line 2
    #    bcg_solver_obj = NativeBlockConjugateGradient(self._mat, self._BX_12, self._BB_12, self._tol, self._maxiter)
    #    self._final_X_bcg_m12, self._final_R_bcg_m12, self._residual_hist_bcg_m12 = bcg_solver_obj.bcg_variant_lstsq_run(0)

    #    #line 3
    #    blbcg_solver_obj = BLBCG()
    #    self._final_x_blbcg_m1s2, self._final_r_blbcg_m1s2, self._residual_hist_blbcg_m1s2 = \
    #            blbcg_solver_obj.blbcg_solver_least_square(self._mat, self._BB_1, self._BX_1, 2, self._tol, self._maxiter, 0)

    #    #line 4
    #    blbcg_solver_obj = BLBCG()
    #    self._final_x_blbcg_m1s6, self._final_r_blbcg_m1s6, self._residual_hist_blbcg_m1s6 = \
    #            blbcg_solver_obj.blbcg_solver_least_square(self._mat, self._BB_1, self._BX_1, 6, self._tol, self._maxiter, 0)

    #    #line 5
    #    blbcg_solver_obj = BLBCG()
    #    self._final_x_blbcg_m6s2, self._final_r_blbcg_m6s2, self._residual_hist_blbcg_m6s2 = \
    #            blbcg_solver_obj.blbcg_solver_least_square(self._mat, self._BB_6, self._BX_6, 2, self._tol, self._maxiter, 0)

    #    #line 6
    #    blbcg_solver_obj = BLBCG()
    #    self._final_x_blbcg_m6s6, self._final_r_blbcg_m6s6, self._residual_hist_blbcg_m6s6 = \
    #            blbcg_solver_obj.blbcg_solver_least_square(self._mat, self._BB_6, self._BX_6, 6, self._tol, self._maxiter, 0)

    #    plot_worker = Presenter()
    #    residual_list = [self._residual_hist_cg, self._residual_hist_bcg_m12,  \
    #                     self._residual_hist_blbcg_m1s2, self._residual_hist_blbcg_m1s6, \
    #                     self._residual_hist_blbcg_m6s2, self._residual_hist_blbcg_m6s6 ]

    #    legend_list = ["cg","bcg_m12", "blbcg_m1s2", "blbcg_m1s6", "blbcg_m6s2", "blbcg_m6s6"]
    #    color_list = ["r","k","b","y","m","g"]
    #    plot_worker.instant_plot_y_log10(residual_list, "test", "#iteration", "$\\mathbf{log_{10}\\frac{||x_1||}{||b_1||}}$", legend_list, color_list)

    #def _bcbcg_blbcg_least_square_exp(self):
    #    """ """
    #    self._BB_1  = np.random.random( ( self._mat.shape[0],1) )
    #    self._BX_1  = np.ones ( (self._mat.shape[1],1) )
    #    self._BB_6  = np.random.random( ( self._mat.shape[0],6) )
    #    self._BX_6  = np.ones ( (self._mat.shape[1],6) )

    #    #line 1
    #    blbcg_solver_obj = BLBCG()
    #    self._final_x_blbcg_m1s2, self._final_r_blbcg_m1s2, self._residual_hist_blbcg_m1s2 = \
    #            blbcg_solver_obj.blbcg_solver_least_square(self._mat, self._BB_1, self._BX_1, 2, self._tol, self._maxiter, 0)

    #    #line 2
    #    blbcg_solver_obj = BLBCG()
    #    self._final_x_blbcg_m1s6, self._final_r_blbcg_m1s6, self._residual_hist_blbcg_m1s6 = \
    #            blbcg_solver_obj.blbcg_solver_least_square(self._mat, self._BB_1, self._BX_1, 6, self._tol, self._maxiter, 0)

    #    #line 3
    #    blbcg_solver_obj = BLBCG()
    #    self._final_x_blbcg_m6s2, self._final_r_blbcg_m6s2, self._residual_hist_blbcg_m6s2 = \
    #            blbcg_solver_obj.blbcg_solver_least_square(self._mat, self._BB_6, self._BX_6, 2, self._tol, self._maxiter, 0)

    #    #line 4
    #    blbcg_solver_obj = BLBCG()
    #    self._final_x_blbcg_m6s6, self._final_r_blbcg_m6s6, self._residual_hist_blbcg_m6s6 = \
    #            blbcg_solver_obj.blbcg_solver_least_square(self._mat, self._BB_6, self._BX_6, 6, self._tol, self._maxiter, 0)

    #    #line 5
    #    bcbcg_solver_obj = BCBCG()
    #    self._final_x_bcbcg_m1s2, self._final_r_bcbcg_m1s2, self._residual_hist_bcbcg_m1s2 = \
    #            bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB_1, self._BX_1, 2, self._tol, self._maxiter, 0)

    #    #line 6
    #    bcbcg_solver_obj = BCBCG()
    #    self._final_x_bcbcg_m1s6, self._final_r_bcbcg_m1s6, self._residual_hist_bcbcg_m1s6 = \
    #            bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB_1, self._BX_1, 6, self._tol, self._maxiter, 0)

    #    #line 7
    #    bcbcg_solver_obj = BCBCG()
    #    self._final_x_bcbcg_m6s2, self._final_r_bcbcg_m6s2, self._residual_hist_bcbcg_m6s2 = \
    #            bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB_6, self._BX_6, 2, self._tol, self._maxiter, 0)

    #    #line 8
    #    bcbcg_solver_obj = BCBCG()
    #    self._final_x_bcbcg_m6s6, self._final_r_bcbcg_m6s6, self._residual_hist_bcbcg_m6s6 = \
    #            bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB_6, self._BX_6, 6, self._tol, self._maxiter, 0)

    #    plot_worker = Presenter()
    #    residual_list = [self._residual_hist_blbcg_m1s2, self._residual_hist_blbcg_m1s6, \
    #                     self._residual_hist_blbcg_m6s2, self._residual_hist_blbcg_m6s6, \
    #                     self._residual_hist_bcbcg_m1s2, self._residual_hist_bcbcg_m1s6, \
    #                     self._residual_hist_bcbcg_m6s2, self._residual_hist_bcbcg_m6s6 ]

    #    legend_list = ["blbcg_m1s2", "blbcg_m1s6", "blbcg_m6s2", "blbcg_m6s6", "bcbcg_m1s2", "bcbcg_m1s6", "bcbcg_m6s2", "bcbcg_m6s6"]
    #    color_list = ["r","k","b","y","m","g", "m", "0.5"]
    #    #plot_worker.instant_plot_y_log10(residual_list, "Chem97ZtZ", "#iteration", "$\\frac{||x_1||}{||b_1||}$", legend_list, color_list)
    #    plot_worker.instant_plot_y_log10(residual_list, "test", "#iteration", "$\\mathbf{log_{10}\\frac{||x_1||}{||b_1||}}$", legend_list, color_list)

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

    #def _db_bcbcg_eigen_param(self):
    #    """ """
    #    self._BB_6  = np.random.random( ( self._mat.shape[0],6) )
    #    self._BX_6  = np.ones ( (self._mat.shape[1],6) )

    #    gerschgorin_estimator = GerschgorinCircleTheoremEigenvalueEstimator()
    #    max_eigenvalue, min_eigenvalue = gerschgorin_estimator.csr_mat_extreme_eigenvalue_estimation(self._mat)
    #    print("################", "max:",max_eigenvalue, " , min:", min_eigenvalue)

    #    bcbcg_solver_obj = BCBCG()
    #    self._final_x_bcbcg_eigenparam_m6s6, self._final_r_bcbcg_eigenparam_m6s6, self._residual_hist_bcbcg_eigenparam_m6s6 = \
    #            bcbcg_solver_obj.bcbcg_solver_least_square_eigen_param(self._mat, self._BB_6, self._BX_6, 6, self._tol, self._maxiter, 0, max_eigenvalue, min_eigenvalue)

    #    bcbcg_solver_obj = BCBCG()
    #    self._final_x_bcbcg_m6s6, self._final_r_bcbcg_m6s6, self._residual_hist_bcbcg_m6s6 = \
    #            bcbcg_solver_obj.bcbcg_solver_least_square(self._mat, self._BB_6, self._BX_6, 6, self._tol, self._maxiter, 0)

    #    plot_worker = Presenter()
    #    residual_list = [self._residual_hist_bcbcg_eigenparam_m6s6, self._residual_hist_bcbcg_m6s6]

    #    legend_list = ["bcbcg_eigenparam_m6s6", "bcbcg_m6s6"]
    #    color_list = ["r","k"]
    #    plot_worker.instant_plot_y_log10(residual_list, "test", "#iteration", "$\\mathbf{log_{10}\\frac{||x_1||}{||b_1||}}$", legend_list, color_list)

    #def _db_usage_scipy_eig(self):
    #   """ """
    #   eigs_vals, eigs_vecs = eigs(self._mat, k=6)
    #   print(eigs_vals)
    #   #eigs_vals, eigs_vecs = eigs(self._mat, k=6, which="LM")
    #   #eigs_vals, eigs_vecs = eigs(self._mat, k=6, tol=1e-3,which="SR")
    #   self._matdense = self._mat.todense()
    #   #print(self._matdense)
    #   eig_vals,eig_vecs = np.linalg.eigh(self._matdense)
    #   #eig_vals,eig_vecs = np.linalg.eig(self._matdense)
    #   #print(eig_vals[0], " , ", eig_vals[-1])
    #   print(eig_vals)
    #   #print(eig_vecs)

    #def _diff_eigen_estimation_test(self):
    #    """ Notice: change a huge into dense and use numpy.linalg.eigh is very dangerous, you computer may be freezing forever"""
    #    gerschgorin_estimator = GerschgorinCircleTheoremEigenvalueEstimator()
    #    gerschgorin_max_eigenvalue, gerschgorin_min_eigenvalue = gerschgorin_estimator.csr_mat_extreme_eigenvalue_estimation(self._mat)
    #    print("################Gershchogrin theorem", "max:",gerschgorin_max_eigenvalue, " , old min:", gerschgorin_min_eigenvalue)
    #    if gerschgorin_min_eigenvalue< 0.:
    #        gerschgorin_min_eigenvalue = 0.
    #    print("################Gershchogrin theorem", "max:",gerschgorin_max_eigenvalue, " , new min:", gerschgorin_min_eigenvalue)

    #    power_method_solver = PowerIteration()
    #    self._init_eigen_vec  = np.random.random( ( self._mat.shape[0],1) )
    #    pm_maxiters = 1000
    #    pm_tol = 1e-6
    #    pm_max_eigen_vec, pm_max_eigen_val, pm_max_eigen_list = power_method_solver.naive_power_iteration (self._mat, self._init_eigen_vec, pm_maxiters, pm_tol)
    #    print("################Power method max:",pm_max_eigen_val, " , iteration:", len(pm_max_eigen_list))
    #    self._init_eigen_vec  = np.random.random( ( self._mat.shape[0],1) )
    #    pm_min_eigen_vec, pm_min_eigen_val, pm_min_eigen_list = power_method_solver.power_iteration_with_shifting_acc1 (self._mat, self._init_eigen_vec, pm_max_eigen_val, pm_maxiters, pm_tol)
    #    pm_min_eigen_val = pm_min_eigen_val + pm_max_eigen_val
    #    print("################Power method min:",pm_min_eigen_val, " , iteration:", len(pm_min_eigen_list))

    #    numpy_eigh_eigen_vals, numpy_eigh_eigen_vecs = np.linalg.eigh(self._mat.todense())
    #    assert numpy_eigh_eigen_vals[0]<numpy_eigh_eigen_vals[-1]
    #    print("################Numpy.linalg.eigh", "max:",numpy_eigh_eigen_vals[-1], " , min:", numpy_eigh_eigen_vals[0])

    #    ##
    #    eigen_repo = {"numpy_eigh":(numpy_eigh_eigen_vals[-1],numpy_eigh_eigen_vals[0]), \
    #                  "gerschgorin":(gerschgorin_max_eigenvalue, gerschgorin_min_eigenvalue),\
    #                  "power_method":(pm_max_eigen_val,pm_min_eigen_val), \
    #                  "mix_method":(pm_max_eigen_val,gerschgorin_min_eigenvalue) \
    #                 }
    #    print(type(eigen_repo))
    #    print(eigen_repo["numpy_eigh"], "," ,eigen_repo["gerschgorin"], " , ", eigen_repo["power_method"])
    #    print("max: ", eigen_repo["numpy_eigh"][0], "," ,eigen_repo["gerschgorin"][0], " , ", eigen_repo["power_method"][0])

    #    self._BB  = np.random.random( ( self._mat.shape[0],4) )
    #    self._BX  = np.ones ( (self._mat.shape[1],4) )

    #    #line 1
    #    bcbcg_solver_obj = BCBCG()
    #    self._final_np_x_bcbcg_m4s4, self._final_np_r_bcbcg_m4s4, self._np_residual_hist_bcbcg_m4s4 = \
    #            bcbcg_solver_obj.bcbcg_solver_least_square_eigen_param(self._mat, self._BB, self._BX, 16, self._tol, self._maxiter, 0, eigen_repo["numpy_eigh"][0], eigen_repo["numpy_eigh"][1])
    #    #line 2
    #    bcbcg_solver_obj = BCBCG()
    #    self._final_gerschgorin_x_bcbcg_m4s4, self._final_gerschgorin_r_bcbcg_m4s4, self._gerschgorin_residual_hist_bcbcg_m4s4 = \
    #            bcbcg_solver_obj.bcbcg_solver_least_square_eigen_param(self._mat, self._BB, self._BX, 16, self._tol, self._maxiter, 0, eigen_repo["gerschgorin"][0], eigen_repo["gerschgorin"][1])
    #    #line 3
    #    bcbcg_solver_obj = BCBCG()
    #    self._final_pm_x_bcbcg_m4s4, self._final_pm_r_bcbcg_m4s4, self._pm_residual_hist_bcbcg_m4s4 = \
    #            bcbcg_solver_obj.bcbcg_solver_least_square_eigen_param(self._mat, self._BB, self._BX, 16, self._tol, self._maxiter, 0, eigen_repo["power_method"][0], eigen_repo["power_method"][1])

    #    #line 4
    #    bcbcg_solver_obj = BCBCG()
    #    self._final_mix_x_bcbcg_m4s4, self._final_mix_r_bcbcg_m4s4, self._mix_residual_hist_bcbcg_m4s4 = \
    #            bcbcg_solver_obj.bcbcg_solver_least_square_eigen_param(self._mat, self._BB, self._BX, 16, self._tol, self._maxiter, 0, eigen_repo["mix_method"][0], eigen_repo["mix_method"][1])

    #    plot_worker = Presenter()
    #    residual_list = [self._np_residual_hist_bcbcg_m4s4, self._gerschgorin_residual_hist_bcbcg_m4s4, self._pm_residual_hist_bcbcg_m4s4,self._mix_residual_hist_bcbcg_m4s4 ]

    #    legend_list = ["N_bcbcg_m4s4", "G_bcbcg_m4s4","P_bcbcg_m4s4","M_bcbcg_m4s4" ]
    #    color_list = ["r","k","b", "y"]
    #    plot_worker.instant_plot_y_log10(residual_list, "Chem97ZtZ", "#iteration", "$\\mathbf{log_{10}\\frac{||x_1||}{||b_1||}}$", legend_list, color_list)

    #def _diff_eigen_estimation_test_b(self):
    #    """ """
    #    print("_diff_eigen_estimation_test_b starting ...")
    #    gerschgorin_estimator = GerschgorinCircleTheoremEigenvalueEstimator()
    #    gerschgorin_max_eigenvalue, gerschgorin_min_eigenvalue = gerschgorin_estimator.csr_mat_extreme_eigenvalue_estimation(self._mat)
    #    print("################Gershchogrin theorem", "min:",gerschgorin_max_eigenvalue, " , old min:", gerschgorin_min_eigenvalue)
    #    if gerschgorin_min_eigenvalue< 0.:
    #        gerschgorin_min_eigenvalue = 0.
    #    print("################Gershchogrin theorem", "min:",gerschgorin_max_eigenvalue, " , new min:", gerschgorin_min_eigenvalue)

    #    eigs_vals, eigs_vecs = eigs(self._mat, k=6, which="LM")
    #    print("eigs_vals max", eigs_vals)
    #    eigs_vals, eigs_vecs = eigs(self._mat, k=6, tol=1e-3,which="SR")
    #    print("eigs_vals min", eigs_vals)

    #    power_method_solver = PowerIteration()
    #    self._init_eigen_vec  = np.random.random( ( self._mat.shape[0],1) )
    #    pm_maxiters = 300
    #    pm_tol = 1e-6
    #    pm_max_eigen_vec, pm_max_eigen_val, pm_max_eigen_list = power_method_solver.naive_power_iteration (self._mat, self._init_eigen_vec, pm_maxiters, pm_tol)
    #    print("################Power method max:",pm_max_eigen_val, " , iteration:", len(pm_max_eigen_list))
    #    self._init_eigen_vec  = np.random.random( ( self._mat.shape[0],1) )
    #    pm_min_eigen_vec, pm_min_eigen_val, pm_min_eigen_list = power_method_solver.power_iteration_with_shifting_acc1 (self._mat, self._init_eigen_vec, pm_max_eigen_val, pm_maxiters, pm_tol)
    #    pm_min_eigen_val = pm_min_eigen_val + pm_max_eigen_val
    #    print("################Power method min:",pm_min_eigen_val, " , iteration:", len(pm_min_eigen_list))


    #    ##
    #    eigen_repo = {"gerschgorin":(gerschgorin_max_eigenvalue, gerschgorin_min_eigenvalue),\
    #                  "power_method":(pm_max_eigen_val,pm_min_eigen_val), \
    #                  "mix_method":(pm_max_eigen_val,gerschgorin_min_eigenvalue) \
    #                 }
    #    #eigen_repo = {"gerschgorin":(gerschgorin_max_eigenvalue, gerschgorin_min_eigenvalue),\
    #    #              "power_method":(pm_max_eigen_val,gerschgorin_min_eigenvalue), \
    #    #              "mix_method":(pm_max_eigen_val,gerschgorin_min_eigenvalue) \
    #    #             }
    #    print(eigen_repo["gerschgorin"], " , ", eigen_repo["power_method"], " , ",eigen_repo["mix_method"])

    #    self._BB  = np.random.random( ( self._mat.shape[0],4) )
    #    self._BX  = np.ones ( (self._mat.shape[1],4) )

    #    #line 2
    #    bcbcg_solver_obj = BCBCG()
    #    self._final_gerschgorin_x_bcbcg_m4s4, self._final_gerschgorin_r_bcbcg_m4s4, self._gerschgorin_residual_hist_bcbcg_m4s4 = \
    #            bcbcg_solver_obj.bcbcg_solver_least_square_eigen_param(self._mat, self._BB, self._BX, 16, self._tol, self._maxiter, 0, eigen_repo["gerschgorin"][0], eigen_repo["gerschgorin"][1])
    #    #line 3
    #    bcbcg_solver_obj = BCBCG()
    #    self._final_pm_x_bcbcg_m4s4, self._final_pm_r_bcbcg_m4s4, self._pm_residual_hist_bcbcg_m4s4 = \
    #            bcbcg_solver_obj.bcbcg_solver_least_square_eigen_param(self._mat, self._BB, self._BX, 16, self._tol, self._maxiter, 0, eigen_repo["power_method"][0], eigen_repo["power_method"][1])

    #    #line 4
    #    bcbcg_solver_obj = BCBCG()
    #    self._final_mix_x_bcbcg_m4s4, self._final_mix_r_bcbcg_m4s4, self._mix_residual_hist_bcbcg_m4s4 = \
    #            bcbcg_solver_obj.bcbcg_solver_least_square_eigen_param(self._mat, self._BB, self._BX, 16, self._tol, self._maxiter, 0, eigen_repo["mix_method"][0], eigen_repo["mix_method"][1])

    #    plot_worker = Presenter()
    #    residual_list = [self._gerschgorin_residual_hist_bcbcg_m4s4, self._pm_residual_hist_bcbcg_m4s4,self._mix_residual_hist_bcbcg_m4s4 ]

    #    legend_list = ["G_bcbcg_m4s4","P_bcbcg_m4s4","M_bcbcg_m4s4" ]
    #    color_list = ["r","k","b"]
    #    plot_worker.instant_plot_y_log10(residual_list, "Chem97ZtZ", "#iteration", "$\\mathbf{log_{10}\\frac{||x_1||}{||b_1||}}$", legend_list, color_list)

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


    #def _db_power_iteration_with_shifting_acc1(self):
    #    """ """
    #    print("_db_power_iteration_with_shifting_acc1 starting ....")
    #    #print(self._mat)
    #    #print(self._mat.diagonal())
    #    #shift_factor = 100.
    #    #print(self._mat.diagonal() + shift_factor)
    #    #self._mat.setdiag(self._mat.diagonal() + shift_factor)
    #    #print("new dia")
    #    #print(self._mat.diagonal())
    #    #print(self._mat)

    #    pm_max_eigen_val = -100
    #    pm_maxiters = 10
    #    pm_tol = 1e-12
    #    power_method_solver = PowerIteration()

    #    self._init_eigen_vec_1  = np.random.random( ( self._mat.shape[0],1) )
    #    self._init_eigen_vec_2  = self._init_eigen_vec_1.copy()
    #    pm_min_eigen_vec, pm_min_eigen_val, pm_min_eigen_list = power_method_solver.power_iteration_with_shifting_acc1 (self._mat, self._init_eigen_vec_1, pm_max_eigen_val, pm_maxiters, pm_tol)
    #    print("###########")
    #    pm_min_eigen_vec, pm_min_eigen_val, pm_min_eigen_list = power_method_solver.power_iteration_with_shifting (self._mat, self._init_eigen_vec_2, pm_max_eigen_val, pm_maxiters, pm_tol)


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

    block_size = 4 
    tol = 1e-10
    maxiter = 1500
    step_val =64

    linear_system_solver_worker_test = WorkerIterativeLinearSystemSolverCG_Exp_160606_isc16(mat_path)
    linear_system_solver_worker_test.conduct_experiments(block_size,tol,maxiter, step_val)
    #linear_system_solver_worker_test.chebyshev_poly_exp_a(0,6)
    #linear_system_solver_worker_test.legendre_poly_exp_a(0,6)
    #linear_system_solver_worker_test.debug_NativeConjugateGradient()



if __name__ == "__main__":
    """ call main funtion for testing """
    main()
