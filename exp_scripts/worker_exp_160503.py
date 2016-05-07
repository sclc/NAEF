""" 
Experiment Diary 2016-05-03
"""
import sys
from scipy import io
import numpy as np
from scipy.sparse.linalg import *
sys.path.append("../src/") 
from worker import Worker
from native_conjugate_gradient import NativeConjugateGradient

class WorkerIterativeLinearSystemSolverCG_Exp_160503_A(Worker):
    """ Description: Experiment A
        Numerical Method: Naive Conjugate Gradient
        tol:
        max_iteration:
        matrix:
        Reference:
            1. 
    """
    def __init__(self, mat_path, tol, max_iteration):
        """ """
        #print ("WorkerIterativeLinearSystemSolver works good")
        Worker.__init__(self)
        self._hist_list = []

        if mat_path == "":
            """ Need to generatre matrix """
            print("calling self._matrix_generation")
            #self._mat = self._matrix_generation()
        else:
            self._mat = io.mmread(mat_path)
            print("Done reading matrix {}, Row:{}, Col:{}".format( mat_path, self._mat.shape[0], self._mat.shape[1]))

        self._tol = tol
        self._max_iteration = max_iteration

    def _matrix_generator(self):
        """ generation of matrix """
        print("_matrix_generator")


    def _setup_testbed(self):
        """ this can considered as a basic experiment input descripting """
        #self._B = np.random.random( ( self._mat.shape[0],1) )
        self._B = np.ones( ( self._mat.shape[0],1) )
        self._X = np.ones ( (self._mat.shape[1],1) ) 
        #self._X = np.zeros ( (self._mat.shape[1],1) ) 

    def _setup_numerical_algorithm(self):
        """ After a linear solver or other numerical methods loaded
            we need to setup the basic prarm for the algorithm
        """
        self._numerical_method = NativeConjugateGradient(self._mat, self._X, \
                                self._B, self._tol, self._max_iteration);

    def conduct_experiments(self):
        """ function to condution the experiment """
        print("to conduct the experient")
        self._setup_testbed()
        self._setup_numerical_algorithm( )
        #print ("before:{}".format(np.inner(self._X[:,0], self._X[:,0])))
        self._hist_list = self._numerical_method.cg_variant_one_run()
        #print (self._hist_list)
        #print ("after:{}".format(np.inner(self._X[:,0], self._X[:,0])))
        print("Experiments done")
        #print("starting scipy.sparse.linalg.cg call ... ...")

    def debug_NativeConjugateGradient(self):
        self._X_test = np.ones ( (self._mat.shape[1],1) ) 
        self._X_test_res, self._X_test_info = cg(self._mat, self._B, self._X_test, maxiter=20)
        print ("my result:{}".format(np.inner(self._X[:,0], self._X[:,0])))

        print ("scipy result:{}".format(np.inner(self._X_test_res, self._X_test_res)))





def main ():
# main function for today's experiments 
    mat_path = "/home/scl/MStore/nasa2146/nasa2146.mtx"
    linear_system_solver_worker_test = WorkerIterativeLinearSystemSolverCG_Exp_160503_A(mat_path, 1e-5, 19)
    linear_system_solver_worker_test.conduct_experiments()
    linear_system_solver_worker_test.debug_NativeConjugateGradient()



if __name__ == "__main__":
    """ call main funtion for testing """
    main()
