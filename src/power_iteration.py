""" Power itertion and its variants"""
import numpy as np
from scipy.sparse.linalg import *

class PowerIteration():
    """ """
    def __init__ (self):
        pass

    def naive_power_iteration (self, mat, init_x, maxiters, tol):
        """ naive power iteration method"""
        lambda_eigen_list=[]
        op_spmv = aslinearoperator(mat)
        v_eigen = init_x/ np.linalg.norm(init_x)
        num_iter = 0
        v_new = op_spmv(v_eigen)
        lambda_eigen = np.inner(v_eigen[:,0], v_new[:,0])
        lambda_eigen_list.append(lambda_eigen)
        v_eigen = v_new / np.linalg.norm(v_new)

        for idx in range(1,maxiters):
            lambda_eigen_old = lambda_eigen
            v_new = op_spmv(v_eigen)
            lambda_eigen = np.inner(v_eigen[:,0], v_new[:,0])
            lambda_eigen_list.append(lambda_eigen)
            #print("new lambda_eigen", lambda_eigen)
            v_eigen = v_new / np.linalg.norm(v_new)

            if (abs(lambda_eigen-lambda_eigen_old)/abs(lambda_eigen_old) < tol).all():
                return v_eigen, lambda_eigen, lambda_eigen_list


        return v_eigen, lambda_eigen, lambda_eigen_list

    def power_iteration_with_shifting (self, mat, init_x, shifting_val, maxiters, tol):
        """ """
        lambda_eigen_list=[]
        mat_temp = mat.copy()
        mat_temp = mat_temp - shifting_val * np.eye(mat.shape[0], mat.shape[1])

        op_spmv = aslinearoperator(mat_temp)
        v_eigen = init_x/ np.linalg.norm(init_x)
        num_iter = 0
        v_new = op_spmv(v_eigen)
        lambda_eigen = np.inner(v_eigen[:,0], v_new[:,0])
        lambda_eigen_list.append(lambda_eigen)
        v_eigen = v_new / np.linalg.norm(v_new)

        for idx in range(1,maxiters):
            lambda_eigen_old = lambda_eigen
            v_new = op_spmv(v_eigen)
            lambda_eigen = np.inner(v_eigen[:,0], v_new[:,0])
            lambda_eigen_list.append(lambda_eigen)
            print("new lambda_eigen", lambda_eigen)
            v_eigen = v_new / np.linalg.norm(v_new)

            if (abs(lambda_eigen-lambda_eigen_old)/abs(lambda_eigen_old) < tol).all():
                return v_eigen, lambda_eigen, lambda_eigen_list


        return v_eigen, lambda_eigen, lambda_eigen_list



def main():
    pass

if __name__ == "__main__":
    main()
