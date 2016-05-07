""" Gerschgorin """
import numpy as np

class GerschgorinCircleTheoremEigenvalueEstimator ():
    """ Gerschgorin Basic Class"""

    def csr_mat_extreme_eigenvalue_estimation (self,mat):
        """ do esitimation work"""
        assert mat.getformat() == "csr", "You give a matrix of format {} rather than csr".format(mat.getformat())
        num_rows = mat.shape[0]
        num_cols = mat.shape[1]
        v_diagonal = mat.diagonal()
        min_eigenvalue = np.float64("inf")
        max_eigenvalue = np.float64("-inf")

        for  iptr in range(0,len(mat.indptr)-1):
            row_abs_sum_exclusive_dia_temp = 0.0
            for idx in range(mat.indptr[iptr], mat.indptr[iptr+1]):
                if iptr != mat.indices[idx]:
                    row_abs_sum_exclusive_dia_temp += abs(mat.data[idx])

            min_temp =  v_diagonal[iptr] - row_abs_sum_exclusive_dia_temp
            if min_eigenvalue > min_temp:
                min_eigenvalue = min_temp
            max_temp = v_diagonal[iptr] + row_abs_sum_exclusive_dia_temp
            if max_eigenvalue <  max_temp:
                max_eigenvalue = max_temp

        return max_eigenvalue, min_eigenvalue
