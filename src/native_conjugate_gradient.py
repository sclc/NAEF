""" """
import scipy
import numpy as np
from scipy.sparse.linalg import *

class NativeConjugateGradient(object):
    """ """
    def __init__(self, mat, X, B, tol, max_iteration):
        """ """ 
        self._mat=mat
        self._X=X
        self._B=B
        self._tol=tol
        self._max_iteration = max_iteration
        self._final_iterations = 0

    def cg_variant_one_run(self):
        """ run cg variant here,
            return convergence history
            s_: sclar
            v_: vector
        """
        hist_list = []
        mat_temp = aslinearoperator(self._mat)
        v_b_temp = self._B
        v_x_temp = self._X

        v_Ax = mat_temp.matvec(v_x_temp)
        v_p = v_r = v_b_temp - v_Ax

        s_rTr = np.inner(v_r[:,0], v_r[:,0])
        hist_list.append(np.sqrt(s_rTr))
        iter_counter = 0

        while (s_rTr>=self._tol).all() and (iter_counter <= self._max_iteration):
            v_Ap      = mat_temp.matvec(v_p)
            #ptp_old_db    = np.inner(v_p[:,0], v_p[:,0])
            #print ("v_Ap shape:{}".format(v_Ap.shape))
            s_ptAp    = np.inner (v_Ap[:,0], v_p[:,0])
            #print("s_ptAP:{}, s_rTr:{}".format(s_ptAp, s_rTr))
            s_alpha   = s_rTr / s_ptAp
            #print("alpha:{}".format(s_alpha))

            v_x_temp += s_alpha * v_p
            #print("v_x_temp:{}".format( np.inner(v_x_temp[:,0],v_x_temp[:,0])))
            v_r      -= s_alpha * v_Ap

            s_rTr_new = np.inner(v_r[:,0],v_r[:,0])
            s_beta    = s_rTr_new /s_rTr
            #print("beta:{}".format(s_beta))
            v_p       = v_r + s_beta * v_p
            s_rTr     = s_rTr_new
            hist_list.append(np.sqrt(s_rTr))
            #print("v_p inner product:{}".format(np.inner(v_p[:,0], v_p[:,0])))
            #print("s_ptAp:{}, s_alpha:{}, s_beta:{}, ptp_old:{}, dot(new v_p):{}, s_rTr_new:{}".format(s_ptAp, s_alpha, s_beta, ptp_old_db,np.inner(v_p[:,0],v_p[:,0]), s_rTr ))

            iter_counter += 1

        self._X = v_x_temp
        self._final_iterations = iter_counter
        return hist_list


def main():
    print("The NativeConjugateGradient method runs")

if  __name__ == "__main__":
    main()
