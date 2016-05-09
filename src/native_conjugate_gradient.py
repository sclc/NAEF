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

        """ v_p = v_r = v_b_temp - v_Ax
            is a dangerous bug, v_p and v_r will be alias of a same memory
        """
        v_p = v_b_temp - v_Ax
        v_r = v_p.copy()

        s_rTr = np.dot(v_r[:,0], v_r[:,0])
        hist_list.append(np.sqrt(s_rTr))
        iter_counter = 0

        while (s_rTr>=self._tol).all() and (iter_counter <= self._max_iteration):
            v_Ap      = mat_temp.matvec(v_p)
            #ptp_old_db    = np.inner(v_p[:,0], v_p[:,0])
            #print ("v_Ap shape:{}".format(v_Ap.shape))
            s_ptAp    = np.dot (v_Ap[:,0], v_p[:,0])
            #print("s_ptAP:{}, s_rTr:{}".format(s_ptAp, s_rTr))
            s_alpha   = s_rTr / s_ptAp
            #print("alpha:{}".format(s_alpha))

            v_x_temp += s_alpha * v_p
            #print("v_x_temp:{}".format( np.inner(v_x_temp[:,0],v_x_temp[:,0])))
            v_r      -= s_alpha * v_Ap

            s_rTr_new = np.dot(v_r[:,0],v_r[:,0])
            s_beta    = s_rTr_new /s_rTr
            #print("beta:{}".format(s_beta))
            if (v_p == v_r).all():
                print("this is wrong")
            v_p       = v_r + s_beta * v_p
            s_rTr     = s_rTr_new
            hist_list.append(np.sqrt(s_rTr))
            #print("v_p inner product:{}".format(np.inner(v_p[:,0], v_p[:,0])))
            #print("s_ptAp:{}, s_alpha:{}, s_beta:{}, ptp_old:{}, dot(new v_p):{}, s_rTr_new:{}".format(s_ptAp, s_alpha, s_beta, ptp_old_db,np.inner(v_p[:,0],v_p[:,0]), s_rTr ))

            iter_counter += 1

        #self._X = v_x_temp
        self._final_iterations = iter_counter
        return hist_list


class NativeBlockConjugateGradient(object):
    """ """
    def __init__(self, mat, initX, B, tol, max_iteration):
        """ """ 
        self._mat=mat
        self._initX=initX
        self._B=B
        self._blocksize = self._B.shape[1]
        self._tol=tol
        self._max_iteration = max_iteration
        self._final_iterations = 0

    def bcg_variant_one_run(self, whichcol):
        """ run cg variant here,
            return convergence history
            s_: sclar
            v_: vector
        """
        hist_list = []
        op_matmatmul = aslinearoperator(self._mat)
        m_B = self._B
        m_initX = self._initX
        m_X = self._initX.copy()

        m_AX= op_matmatmul(m_initX)

        """ v_p = v_r = v_b_temp - v_Ax
            is a dangerous bug, v_p and v_r will be alias of a same memory
        """
        m_R = m_B - m_AX
        m_P = m_R.copy()

        m_RtR = np.matmul(m_R.T, m_R)
        vec_norm_ratio_cal = lambda x: np.linalg.norm(m_R[:,x])/np.linalg.norm(m_B[:,x])
        hist_list.append( vec_norm_ratio_cal(whichcol) )
        iter_counter = 0

        while hist_list[-1]>self._tol and iter_counter <= self._max_iteration:
            m_AP      = op_matmatmul(m_P)
            m_PtAP    = np.matmul (m_P.T, m_AP)
            m_alpha   = np.matmul( np.linalg.inv(m_PtAP), m_RtR )

            m_X += np.matmul(m_P, m_alpha)
            m_R -= np.matmul( m_AP, m_alpha)

            m_RtR_new = np.matmul(m_R.T, m_R)
            m_beta    = np.matmul( np.linalg.inv(m_RtR), m_RtR_new)
            m_P       = m_R + np.matmul(m_P, m_beta)
            m_RtR     = m_RtR_new
            hist_list.append( vec_norm_ratio_cal(whichcol) )

            iter_counter += 1

        self._final_iterations = iter_counter
        return m_X, m_R, hist_list

def main():
    print("The NativeConjugateGradient method runs")

if  __name__ == "__main__":
    main()
