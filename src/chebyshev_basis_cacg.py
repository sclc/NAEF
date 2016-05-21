"""CBCG method """
from chebyshev_polynomial import ChebyshevPolynomial
from gerschgorin_circle_theorem import GerschgorinCircleTheoremEigenvalueEstimator
import numpy as np
from scipy.sparse import linalg

class CBCG():
    """ """
    def __init__(self):
        pass

    def cbcg_solver(self, mat, rhs, init_x, step_val, tol, maxiter):
        gerschgorin_estimator = GerschgorinCircleTheoremEigenvalueEstimator()
        max_eigenvalue, min_eigenvalue = gerschgorin_estimator.csr_mat_extreme_eigenvalue_estimation(mat)
        chebyshev_basis_generator = ChebyshevPolynomial()

        op_A = linalg.aslinearoperator(mat)

        v_r = rhs - op_A(init_x)
        v_x = init_x.copy()

        s_normb = np.linalg.norm(rhs)
        residual_ratio_hist  = [np.linalg.norm(v_r)/s_normb]

        for itercounter in range(1, maxiter+1):
            m_chebyshev_basis = \
                    chebyshev_basis_generator.basis_generation_with_eigenvalue_shifting_and_scaling_single_vec(\
                    mat, v_r, step_val, max_eigenvalue, min_eigenvalue)

            if itercounter == 1:
                m_Q = m_chebyshev_basis
            else:
                #op_AQ_trans = linalg.aslinearoperator(m_AQ.transpose())
                #AQ_trans_mul_chebyshev_basis = op_AQ_trans.matmat(m_chebyshev_basis)
                #m_B = linalg.aslinearoperator(Q_trans_AQ_inverse).matmat (AQ_trans_mul_chebyshev_basis)
                m_AQ_trans_mul_chebyshev_basis = np.matmul(m_AQ.T , m_chebyshev_basis)
                m_B = np.matmul(m_Q_trans_AQ_inverse , m_AQ_trans_mul_chebyshev_basis)
                m_Q = m_chebyshev_basis - np.matmul(m_Q, m_B)

            m_AQ = op_A.matmat(m_Q)
            #m_Q_trans_AQ = linalg.aslinearoperator(m_Q.transpose())(m_AQ)
            m_Q_trans_AQ = np.matmul(m_Q.T, m_AQ)
            m_Q_trans_AQ_inverse = np.linalg.inv(m_Q_trans_AQ)
            v_alpha = np.matmul( m_Q_trans_AQ_inverse, np.matmul(m_Q.T, v_r) )

            v_x += np.matmul(m_Q,v_alpha)
            v_r -= np.matmul(m_AQ, v_alpha)

            residual_ratio_hist.append( np.linalg.norm(v_r)/s_normb)
            print(itercounter, ": ", np.linalg.norm(v_r)/s_normb)
            if residual_ratio_hist[-1] <= tol:
                return v_x, v_r, residual_ratio_hist

        return v_x, v_r, residual_ratio_hist

    def cbcg_solver_least_square(self, mat, rhs, init_x, step_val, tol, maxiter):
        """get the inverse matrix by least square method"""
        gerschgorin_estimator = GerschgorinCircleTheoremEigenvalueEstimator()
        max_eigenvalue, min_eigenvalue = gerschgorin_estimator.csr_mat_extreme_eigenvalue_estimation(mat)
        chebyshev_basis_generator = ChebyshevPolynomial()

        op_A = linalg.aslinearoperator(mat)

        v_r = rhs - op_A(init_x)
        v_x = init_x.copy()

        s_normb = np.linalg.norm(rhs)
        residual_ratio_hist  = [np.linalg.norm(v_r)/s_normb]

        for itercounter in range(1, maxiter+1):
            m_chebyshev_basis = \
                    chebyshev_basis_generator.basis_generation_with_eigenvalue_shifting_and_scaling_single_vec(\
                    mat, v_r, step_val, max_eigenvalue, min_eigenvalue)

            if itercounter == 1:
                m_Q = m_chebyshev_basis
            else:
                #op_AQ_trans = linalg.aslinearoperator(m_AQ.transpose())
                #AQ_trans_mul_chebyshev_basis = op_AQ_trans.matmat(m_chebyshev_basis)
                #m_B = linalg.aslinearoperator(Q_trans_AQ_inverse).matmat (AQ_trans_mul_chebyshev_basis)
                m_AQ_trans_mul_chebyshev_basis = np.matmul(m_AQ.T , m_chebyshev_basis)

                #m_B = np.matmul(m_Q_trans_AQ_inverse , m_AQ_trans_mul_chebyshev_basis)
                m_B = np.linalg.lstsq(m_Q_trans_AQ, m_AQ_trans_mul_chebyshev_basis)[0]

                m_Q = m_chebyshev_basis - np.matmul(m_Q, m_B)

            m_AQ = op_A.matmat(m_Q)
            #m_Q_trans_AQ = linalg.aslinearoperator(m_Q.transpose())(m_AQ)
            m_Q_trans_AQ = np.matmul(m_Q.T, m_AQ)

            #m_Q_trans_AQ_inverse = np.linalg.inv(m_Q_trans_AQ)
            #v_alpha = np.matmul( m_Q_trans_AQ_inverse, np.matmul(m_Q.T, v_r) )
            v_alpha = np.linalg.lstsq(m_Q_trans_AQ, np.matmul(m_Q.T, v_r))[0]

            v_x += np.matmul(m_Q,v_alpha)
            v_r -= np.matmul(m_AQ, v_alpha)

            residual_ratio_hist.append( np.linalg.norm(v_r)/s_normb)
            print(itercounter, ": ", np.linalg.norm(v_r)/s_normb)
            if residual_ratio_hist[-1] <= tol:
                return v_x, v_r, residual_ratio_hist

        return v_x, v_r, residual_ratio_hist


class BCBCG():
    """ """
    def __init__(self):
        pass

    def bcbcg_solver(self, mat, RHS, init_X, step_val, tol, maxiter,whichcol):
        gerschgorin_estimator = GerschgorinCircleTheoremEigenvalueEstimator()
        max_eigenvalue, min_eigenvalue = gerschgorin_estimator.csr_mat_extreme_eigenvalue_estimation(mat)
        chebyshev_basis_generator = ChebyshevPolynomial()

        op_A = linalg.aslinearoperator(mat)

        m_R = RHS - op_A(init_X)
        m_X = init_X.copy()

        R_to_RHS_norm_ratio = lambda x: np.linalg.norm(m_R[:,x])/np.linalg.norm(RHS[:,x])
        residual_ratio_hist  = [R_to_RHS_norm_ratio(whichcol)]

        for itercounter in range(1, maxiter+1):
            m_chebyshev_basis = \
                    chebyshev_basis_generator.basis_generation_with_eigenvalue_shifting_and_scaling_block_vecs(\
                    mat, m_R, step_val, max_eigenvalue, min_eigenvalue)

            #print("basis rank",np.linalg.matrix_rank(m_chebyshev_basis))
            #return

            if itercounter == 1:
                m_Q = m_chebyshev_basis
            else:
                m_AQ_trans_mul_chebyshev_basis = np.matmul(m_AQ.T , m_chebyshev_basis)
                m_B = np.matmul(m_Q_trans_AQ_inverse , m_AQ_trans_mul_chebyshev_basis)
                m_Q = m_chebyshev_basis - np.matmul(m_Q, m_B)


            m_AQ = op_A.matmat(m_Q)
            m_Q_trans_AQ = np.matmul(m_Q.T, m_AQ)

            m_Q_trans_AQ_inverse = np.linalg.inv(m_Q_trans_AQ)
            m_alpha = np.matmul( m_Q_trans_AQ_inverse, np.matmul(m_Q.T, m_R) )

            m_X += np.matmul(m_Q,m_alpha)
            m_R -= np.matmul(m_AQ, m_alpha)

            residual_ratio_hist.append(R_to_RHS_norm_ratio(whichcol))

            print(itercounter, ": ", R_to_RHS_norm_ratio(whichcol))
            if residual_ratio_hist[-1] <= tol:
                return m_X, m_R, residual_ratio_hist

        return m_X, m_R, residual_ratio_hist




