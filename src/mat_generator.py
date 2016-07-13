""" """
import numpy as np
#from scipy.sparse import coo_matrix
from scipy.sparse import dia_matrix
from scipy.sparse import eye 

class MatGenerator():
    """ """
    def __init__(self):
        """ """
        print("I am going to generate a particular by features specified by you")


    def dia_mat_band_random_square (self, dim, left_semi_bw, right_semi_bw, val_min, val_max):
        """ """
        assert val_max > val_min
        val_range = val_max - val_min

        dia_mat_nnz  = np.random.random( ( left_semi_bw + right_semi_bw + 1)*dim )
        dia_mat_nnz = val_min + val_range * dia_mat_nnz
        #print(dia_mat_nnz)
        dia_mat_nnz = dia_mat_nnz.reshape( (left_semi_bw + right_semi_bw + 1, dim) )
        #print(dia_mat_nnz)


        left_dia_array = np.arange(-left_semi_bw, 0)
        main_dia_array = np.array([0])
        right_dia_array = np.arange(1, right_semi_bw+1)

        dia_offsets = np.append(left_dia_array, right_dia_array)
        dia_offsets = np.append(main_dia_array, dia_offsets)
        print(dia_offsets)

        return dia_matrix((dia_mat_nnz, dia_offsets), shape=(dim,dim))

    def dia_mat_band_random_square_PDpossible (self, dim, left_semi_bw, right_semi_bw, val_min, val_max, factor_diag):
        """ """
        assert val_max > val_min
        val_range = val_max - val_min

        dia_mat_nnz  = np.random.random( ( left_semi_bw + right_semi_bw + 1)*dim )
        dia_mat_nnz = val_min + val_range * dia_mat_nnz
        #print(dia_mat_nnz)
        dia_mat_nnz = dia_mat_nnz.reshape( (left_semi_bw + right_semi_bw + 1, dim) )
        #print(dia_mat_nnz)


        left_dia_array = np.arange(-left_semi_bw, 0)
        main_dia_array = np.array([0])
        right_dia_array = np.arange(1, right_semi_bw+1)

        dia_offsets = np.append(left_dia_array, right_dia_array)
        dia_offsets = np.append(main_dia_array, dia_offsets)
        print(dia_offsets)

        mat_dia = dia_matrix((dia_mat_nnz, dia_offsets), shape=(dim,dim))
        #print(mat_dia.todense())

        return mat_dia + factor_diag * eye(dim)

    def dia_mat_band_random_square_sym (self, dim, semi_bw, val_min, val_max):
        """ """
        assert val_max > val_min
        assert semi_bw < dim
        val_range = val_max - val_min

        dia_mat_nnz  = np.random.random( ( semi_bw + 1)*dim )
        dia_mat_nnz = val_min + val_range * dia_mat_nnz
        #print(dia_mat_nnz)
        dia_mat_nnz = dia_mat_nnz.reshape( (semi_bw + 1, dim) )
        #print(dia_mat_nnz)

        left_dia_array = np.arange(-semi_bw, 0)
        main_dia_array = np.array([0])
        dia_offsets = np.append(main_dia_array, left_dia_array)
        print(dia_offsets)

        mat_dia =  dia_matrix((dia_mat_nnz, dia_offsets), shape=(dim,dim))
        return mat_dia + mat_dia.transpose() - dia_matrix( (mat_dia.diagonal(), ([0])) , shape=(dim,dim) )

    def dia_mat_band_random_square_sym_PDpossible (self, dim, semi_bw, val_min, val_max, factor_diag):
        """ """
        assert val_max > val_min
        assert semi_bw < dim
        val_range = val_max - val_min

        dia_mat_nnz  = np.random.random( ( semi_bw + 1)*dim )
        dia_mat_nnz = val_min + val_range * dia_mat_nnz
        #print(dia_mat_nnz)
        dia_mat_nnz = dia_mat_nnz.reshape( (semi_bw + 1, dim) )
        #print(dia_mat_nnz)

        left_dia_array = np.arange(-semi_bw, 0)
        main_dia_array = np.array([0])
        dia_offsets = np.append(main_dia_array, left_dia_array)
        print(dia_offsets)

        mat_dia =  dia_matrix((dia_mat_nnz, dia_offsets), shape=(dim,dim))
        mat_dia_sym = mat_dia + mat_dia.transpose() - dia_matrix( (mat_dia.diagonal(), ([0])) , shape=(dim,dim) )
        #print(mat_dia_sym.todense() )

        return mat_dia_sym + factor_diag * eye(dim)

def main():
    """ """
    obj=MatGenerator();
    #mat_dia = obj.dia_mat_band_random_square(5, 1, 0, -10., 10.)
    #mat_dia = obj.dia_mat_band_random_square_PDpossible(5, 1, 0, 1., 10., 10.)
    #mat = obj.dia_mat_band_random_square_sym( 10 , 8, 1., 10.)
    mat = obj.dia_mat_band_random_square_sym_PDpossible( 6 , 1, 1., 10., 10.)
    #print(mat)
    print(type(mat))
    print(mat.todense())
    print ( (mat - mat.transpose()).todense() )
    #print(mat_dia.tocsr())
    #print(mat_dia.diagonal())
    #print("")
    #print(mat_dia.transpose().todense())
    #print("")
    #print( (mat_dia + mat_dia.transpose() - dia_matrix( (mat_dia.diagonal(), ([0])) , shape=(5,5))).todense())


if __name__== "__main__":
    """ call main function to test"""
    main()
