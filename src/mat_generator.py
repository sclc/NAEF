""" """
import numpy as np
#from scipy.sparse import coo_matrix
from scipy.sparse import dia_matrix

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




def main():
    """ """
    obj=MatGenerator();
    mat_dia = obj.dia_mat_band_random_square(5, 1, 0, -10., 10.)
    print(mat_dia)
    print(type(mat_dia))
    print(mat_dia.todense())
    print(mat_dia.tocsr())


if __name__== "__main__":
    """ call main function to test"""
    main()
