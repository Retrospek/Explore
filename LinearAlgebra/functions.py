import numpy as np
import sympy

def gram_shmidt(matrix):
    """
    Arguments:
    - Matrix: Let's take in a numpy random array 

    Goal:
    - Convert and find a orthonormal basis

    Process:
    - We will take in some X = {x1,x2, ..., xn} for some vector space V
        - We will then find the basis for this vector space V
    - Again let's assume that each vector has some form of a linear independent "direction"

    Algorithm:
    - We will take the previous vector that incorporates all the unique directions (This is referencing the geometric intuition)
        and we will then just iterate how many LI independent vector times
    """

    # Step1: Find the rank or dimension of the provided matrix by finding the dimension of the column space
    row_size = matrix.shape[0]
    col_size = matrix.shape[1]

    sympy_matrix = sympy.Matrix(matrix)

    ref_matrix = sympy_matrix.rref() # Reduced Row Echelon Form and all non-zero value are pivots

    rank = 0

    for row in ref_matrix:
        for value in row:
            if value != 0:
                rank += 1