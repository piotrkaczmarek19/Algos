# -*-coding:utf-8 -#
# http://stackoverflow.com/questions/9622778/strassens-subcubic-matrix-multiplication-algorithm-with-recursion #


matrix1 = [[1 for i in xrange(4)] for i in xrange(4)]
matrix2 = [[1 for i in xrange(4)] for i in xrange(4)]


def matrixMultiplication(matrix1, matrix2, *args):
	size_matrices = len(matrix1)
	if size_matrices == 1:
		print(matrix1[0][0])
		d = [[0]]
		d[0] = matrix1[0][0]*matrix2[0][0]
		return d
	
	middle = size_matrices/2
	first_quadrantM1 = matrix1[:middle][:middle]
	second_quadrantM1 = matrix1[middle:][:middle]
	third_quadrantM1 = matrix1[:middle][middle:]
	fourth_quadrantM1 = matrix1[middle:][middle:]

	first_quadrantM2 = matrix2[:middle][:middle]
	second_quadrantM2 = matrix2[middle:][:middle]
	third_quadrantM2 = matrix2[:middle][middle:]
	fourth_quadrantM2 = matrix2[middle:][middle:]

	print(first_quadrantM1, third_quadrantM2) 

	
	
print(matrixMultiplication(matrix1, matrix2))