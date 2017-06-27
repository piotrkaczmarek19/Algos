# -*-coding:utf-8 -#

matrix1 = [[1 for i in xrange(5)] for i in xrange(5)]
matrix2 = [[1 for i in xrange(5)] for i in xrange(5)]
def matrixMultiplication(matrix1, matrix2, *args):
	line_length = len(matrix1[0])

	end_matrix = [[0 for i in xrange(line_length)] for i in xrange(len(matrix2))]
	
	k = 0
	l = 0
	while k<line_length and l<line_length:
		j = 0
		c = [0]*line_length	
		while j<line_length:
			i = 0
			while i<line_length: 
				partial_product = matrix1[k][i]*matrix2[i][l]
				c[j] = c[j] + partial_product
				i = i + 1
			end_matrix[k][j] = end_matrix[k][j] + c[j]
			j = j + 1	
		k = k+1
		l = l+1			
	return end_matrix
print(matrixMultiplication(matrix1, matrix2))

