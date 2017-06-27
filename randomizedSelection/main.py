#-* coding:utf-8 -*#
import random

def partition(A, p_index):
	# Swapping pivot with first element
	pivot = A[p_index]
	A[p_index], A[-1] = (A[-1], A[p_index])


	i = -1
	for j in xrange(0, len(A)):
		if A[j] < pivot:
			A[j], A[i+1] = (A[i+1], A[j])
			i = i + 1

	A[i+1], A[-1] = (A[-1], A[i+1])
	return i + 1

def quickSelect(alist, ith):
	if len(alist)<2:
		return alist[0]

	# Choosing pivot at random
	p_index = random.randint(0,len(alist)-1)

	# partitioning the main array
	p_index = partition(alist, p_index)

	# Choosing appropriate recursion depending on where the pivot is compared to wanted order statistic 
	if p_index == ith - 1:
		return alist[p_index]
	elif p_index>ith:
		return quickSelect(alist[:p_index],ith)
	else:
		return quickSelect(alist[p_index+1:], ith-p_index-1)
	


alist = [2,5,0,3,8,99999999]

print(quickSelect(alist, 6))
print(alist)