#-* coding:utf-8 -*#

def quicksort(alist, start, end):
	if start < end:
		temp = alist[end]
		alist[end] = alist[start]
		alist[start] = temp
		pivot = partition(alist, start, end)
		quicksort(alist, start, pivot-1)
		quicksort(alist, pivot+1, end)
	return alist

def partition(A, l, r):
	p = A[l]
	i = l + 1
	for j in range(l+1,r+1):
		if A[j] < p:
			temp = A[j]
			A[j] = A[i]
			A[i] = temp
			i = i + 1
	temp = A[l]
	A[l] = A[i-1]
	A[i - 1] = temp
	return i -1



alist = [54,26,93,17,77,31,44,55,20,11]
print(quicksort(alist, 0, len(alist)-1))

