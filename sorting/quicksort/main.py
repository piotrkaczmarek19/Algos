#-* coding:utf-8 -*#
counter = 0

def quicksort(alist, start, end):
	if start < end:
		pivot = partition(alist, start, end)
		quicksort(alist, start, pivot-1)
		quicksort(alist, pivot+1, end)
	return alist


def partition(A, l, r):
	global counter	
	counter = counter + r - l -1
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
	return i

def readFile(filename):
    with open(filename) as f:
        return [int(x) for x in f]

integers = "Integers.txt"
integers_array = readFile(integers)

Question1 = 63213678
Question2 = 0
print(integers_array)
print(quicksort(integers_array, 0, len(integers_array)-1))
print(counter)


