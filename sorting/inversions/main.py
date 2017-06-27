# -*-coding:utf-8 -#
# variation on merge sort for counting variations in an array #
# Alternatives on: http://codereview.stackexchange.com/questions/12922/inversion-count-using-merge-sort#

def mergeSort(array, c):
	if len(array)<2: return array
	middle = len(array)/2

	return merge(mergeSort(array[:middle],c), mergeSort(array[middle:], c),c)

def merge(l,r,c):
	result = []
	while l and r:
		s = l if l[0]<r[0] else r
		result.append(s.pop(0))
		if(s==r): c[0] += len(l) #if next number is smaller than the already sorted numbers, increase the counter
	result.extend(l if l else r)
	return result

def readFile(filename):
    with open(filename) as f:
        return [int(x) for x in f]

integers = "IntegerArray.txt"
integers_array = readFile(integers)
c = [0]
print(mergeSort(integers_array, c))
print c[0]
	
	