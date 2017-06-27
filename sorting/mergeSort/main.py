# -*-coding:utf-8 -#

import time

def mergeSort(numbers, *args):	
	pointer = []
	start_time = time.time()
	if len(numbers) == 1:
		return numbers 

	middle = len(numbers)//2
	first_array = numbers[:middle]
	second_array = numbers[middle:]

	mergeSort(first_array)
	mergeSort(second_array)

	i=0
	j=0
	k=0
	while i<len(first_array) and j<len(second_array):
		if first_array[i]<second_array[j]:
			numbers[k] = first_array[i]
			i= i+1
		else:
			numbers[k] = second_array[j]
			j= j+1
		k = k+1

	while i<len(first_array):
		numbers[k] = first_array[i]
		i = i+1
		k = k+1
	while j<len(second_array):
		numbers[k] = second_array[j]
		j = j+1
		k = k+1
	
	print("--- %s seconds ---" % (time.time() - start_time))
	return numbers

numbers = [5,1,4,6,9,45,13,26,45,5,5,54,54,2,61,65,269,9,49,1,6,69,6,848545,2,921,]

print(mergeSort(numbers))