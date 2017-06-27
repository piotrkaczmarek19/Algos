# -*-coding:utf-8 -#
import math

pointA = [3,13]
pointB = [0,0]
pointC = [1,99]
pointD = [100,13]
pointE = [1000,999]
pointF = [1,1000]
pointG = [125,13]
pointH = [0,60]

array = [pointA, pointB, pointC, pointD, pointE, pointF]

def calcDistance(pointA, pointB):
	return math.sqrt((pointA[0]-pointB[0])**2 + (pointA[1]-pointB[1])**2)

def sortPointsBy(axis, array):
# sorting array of points by x coordinate
	if axis == "x":
		if len(array) == 1:
			return array 

		middle = len(array)//2
		first_array = array[:middle]
		second_array = array[middle:]

		sortPointsBy("x",first_array)
		sortPointsBy("x",second_array)

		i=0
		j=0
		k=0
		while i<len(first_array) and j<len(second_array):
			if first_array[i][0]<second_array[j][0]:
				array[k] = first_array[i]
				i= i+1
			else:
				array[k] = second_array[j]
				j= j+1
			k = k+1

		while i<len(first_array):
			array[k] = first_array[i]
			i = i+1
			k = k+1
		while j<len(second_array):
			array[k] = second_array[j]
			j = j+1
			k = k+1
	elif axis == "y":
		if len(array) == 1:
			return array 

		middle = len(array)//2
		first_array = array[:middle]
		second_array = array[middle:]

		sortPointsBy("y",first_array)
		sortPointsBy("y",second_array)

		i=0
		j=0
		k=0
		while i<len(first_array) and j<len(second_array):
			if first_array[i][1]<second_array[j][1]:
				array[k] = first_array[i]
				i= i+1
			else:
				array[k] = second_array[j]
				j= j+1
			k = k+1

		while i<len(first_array):
			array[k] = first_array[i]
			i = i+1
			k = k+1
		while j<len(second_array):
			array[k] = second_array[j]
			j = j+1
			k = k+1
	return array	



def bestPair(array, *arg):
	bestPair = [array[0], array[1]]
	for i in range(len(array)):
		for j in range(len(array)):
			if calcDistance(array[i],array[j])<calcDistance(bestPair[0], bestPair[1])  and j != i:
				bestPair[0] = array[i]
				bestPair[1] = array[j]
	return bestPair

def findClosestPair(array, *arg):
	if len(array)<4:
		return bestPair(array)

	# O(n(ln n)) sorting by X coordinate and Y coordinate
	xSortedPoints = sortPointsBy("x", array)
	ySortedPoints = sortPointsBy("y", array)

	# dividing points into two equal halfs. Q being right side and P being left side
	middle = len(array)//2
	Px = xSortedPoints[:middle]
	Qx = xSortedPoints[middle:]
	Py = ySortedPoints[:middle]
	Qy = ySortedPoints[middle:]

	#finding the optimal distances on both the left and right side of the dividing line
	minimumP = findClosestPair(Px)
	deltaP = calcDistance(minimumP[0], minimumP[1])

	minimumQ = findClosestPair(Qx)
	deltaQ = calcDistance(minimumQ[0], minimumQ[1])

	delta = deltaP if deltaP<deltaQ else deltaQ
	minimum = minimumP if deltaP<deltaQ else minimumQ

	bestPairArray = minimum

	# worst case scenario: looking for overlaping points on a 2-delta wide area around the central point
	neighborhood = []
	for j in range(len(ySortedPoints)):
		neighborhood.append(ySortedPoints[j]) if abs(ySortedPoints[j][0]-middle) < delta else 1
	print(minimum)
	i = 0
	k = 0
	
	delta3 = delta
	if len(neighborhood)>1:
		for i in range(len(neighborhood)):
			for j in range(min(7, len(neighborhood))):
				if calcDistance(neighborhood[i],neighborhood[j])<delta  and j != i:
					bestPairArray = [neighborhood[i],neighborhood[i+j]]
		delta3 = calcDistance(bestPairArray[0], bestPairArray[1]) 		

	return bestPairArray

print(findClosestPair(array))