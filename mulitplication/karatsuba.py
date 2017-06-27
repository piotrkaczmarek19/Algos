#- -*-coding:utf-8 -#
import math

def karatsuba(x,y):
	x = str(x)
	y = str(y)
	if len(x) == 1 or len(y) == 1:
		return x*y
	
	expX = math.ceil(int(len(x)/2))
	expY = math.ceil(int(len(y)/2))

	exp = expX if expX<expY else expY

	middleX = expX - exp
	middleY = expY - exp

	xLeft = int(x)//(10**math.floor(middleX))
	xRight = int(x)%(10**math.floor(middleX))
	
	yLeft = int(y)//(10**math.floor(middleY))
	yRight = int(y)%(10**math.floor(middleY))


	a = karatsuba(xLeft,yRight) 
	b = karatsuba(yRight,xLeft)
	c = karatsuba(xLeft+xRight,yLeft+yRight)-a-b

	result = a + (c-b-a) * 10**(exp) + b * 10**(exp*2)
	return result

print(karatsuba(1505,1505))
print(1505*15)