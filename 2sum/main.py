import time

# Building the hash table from the file
def init(file):
	with open("nums.txt") as f:
		lines = f.readlines()
		f.close()
		lines = [int(line) for line in lines]

		for line in lines:
		    array[line] = 1

# iterating over the table using bloom
def two_sum(hash_table, width):
	out = []
	counter = 0
	for integer in range(-width, width):
		hash_copy = hash_table.copy()
		if integer in hash_table:
			for key in hash_table:
				if (integer - key in hash_copy) and (key in hash_copy):
					if integer != key*2:
						counter = counter + 1
						print key
						del hash_copy[key]
						del hash_copy[integer - key]
					out.append([key, integer - key])
	return counter, out

def main(array):
	return two_sum(array, 10000)


if __name__ == '__main__':
    start = time.time()
    array = {}
    init('nums.txt')
    print main(array)
    print time.time() - start

