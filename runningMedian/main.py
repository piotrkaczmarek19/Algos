from __future__ import division
import heapq
import time


def readFile(afile, stream):
	with open(afile) as f:
		lines = f.readlines()
		f.close
		lines = [int(line) for line in lines]

		for line in lines:
			stream.append(line)

# Keep track of median in an array receiving a stream of numbers
def running_median(stream):
	heap_lo, heap_hi = [], []
	y = 0
	total = len(stream)
	medians = []

	# Initialize low heap
	if len(heap_lo) == 0:
		heapq.heappush(heap_lo, -y)
		m = -heap_lo[0]

	while stream:
		y = stream.pop()
		m = -heap_lo[0]

		if len(heap_lo) == 0:
			heapq.heappush(heap_lo, -y)
			m = -heap_lo[0]
		# If new int is bigger than current median, push in hepa_hi
		if y > m :
			heapq.heappush(heap_hi, y)
			# Balancing heap_lo and heap_hi
			if len(heap_hi) > len(heap_lo):
				x = heapq.heappop(heap_hi)
				heapq.heappush(heap_lo, -x)
		# if new int is smaller than current median, push in heap_lo
		else:
			heapq.heappush(heap_lo, -y)
			if len(heap_lo) > len(heap_hi):
				x = heapq.heappop(heap_lo)
				heapq.heappush(heap_lo, -x)
		# appending maximum of heap_lo to medians array
		medians.append(-heap_lo[0])

	print medians

def main(afile):
	stream = []
	readFile(afile, stream)
	running_median(stream)


if __name__ == '__main__':
    start = time.time()
    main('nums.txt')
    print("Running time: %ss" %(time.time() - start) )