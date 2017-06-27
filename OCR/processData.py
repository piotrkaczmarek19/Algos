import os
import numpy as np

# Scan directory for .dat files and merge them into one
def mergeFiles(adir):
	data = []
	for afile in os.listdir(adir):
		if afile.endswith(".dat"):
			data_chunk = np.loadtxt(afile, dtype=object)
			data.append(data_chunk)
				
	# transforming array into numpy array and merging instances together
	np_data = np.array(data)
	d = np_data[0]
	for array in np_data:
		d = np.vstack((d,array))

	# converting to int and replacing car brands with classificators
	for line in d:
		for i in range(len(line)):
			if line[i] in ['bus', 'van']:
				line[i] = 0
			elif line[i] in ['opel', 'saab']:
				line[i] = 1
			line[i] = int(line[i])
	return d






