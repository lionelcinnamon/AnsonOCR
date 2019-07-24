import numpy as np

# TODO: implement other decoding techniques

def ctcBestPath(mat, classes):
	"implements best path decoding as shown by Graves (Dissertation, p63)"

	# dim0=t, dim1=c
	maxT, maxC = mat.shape
	label = ''
	blankIdx = len(classes)
	lastMaxIdx = maxC # init with invalid label	

	for t in range(maxT):
		maxIdx = np.argmax(mat[t, :])

		if maxIdx != blankIdx and maxIdx != lastMaxIdx:
			label += classes[maxIdx]

		lastMaxIdx = maxIdx

	return label



def argmax_ctcBestPath(maxIdxs, classes):
    "implements best path decoding as shown by Graves (Dissertation, p63)"
    label = ''
    blankIdx = len(classes)
    lastMaxIdx = len(classes) # init with invalid label	
    
    for maxIdx in maxIdxs:
        if maxIdx == -1:
            break
            
        if maxIdx != blankIdx and maxIdx != lastMaxIdx:
            label += classes[maxIdx]

        lastMaxIdx = maxIdx

    return label