import sys
import scipy.io
from igraph import *

#Execute Maximum independent set from a Matlab program
#Input: mat file contains a matrix

#Intel Labs 2017
#Author: Nesreen K. Ahmed

def mis_compute(x):
	mat_content = scipy.io.loadmat(x)
	A_mat = mat_content['A']
	A = A_mat.tolist()
	A_size = len(A)
	edgs = []
	num_nods = A_size
	gsim = Graph(n=num_nods,directed=False)
	for ai in range(A_size):
		for aj in range(ai+1,A_size):
			if A[ai][aj]>0:
				edgs.append((ai,aj))
	gsim.add_edges(edgs)
	alpha = gsim.alpha()
	return alpha


if __name__ == '__main__':
    x = sys.argv[1]
    sys.stdout.write(str(mis_compute(x)))
