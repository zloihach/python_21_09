import scipy.io as sio
import numpy as np
import lib.svm as svm

mat = sio.loadmat("dataset1.mat")

y = np.float64(mat["y"])
x = np.float64(mat["X"])
Ф
svm.visualize_boundary_linear(x, y, "", "exc 1")
