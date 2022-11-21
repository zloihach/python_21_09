import scipy.io as sio
import numpy as np
import svm as svm

mat = sio.loadmat("./assets/dataset1.mat")

Y = np.float64(mat["y"])
X = np.float64(mat["X"])

svm.visualize_boundary_linear(X, Y, "", "lab1_1")
