import lib.svm as svm
import scipy.io as sio
import numpy as np

mat = sio.loadmat("dataset2.mat")

y = np.float64(mat["y"])
x = np.float64(mat["X"])

C = 1.0
sigma = 0.1

gaussian = svm.partial(svm.gaussian_kernel, sigma=sigma)
gaussian.__name__ = svm.gaussian_kernel.__name__
model = svm.svm_train(x,y, C, gaussian)
svm.visualize_boundary(x, y, model)