import scipy.io as sio;
import numpy as np;
import svm as svm;

mat = sio.loadmat("./assets/dataset1.mat")

y = np.float64(mat["y"])
x = np.float64(mat["X"])

C = 1;

model = svm.svm_train(x, y, C, svm.linear_kernel, 0.001,20);

svm.visualize_boundary_linear(x,y,model, "lab2_2_C=1");