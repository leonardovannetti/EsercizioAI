# Modulo per importare il dataset

import numpy as np

X_train = np.loadtxt("avila-tr.txt", delimiter=',', usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
y_train = np.loadtxt("avila-tr.txt", dtype='str', delimiter=',', usecols=10)

X_test = np.loadtxt("avila-ts.txt", delimiter=',', usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
y_test = np.loadtxt("avila-ts.txt", dtype='str', delimiter=',', usecols=10)
