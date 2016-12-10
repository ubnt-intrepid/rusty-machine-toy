#!/usr/bin/env python

import sys
import msgpack
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

data = msgpack.unpackb(sys.stdin.buffer.read())

train_inputs = np.matrix(data[0])
test_inputs = np.matrix(data[1])
probs = np.array(data[2])
print(probs)

mixutre_weights = np.array(data[5])
means = list(np.array(m) for m in data[3])
covariances = list(np.matrix(m) for m in data[4])

plt.scatter(train_inputs[:,0], train_inputs[:,1], color="red", label="train")
plt.scatter(test_inputs[:,0], test_inputs[:,1], color="blue", label="test")
plt.legend(loc="best")
plt.savefig("result.png")
