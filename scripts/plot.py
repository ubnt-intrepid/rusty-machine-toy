#!/usr/bin/env python

import sys
import msgpack

# import matplotlib as mpl
# import matplotlib.pyplot as plt
# mpl.use('Agg')

data = msgpack.unpackb(sys.stdin.buffer.read())

inputs = data[0]
targets = data[1]
params = data[2]

x1_test   = data[3]
x2_test   = data[4]
predicted = data[5]

print(x1_test, x2_test, predicted)
