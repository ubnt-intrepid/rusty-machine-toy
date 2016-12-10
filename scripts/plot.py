#!/usr/bin/env python

import sys
import msgpack
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('Agg')

data = msgpack.unpackb(sys.stdin.buffer.read())
print(data)
