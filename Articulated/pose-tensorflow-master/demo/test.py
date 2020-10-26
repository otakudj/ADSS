import numpy as np
import pylab as plt
import matplotlib
#matplotlib.use('Agg')

X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
C,S = np.cos(X), np.sin(X)

plt.plot(X,C)
plt.plot(X,S)

plt.show()