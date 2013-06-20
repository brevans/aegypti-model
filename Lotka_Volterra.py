#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import dde

def model(Y, t, d):
    x, y = Y(t)
    xd, yd = Y(t-d)
    return np.array([0.5*x*(1-yd), -0.5*y*(1-xd)])
 
g = lambda t : np.array([0, 0])

tt = np.linspace(2, 30, 20000)
(fig, ax) = plt.subplots(1)
 
for d in [0, 0.2]:
    yy = dde.ddeint(model, g, tt, fargs=(d,))
    # WE PLOT X AGAINST Y
    ax.plot(yy[:, 0], yy[:, 1], lw=2, label='delay = {0:.01f}'.format(d))
 
ax.legend()
plt.show()