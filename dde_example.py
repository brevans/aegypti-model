#!/usr/bin/env python
from PyDDE import pydde
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, precision=6, linewidth=280)


'''
DDE example from Solv95 distribution.

This model is a model for Nicholson's (1954) blowflies, as given by Gurney and Nisbet (1981)
'''

dde_eg = pydde.dde()

def ddegrad(s, c, t):
    alag = 0.0
    if (t > c[0]):
        alag = pydde.pastvalue(0, t - c[0], 0)
    return np.array([c[2] * alag * np.exp(-alag / c[3] ) - c[1] * s[0]])

def ddehist(g, s, c, t):
    return(s, g)

dde_cons = np.array([12.0, 0.25, 10.0, 1000.0, 100.0])
dde_ist = np.array([dde_cons[4]])
dde_stsc = np.array([0])

dde_eg.initproblem(no_vars = 1, no_cons = 5, nlag = 1, nsw=0, t0 = 0.0,
                   t1 = 1000.0, initstate = dde_ist, c = dde_cons,
                   otimes = np.arange(0.0, 1000.0, 1.0), grad = ddegrad, 
                   storehistory = ddehist)

dde_eg.initsolver(tol = 0.000005, hbsize = 100000, dt = 1.0, statescale = dde_stsc)

dde_eg.solve()
plt.plot(dde_eg.data[:,1])
plt.show()


'''
SDDE (DDE with switches) example from Solv95 distribution.
'''

sdde_eg = pydde.dde()

def sddegrad(s, c, t):
    g = np.array([0.0,0.0])
    g[0] = -c[0]*s[0]*s[1]
    g[1] = -c[5]*s[1]
    if (t>c[4]):
        g[1] = c[3]*pydde.pastvalue(0,t-c[4],0)*pydde.pastvalue(1,t-c[4],0)-c[5]*s[1]
    return g

def sddesw(s, c, t):
    sw = [0.0,0.0]
    sw[0]=np.sin(2*np.pi*t/c[1])         # add resource
    sw[1]=np.sin(2*np.pi*(t-c[4])/c[1])  # step nicely around a discontinuity
    return np.array(sw)
    
def sddemaps(s, c, t, swno):
    if (swno==0):
        return (np.array([s[0]+c[2],s[1]]),c)
    else:
        return (s,c)

sddecons = np.array([0.1, 10.0, 50.0, 0.05, 5.0, 0.02])
sddeist = np.array([0.0, 1.0])
sddestsc = 0*sddeist

sdde_eg.dde(y=sddeist, times=np.arange(0.0, 300.0, 1.0), 
           func=sddegrad, parms=sddecons, 
           switchfunc=sddesw, mapfunc=sddemaps,
           tol=0.000005, dt=1.0, hbsize=1000, nlag=1, nsw=2, ssc=sddestsc)

plt.plot(sdde_eg.data[:,2])
plt.show()