#!/usr/bin/env python
import numpy as np
from PyDDE import pydde
import matplotlib.pyplot as plt

ae_dde = pydde.dde()
np.set_printoptions(suppress=True, precision=6, linewidth=280)

RUNS = 100
NUMPARAMS = 22
NUMSTATEVARS = 17

N_0 = 10000
YEARS = 1
RELEASE_RATIO = 10
CONTROL_START = 365.0*2
CONTROL_END = 365.0 * YEARS + CONTROL_START

'''PARAMETERS'''
'''time steps, 1 day'''
step = np.empty(RUNS)
step.fill(1)

'''adult mosqiuto death rate'''
Sigma = np.random.uniform(low = 1/15.0, high = 1/3.0, size=RUNS)

'''Mosquito Generation Time egg->emerging adult'''
T = np.random.uniform(low = 16.9, high = 20.1, size=RUNS)

'''Daily egg production rate per adult mosquito (a female lays 2E)'''
E = np.random.uniform(low = 7.0, high = 9.0, size=RUNS)

'''number of offspring produced by each adult per day that will 
survive to adulthood in the absence of density dependent mortality 
(i.e. E adjusted for density-independent egg-to-adult survival) 
estimated value, calculated using field data, depends 
on value of s (P/s is net reproductive rate)'''
P = np.random.uniform(low = 7.0, high = 9.0, size=RUNS)

'''Average number of vectors (adult female mosquitoes) per host, 
initial pop is N_0'''
k = np.random.uniform(low = 0.3, high = 20, size=RUNS)

'''Strength of larval density dependence'''
Beta = np.random.uniform(low = 0.302, high = 1.5, size=RUNS)

'''breeding site multiplier'''
Alpha = np.log(P / Sigma) / (2 * k * N_0 * E ) ** Beta

'''Maintained ratio of RIDL males to pre-release equilibrium number of \
adult males (constant release policy)'''
C = np.empty(RUNS)
C.fill(RELEASE_RATIO)

'''Human per capita birth rate (per day) Equal to human death rate'''
v = np.random.uniform(low = 1/(60*365.0), high = 1/(68*365.0), size=RUNS)

'''Human per capita birth rate (per day)'''
Mu = v

'''biting rate (number of bites per mosquito per day)'''
b = np.random.uniform(low = 0.33, high = 1.0, size=RUNS)

'''Proportion of bites that successfully infect a susceptible human'''
a = np.random.uniform(low = 0.25, high = 0.75, size=RUNS)

'''Proportion of bites that successfully infect a susceptible mosquito'''
c = np.random.uniform(low = 0.25, high = 0.75, size=RUNS)

'''Virus latent period in humans (days) Intrinsic incubation period'''
Tau = np.random.uniform(low = 3.0, high = 12.0, size=RUNS)

'''Virus latent period in vectors (days) Extrinsic incubation period'''
Omega = np.random.uniform(low = 7.0, high = 14.0, size=RUNS)

'''Human recovery rate (per day) 1/infectious period'''
Gamma = np.random.uniform(low = 1/10.0, high = 1/2.0, size=RUNS)

'''Rate at which humans lose cross-immunity (per day)'''
Psi = np.random.uniform(low = 1/(365*5/12.0), high = 1/(365*2/12.0), size=RUNS)

'''Increased host susceptibility due to ADE'''
Chi = np.random.uniform(low = 1.0, high = 3.0, size=RUNS)

'''Alternative to Chi, Increased transmissibility due to ADE'''
Zeta = np.empty(RUNS)
Zeta.fill(1)

'''Proportion of hosts that recover from secondary infection 
(1-Rho die from DHF/DSS)'''
Rho = np.random.uniform(low = 0.9935, high = 1, size=RUNS)

'''Stable equilibrium of pre-release mosquito population'''
F_star = ((1/Alpha * np.log(P / Sigma)) ** 1/Beta) / 2 * E


'''Name the variable indexes for sanity in coding the model... Will make
c[1] and c[_T], for example, synonymous'''
(_step, _Sigma, _T, _E, _P, _k, _Beta, _Alpha, _C, _v, _Mu, _b, _a, _c, 
 _Tau, _Omega, _Gamma, _Psi, _Chi, _Zeta, _Rho, _F_star) = range(NUMPARAMS)

'''Name the variable indexes for sanity in coding the model... Will make
s[1] and s[_T], for example, synonymous'''
(_F, _X, _Y_i, _Y_j, _Y_ij, _Y_ji, _N, _S, _I_i, _I_j, _C_i, 
 _C_j, _R_i, _R_j, _I_ij, _I_ji, _R) = range(NUMSTATEVARS)


params = np.vstack((step, Sigma, T, E, P, k, Beta, Alpha, C, v, Mu, b, a, c, 
                   Tau, Omega, Gamma, Psi, Chi, Zeta, Rho, F_star)).T

def ddehist(g, s, c, t):
    return(s, g)

def RIDL_dde(s, c, t):
    '''STATE VARIABLES
    MOSQUITOS
    F: Total number of female vectors
    X: Suceptible female vectors
    ( Y_i, Y_j, Y_ij, Y_ji )infectious with serotype j, i or first one then the other

    HUMANS
    N:   Total number of hosts
    S:   Suceptible to all serotypes
    ( I_i, I_j ):  Primary infection with serotype i or j
    ( C_i, C_j ):  Recovered from primary infection with serotype i or j, temporarily cross-imune
    ( R_i, R_j ):  Recovered from serotype i, suceptible to all other types
    ( I_ij, I_ji ) Secondary infection with serotype j, following primary infection with serotype i
    R:   Recovered from secondary infection, immune to all serotypes
    '''
    print(s, c, t)
    F_prev = pydde.pastvalue(_F, t - c[_step] , 0)

    '''modified equation S1'''
    if t <= CONTROL_START:
        C_tilde = 0
    else:
        C_tilde = c[_C]

    F_now = (c[_P] * F_prev * (F_prev / (F_prev+C_tilde*c[_F_star])) * 
             np.exp(-c[_Alpha] * (2*c[_E]*F_prev) ** c[_Beta] ) - c[_Sigma] * s[_F] )
    print(F_now)
    return np.array([F_now])
'''
ae_dde.initproblem(no_vars = 1, no_cons = NUMPARAMS, nlag = 1, nsw=0, t0 = 0,
                   t1 = CONTROL_START+CONTROL_END, initstate = np.array([10]), c = params[1,:],
                   otimes = np.arange(0, CONTROL_START + CONTROL_END, 1.0), grad = RIDL_dde, 
                   storehistory = ddehist)


ae_dde.initsolver(tol = 0.000005, hbsize = (CONTROL_START + CONTROL_END)*22, dt = 1.0)
'''

ae_dde.dde(y=np.array([1.0]), times=np.arange(0, CONTROL_END, 1.0),
           func=RIDL_dde, parms=params[1,:], tol=0.000005, dt=1.0, hbsize=1000, nlag=1)



#ae_dde.solve()
plt.plot(ae_dde.data[:,1])
plt.show()
