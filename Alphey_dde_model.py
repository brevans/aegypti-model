#!/usr/bin/env python
'''
An attempt at implementing the model presented in Alphey, et al.
A model framework to estimate impact and cost of genetics-based
sterile insect methods for dengue vector control.
PloS one 6, e25384 (2011).
'''
#from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
import dde

NUMPARAMS = 21
NUMSTATEVARS = 15
#Name the variable indexes for sanity in coding the model... Will make
#p[1] and p[_T], for example, synonymous
(_Sigma, _T, _E, _P, _k, _Beta, _Alpha, _C, _v, _Mu, _b, _a, _c, 
 _Tau, _Omega, _Gamma, _Psi, _Chi, _Zeta, _Rho, _F_star) = range(NUMPARAMS)

(_F, _X, _Yi, _Yj, _S, _Ii, _Ij, _Ci, _Cj, _Ri, _Rj, 
 _Iij, _Iji, _R, _N) = range(NUMSTATEVARS)

#the names of the state variables and parameters
p_names = ["Sigma", "T", "E", "P", "k", "Beta", "Alpha", "C",
           "v", "Mu", "b", "a", "c", "Tau", "Omega", "Gamma", "Psi",
           "Chi", "Zeta", "Rho", "F_star"]

s_names = ["F", "X", "Yi", "Yj", "S", "Ii", "Ij", "Ci", "Cj", "Ri",
          "Rj", "Iij", "Iji", "R", "N"]


def params():
    #PARAMETERS
    #adult mosqiuto death rate
    Sigma = np.random.uniform(low = 1/15.0, high = 1/3.0, size=RUNS)

    #Mosquito Generation Time egg->emerging adult
    T = np.random.uniform(low = 16.9, high = 20.1, size=RUNS)

    #Daily egg production rate per adult mosquito (a female lays 2E)
    E = np.random.uniform(low = 7.0, high = 9.0, size=RUNS)

    #number of offspring produced by each adult per day that will 
    #survive to adulthood in the absence of density dependent mortality 
    #(i.e. E adjusted for density-independent egg-to-adult survival) 
    #estimated value, calculated using field data, depends 
    #on value of s (P/s is net reproductive rate)
    P = np.random.uniform(low = 7.0, high = 9.0, size=RUNS)

    #Average number of vectors (adult female mosquitoes) per host, 
    #initial pop is N0
    k = np.random.uniform(low = 0.3, high = 20, size=RUNS)

    #Strength of larval density dependence
    Beta = np.random.uniform(low = 0.302, high = 1.5, size=RUNS)

    #breeding site multiplier
    Alpha = np.log(P / Sigma) / (2 * k * N0 * E ) ** Beta

    #Maintained ratio of RIDL males to pre-release equilibrium number of 
    #adult males (constant release policy)
    C = np.empty(RUNS)
    C.fill(RELEASE_RATIO)

    #Human per capita birth rate (per day) Equal to human death rate
    v = np.random.uniform(low = 1/(60*365.0), high = 1/(68*365.0), size=RUNS)

    #Human per capita birth rate (per day)
    Mu = v

    #biting rate (number of bites per mosquito per day)
    b = np.random.uniform(low = 0.33, high = 1.0, size=RUNS)

    #Proportion of bites that successfully infect a susceptible human
    a = np.random.uniform(low = 0.25, high = 0.75, size=RUNS)

    #Proportion of bites that successfully infect a susceptible mosquito
    c = np.random.uniform(low = 0.25, high = 0.75, size=RUNS)

    #Virus latent period in humans (days) Intrinsic incubation period
    Tau = np.random.uniform(low = 3.0, high = 12.0, size=RUNS)

    #Virus latent period in vectors (days) Extrinsic incubation period
    Omega = np.random.uniform(low = 7.0, high = 14.0, size=RUNS)

    #Human recovery rate (per day) 1/infectious period
    Gamma = np.random.uniform(low = 1/10.0, high = 1/2.0, size=RUNS)

    #Rate at which humans lose cross-immunity (per day)
    Psi = np.random.uniform(low = 1/(365*5/12.0), high = 1/(365*2/12.0), size=RUNS)

    #Increased host susceptibility due to ADE
    Chi = np.random.uniform(low = 1.0, high = 3.0, size=RUNS)

    #Alternative to Chi, Increased transmissibility due to ADE
    Zeta = np.empty(RUNS)
    Zeta.fill(1)

    #Proportion of hosts that recover from secondary infection 
    #(1-Rho die from DHF/DSS)
    Rho = np.random.uniform(low = 0.9935, high = 1, size=RUNS)

    #Stable equilibrium of pre-release mosquito population
    F_star = ((1/Alpha * np.log(P / Sigma)) ** 1/Beta) / 2 * E

    return np.vstack((Sigma, T, E, P, k, Beta, Alpha, C, v, Mu, b, a, c, 
                       Tau, Omega, Gamma, Psi, Chi, Zeta, Rho, F_star)).T

def original_params():
    '''Returns the parameters used in the manuscript as an array
    '''
    Sigma = 1/14.0
    T = 18.5
    E = 8.0
    P = 0.7
    k = 2.0
    Beta = 1.0
    Alpha = 1.5e-8
    C = RELEASE_RATIO
    v = 1/(60*365.0)
    Mu = 1/(60*365.0)
    b = 0.5
    a = 0.38
    c = 0.38
    Tau = 5
    Omega = 10
    Gamma = 1/6.0
    Psi = 1/(365*4/12.0)
    Chi = 1.5
    Zeta = 1
    Rho = 0.9999
    F_star = ((1/Alpha * np.log(P / Sigma)) ** 1/Beta) / 2 * E

    return np.array([Sigma, T, E, P, k, Beta, Alpha, C, v, Mu, b, a, c, Tau,
                     Omega, Gamma, Psi, Chi, Zeta, Rho, F_star])

def RIDL_dde(Y, t, p):
    '''
    This function contains the system of delay diff equations.
    INPUTS
    Y: a history function, where Y(t) returns a 1d array of the values of the 
       state variables at time t.
    t: the current time in days
    p: a 1d array that contains the parameters for the model

    STATE VARIABLES
    MOSQUITOS
    F: Total number of female vectors
    X: Suceptible female vectors
    Yi, Yj: infectious with serotype j or i 

    HUMANS
    N:   Total number of hosts
    S:   Suceptible to all serotypes
    Ii, Ij:   Primary infection with serotype i or j
    Ci, Cj:   Recovered from primary infection with serotype i or j,
                    temporarily cross-imune
    Ri, Rj:   Recovered from serotype i, suceptible to all other types
    Iij, Iji: Secondary infection with other serotype
    R:              Recovered from secondary infection, immune to all serotypes
    '''
    #state variables at time t
    s = [x if x > .001 else 0 for x in  Y(t)]

    #state variables at time t - T delay (mosquito generation time)
    ds = [x if x > .001 else 0 for x in Y(t-p[_T])]
    
    #state variables at time t - omega delay (extrinsic incubation period)
    os = [x if x > .001 else 0 for x in Y(t-p[_Omega])]

    #state variables at time t - tau delay (inrinsic incubation period)
    ts = [x if x > .001 else 0 for x in Y(t-p[_Tau])]

    #equation 1
    if t <= CONTROL_START:
        C_tilde = 0
    elif t <= CONTROL_END and t >= CONTROL_START:
        C_tilde = p[_C]
    else:
        C_tilde = 0
    
    #equation 1
    if ds[_F] == 0:
        F = 0
    else:
        F = ((p[_P] * ds[_F]) * (ds[_F] / (ds[_F] + C_tilde * p[_F_star])) * 
                np.exp(-(p[_Alpha] * (2*p[_E]*ds[_F]) ** p[_Beta] )) - 
                p[_Sigma] * s[_F] )
    
    #equation 2
    if ds[_F] == 0:
        X = 0
    else:
        X = ((p[_P] * ds[_F]) * (ds[_F] / (ds[_F] + C_tilde * p[_F_star])) * 
                np.exp(-(p[_Alpha] * (2*p[_E]*ds[_F]) ** p[_Beta] )) - 
                p[_c] * p[_b] * s[_X] / s[_N] *
                (s[_Ii] + s[_Ij] + p[_Zeta] * s[_Iij] + p[_Zeta] * s[_Iji])
                - p[_Sigma] * s[_X] )
        
    #equation 3
    Yj = np.exp(-p[_Sigma] * p[_Omega]) * ((p[_c] * os[_X] / os[_N]) *
     (os[_Ij] + p[_Zeta] * os[_Iji]) ) - p[_Sigma] * s[_Yj]
    Yi = np.exp(-p[_Sigma] * p[_Omega]) * ((p[_c] * os[_X] / os[_N]) *
     (os[_Ii] + p[_Zeta] * os[_Iij]) ) - p[_Sigma] * s[_Yi]

    #equation 4
    S = ( p[_v] * s[_N] - (p[_a] * p[_b] * 
        (s[_S] * (s[_Yi] + s[_Yj]))/s[_N] ) - p[_Mu] * s[_S] )

    #equation 5
    Ii = (np.exp(-p[_Mu] * p[_Tau]) * (p[_a] * p[_b]) *
          (ts[_S] * ts[_Yi] / ts[_N]) - s[_Ii] * (p[_Gamma] + p[_Mu]))
    Ij = (np.exp(-p[_Mu] * p[_Tau]) * (p[_a] * p[_b]) *
          (ts[_S] * ts[_Yj] / ts[_N]) - s[_Ij] * (p[_Gamma] + p[_Mu]))

    #equation 6
    Ci = p[_Gamma] * s[_Ii] - s[_Ci] * (p[_Psi] + p[_Mu])
    Cj = p[_Gamma] * s[_Ij] - s[_Cj] * (p[_Psi] + p[_Mu])

    #equation 7
    Ri = ( s[_Ci] * p[_Psi] - (p[_Chi] * p[_a] * p[_b] *
          s[_Ri] * s[_Yj] / s[_N]) - s[_Ri] * p[_Mu] )
    Rj = ( s[_Cj] * p[_Psi] - (p[_Chi] * p[_a] * p[_b] *
          s[_Rj] * s[_Yi] / s[_N]) - s[_Rj] * p[_Mu] )

    #equation 8
    Iij = (np.exp(-p[_Mu] * p[_Tau]) * (p[_Chi] * p[_a] * p[_b]) *
           (ts[_Ri] * ts[_Yj] / ts[_N]) - (s[_Iij] * (p[_Gamma] + p[_Mu])))
    Iji = (np.exp(-p[_Mu] * p[_Tau]) * (p[_Chi] * p[_a] * p[_b]) *
           (ts[_Rj] * ts[_Yi] / ts[_N]) - (s[_Iji] * (p[_Gamma] + p[_Mu])))

    #equation 9
    R = (p[_Rho] * p[_Gamma] * (s[_Iij] + s[_Iji])) - p[_Mu] * s[_R]

    #equation 10
    N = ((s[_N] * p[_v]) - ((1-p[_Rho]) * p[_Gamma] * (s[_Iij] + s[_Iji]))
         - (s[_N] * p[_Mu]))

    state = np.array([F, X, Yi, Yj, S, Ii, Ij, Ci, Cj, Ri, Rj, Iij, Iji, R, N])
    return state

np.set_printoptions(suppress=True, precision=6, linewidth=280)
np.seterr(all='raise')

RUNS = 1

N0 = 1000000.0
CONTROL_YEARS = 1
PRE_CONTROL_YEARS = 50
POST_CONTROL_YEARS = 20
DT = 1.0
RELEASE_RATIO = 10
CONTROL_START = 365.0 * PRE_CONTROL_YEARS
CONTROL_END = 365.0 * CONTROL_YEARS + CONTROL_START
SIMULATION_END =  365.0 * POST_CONTROL_YEARS + CONTROL_END 

start_state = np.array([N0, N0, 0.0, 0.0, N0, 
                        0.01*N0, 0.01*N0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, N0])
print("\nStarting values")
for i, j in zip(s_names, start_state):
    print('{}: {}'.format(i, j))
print("")

g = lambda t : start_state

tt = np.linspace(0, SIMULATION_END, SIMULATION_END*DT, endpoint=True)
yys = []

params = np.vstack((original_params().T, params()))

good_runs = []
for r in range(RUNS):
    par = params[r, :]
    print("Parameters for run {}:".format(r))
    for i, j in zip(p_names, par):
        print('{}: {}'.format(i, j))
    print("")
    #only keep runs where the solver completes successfully
    #try:
    #    yys.append(dde.ddeint(RIDL_dde, g, tt, fargs=(par, )))
    #    good_runs.append(r)
    #except FloatingPointError:
    #    continue  
    yys.append(dde.ddeint(RIDL_dde, g, tt, fargs=(par, )))
    good_runs.append(r)

out = open('last_parameters.csv', 'w')
out.write(','.join(p_names)+'\n')
for r in good_runs:
    out.write(','.join(str(x) for x in params[r, :])+'\n')
out.close()

gr = len(good_runs)
if gr >= 1:
    vars_to_graph = [_F, _Yi, _S, _Ii, _Ri, _Iij, _R, _N]
    num_vars = len(vars_to_graph)
    f, axarr = plt.subplots(num_vars)
    graph_start = 0#int(CONTROL_START - 300)
    graph_end = int(SIMULATION_END)

    for i, v in enumerate(vars_to_graph):
        #lines = ["-","--","-.",":"]
        #linecycler = cycle(lines)
        yy = np.array([y[graph_start:graph_end, v] for y in yys])
        axarr[i].plot(tt[graph_start:graph_end], yy.T,
                      #linestyle = next(linecycler)
                      )
        axarr[i].set_xlim(graph_start, graph_end)
        axarr[i].axvline(CONTROL_START, color='black')
        axarr[i].axvline(CONTROL_END, color='black')
        if i+1 != num_vars:
            axarr[i].set_xticks([])
        else:
            axarr[i].set_xlabel('Days')

        axarr[i].set_ylabel(s_names[v], rotation='horizontal')
        axarr[i].yaxis.set_label_position("right")

    plt.figlegend(labels=('Run_{}'.format(x) for x in good_runs),
                  handles= axarr[-1].get_lines(), loc=1)
    plt.show()
