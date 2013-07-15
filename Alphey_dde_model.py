#!/usr/bin/env python
'''
Alphey_dde_model:
An attempt at implementing the model presented in Alphey, et al.
A model framework to estimate impact and cost of genetics-based
sterile insect methods for dengue vector control.
PloS one 6, e25384 (2011).

You need numpy, scipy, and matplotlib installed for this to work.
You can specify some of the starting conditions via the command line, or just
run something like what's described in the paper by default.

Usage:
    Alphey_dde_model.py [-r 10] [-c 1] [-o 249] [-t 1] [-b 250] [-a 10] 
                        [-i 50000]
Options:
    -r, --runs=num           Number of runs to simulate the release.  The first 
                              run will always be the parameters used in the 
                              manuscript. [default: 1]
    -c, --releaseratio=num   Release ratio, relative to the natural mosquito 
                              population of the RIDL mosquitoes [default: 1]
    -o, --outbreak=num       The time the outbreak starts [default: 4.5]
    -n, --initialpop=num     The initial size of the human population 
                              [default: 1000000]
    -t, --releasetime=num    The ammount of time to stage the release in years 
                              [default: 1]
    -b, --timebefore=num     The ammount of time to wait before the release 
                              starts in years [default: 5]
    -a, --timeafter=num      The ammount of time to continue the simulation 
                              after the release in years [default: 5]
    -i, --initialpop=num     The initial human population size [default: 100000]

    -h, --help               Show this message

'''
import sys
from os import path

#try to download docopt if you don't have it, in a 2 or 3 compatible way
try:
    from docopt import docopt
except ImportError:
    #yay for the weirdness that is urllib in 2&3
    try:
        from urllib.request import urlopen
    except ImportError:
        from urllib import urlopen
    r = urlopen('https://raw.github.com/docopt/docopt/master/docopt.py')
    docopt_raw = r.read()
    docopt_out = open(path.join(path.dirname(path.abspath(sys.argv[0])),
                                'docopt.py'), 'w')
    docopt_out.write(docopt_raw.decode())
    docopt_out.close()
    from docopt import docopt
import numpy as np
import matplotlib.pyplot as plt
import dde

#hard-coded to force myself to pay attention to parameters and state variables
NUMPARAMS = 25
NUMSTATEVARS = 15

#Name the variable indexes for sanity in coding the model... Will make
#p[1] and p[_T], for example, synonymous
(_Sigma, _T, _E, _P, _k, _Beta, _Alpha, _C, _v, _Mu, _b, _a, _c, 
 _Tau, _Omega, _Gamma, _Psi, _Chi, _Zeta, _Rho, _F_star, 
 _c_start, _c_end, _o_start, _o_alert) = range(NUMPARAMS)

(_F, _X, _Yi, _Yj, _S, _Ii, _Ij, _Ci, _Cj, _Ri, _Rj, 
 _Iij, _Iji, _R, _N) = range(NUMSTATEVARS)

#the names of the state variables and parameters
p_names = ["Sigma", "T", "E", "P", "k", "Beta", "Alpha", "C",
           "v", "Mu", "b", "a", "c", "Tau", "Omega", "Gamma", "Psi",
           "Chi", "Zeta", "Rho", "F_star", "Control_start", "Control_end",
           "Outbreak_start"]

s_names = ["F", "X", "Yi", "Yj", "S", "Ii", "Ij", "Ci", "Cj", "Ri",
          "Rj", "Iij", "Iji", "R", "N"]

def original_params(release_ratio, N0, c_start, c_end, otime):
    '''Returns the parameters used in the manuscript as an array
    '''
    Sigma = 1/14.0
    T = 18.5
    E = 8.0
    P = 0.7
    k = 2.0
    Beta = 1.0
    #Alpha = 1.5e-8
    Alpha = np.log(P / Sigma) / ((2 * k * N0 * E ) ** Beta)
    C = release_ratio
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
    o_alert = 0

    return np.array([Sigma, T, E, P, k, Beta, Alpha, C, v, Mu, b, a, c, Tau,
                     Omega, Gamma, Psi, Chi, Zeta, Rho, F_star, c_start, 
                     c_end, otime, o_alert]).T

def generate_params(runs, release_ratio, N0, c_start, c_end, otime):

    original = original_params(release_ratio, N0, c_start, c_end, otime)
    if runs == 1:
        return original
    else:
        runs = runs - 1

    #PARAMETERS
    #adult mosqiuto death rate
    Sigma = np.random.uniform(low = 1/15.0, high = 1/3.0, size=runs)

    #Mosquito Generation Time egg->emerging adult
    T = np.random.uniform(low = 16.9, high = 20.1, size=runs)

    #Daily egg production rate per adult mosquito (a female lays 2E)
    E = np.random.uniform(low = 7.0, high = 9.0, size=runs)

    #number of offspring produced by each adult per day that will 
    #survive to adulthood in the absence of density dependent mortality 
    #(i.e. E adjusted for density-independent egg-to-adult survival) 
    #estimated value, calculated using field data, depends 
    #on value of s (P/s is net reproductive rate)
    P = np.random.uniform(low = 0.2, high = 0.7, size=runs)

    #Average number of vectors (adult female mosquitoes) per host, 
    #initial pop is N0
    k = np.random.uniform(low = 0.3, high = 20, size=runs)

    #Strength of larval density dependence
    Beta = np.random.uniform(low = 0.302, high = 1.5, size=runs)

    #breeding site multiplier
    Alpha = np.log(P / Sigma) / ((2 * k * N0 * E ) ** Beta)

    #Maintained ratio of RIDL males to pre-release equilibrium number of 
    #adult males (constant release policy)
    C = np.empty(runs)
    C.fill(release_ratio)

    #Human per capita birth rate (per day) Equal to human death rate
    v = np.random.uniform(low = 1/(60*365.0), high = 1/(68*365.0), size=runs)

    #Human per capita birth rate (per day)
    Mu = v

    #biting rate (number of bites per mosquito per day)
    b = np.random.uniform(low = 0.33, high = 1.0, size=runs)

    #Proportion of bites that successfully infect a susceptible human
    a = np.random.uniform(low = 0.25, high = 0.75, size=runs)

    #Proportion of bites that successfully infect a susceptible mosquito
    c = np.random.uniform(low = 0.20, high = 0.75, size=runs)

    #Virus latent period in humans (days) Intrinsic incubation period
    Tau = np.random.uniform(low = 3.0, high = 12.0, size=runs)

    #Virus latent period in vectors (days) Extrinsic incubation period
    Omega = np.random.uniform(low = 7.0, high = 14.0, size=runs)

    #Human recovery rate (per day) 1/infectious period
    Gamma = np.random.uniform(low = 1/10.0, high = 1/2.0, size=runs)

    #Rate at which humans lose cross-immunity (per day)
    Psi = np.random.uniform(low = 1/(365*5/12.0), high = 1/(365*2/12.0),
                            size=runs)

    #Increased host susceptibility due to ADE
    Chi = np.random.uniform(low = 1.0, high = 3.0, size=runs)

    #Alternative to Chi, increased transmissibility due to ADE
    Zeta = np.ones(runs)

    #Proportion of hosts that recover from secondary infection 
    #(1-Rho die from DHF/DSS)
    Rho = np.random.uniform(low = 0.9935, high = 1, size=runs)

    #Stable equilibrium of pre-release mosquito population
    F_star = ((1/Alpha * np.log(P / Sigma)) ** (1/Beta)) / 2 * E

    #start and stop of control effort, in days
    control_start = np.empty(runs)
    control_start.fill(c_start)
    control_end = np.empty(runs)
    control_end.fill(c_end)

    #time of outbreak
    outbreak = np.empty(runs)
    outbreak.fill(otime)

    #notification of outbreak
    out_alert = np.empty(runs)
    out_alert.fill(0.0)

    sampled = np.vstack((Sigma, T, E, P, k, Beta, Alpha, C, v, Mu, b, a, c, 
                       Tau, Omega, Gamma, Psi, Chi, Zeta, Rho, F_star, 
                       control_start, control_end, outbreak, out_alert)).T

    return np.vstack((original, sampled))

def generate_start_states(p, n):
    
    if len(p.shape) == 1:
        z=0.0
        sstates = np.array([n, n, z, z, n, z, 
                           z, z, z, z, z, z, z, z, n])
    else:
        runs = p.shape[0]
        #fstar = p[:, _F_star].reshape(runs,1)
        n0 = np.empty(runs).reshape(runs,1)
        n0.fill(n)
        z = np.zeros(runs).reshape(runs,1)
        sstates = np.hstack((n0, n0, z, z, n0, z,
                            z, z, z, z, z, z, z, z, n0))

    return sstates

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
    R:        Recovered from secondary infection, immune to all serotypes
    '''
    #state variables at time t
    s = [x if x > .001 else 0 for x in  Y(t)]

    #state variables at time t - T delay (mosquito generation time)
    ds = [x if x > .001 else 0 for x in Y(t-p[_T])]
    
    #state variables at time t - omega delay (extrinsic incubation period)
    os = [x if x > .001 else 0 for x in Y(t-p[_Omega])]

    #state variables at time t - tau delay (inrinsic incubation period)
    ts = [x if x > .001 else 0 for x in Y(t-p[_Tau])]
    #control switch
    if t <= p[_c_start]:
        C_tilde = 0
    elif t <= p[_c_end] and t >= p[_c_start]:
        C_tilde = p[_C]
    else:
        C_tilde = 0

    #equation 5
    if t >= p[_o_start] and t <= p[_o_start]+max([p[_Omega], p[_Tau]]):
        s[_Ii] = 10.0
        os[_Ii] = 10.0
        ts[_Ii] = 10.0

        s[_Ij] = 10.0
        os[_Ij] = 10.0
        ts[_Ij] = 10.0
        if p[_o_alert] == 0.0:
            print("Outbreak Started at {}".format(t))
            p[_o_alert] = 1.0

    Ii = (np.exp(-p[_Mu] * p[_Tau]) * (p[_a] * p[_b]) *
          (ts[_S] * ts[_Yi] / ts[_N]) - s[_Ii] * (p[_Gamma] + p[_Mu]))
    Ij = (np.exp(-p[_Mu] * p[_Tau]) * (p[_a] * p[_b]) *
          (ts[_S] * ts[_Yj] / ts[_N]) - s[_Ij] * (p[_Gamma] + p[_Mu]))

    #equation 1
    if ds[_F] == 0:
        F = 0
    else:
        #F = ((p[_P] * ds[_F]) * (ds[_F] / (ds[_F] + C_tilde * p[_F_star])) * 
        F = ((p[_P] * ds[_F]) * (ds[_F] / (ds[_F] + C_tilde * p[_k] * s[_N])) * 
                np.exp(-(p[_Alpha] * (2*p[_E]*ds[_F]) ** p[_Beta] )) - 
                p[_Sigma] * s[_F] )
    
    #equation 2
    if ds[_X] == 0:
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

def zip_p(a, b):
    for i, j in zip(a, b):
        print('{}: {}'.format(i, j))

def print_run_info(s, p, r):
    print("\n##Starting State for run {}:".format(r))
    zip_p(s_names, s)
    print("##Parameters for run {}:".format(r))
    zip_p(p_names, p)

def main():
    np.set_printoptions(suppress=True, precision=10)
    np.seterr(all='raise')
    args = docopt(__doc__)
    runs = int(args['--runs'])

    N0 = float(args['--initialpop'])
    outbreak_start = 365 * float(args['--outbreak'])
    control_years = float(args['--releasetime'])
    pre_control_years = float(args['--timebefore'])
    post_control_years = float(args['--timeafter'])
    release_ratio = float(args['--releaseratio'])
    control_start = 365.0 * pre_control_years
    control_end = 365.0 * control_years + control_start
    simulation_end =  365.0 * post_control_years + control_end
    dt = 1.0

    tt = np.linspace(0, simulation_end, simulation_end*dt, endpoint=True)
    yys = []
    parameters = generate_params(runs, release_ratio, N0, control_start, 
                                 control_end, outbreak_start)
    start_states = generate_start_states(parameters, N0)

    finished_runs = []
    for r in range(runs):
        #states for this run
        ss = start_states if runs == 1 else start_states[r, :]
        #history Function
        g = lambda t : ss
        #parameters for this run
        par = parameters if runs == 1 else parameters[r, :]
        print_run_info(ss, par, r)
        try:
            yys.append(dde.ddeint(RIDL_dde, g, tt, fargs=(par, )))
        except FloatingPointError as err:
            print("RUN {} FAILED: {}".format(r, err))
        else:
            finished_runs.append(r)

    out = open('run_parameters.csv', 'w')
    out.write(','.join(p_names)+'\n')
    if len(finished_runs) == 1:
        out.write(','.join(str(x) for x in parameters)+'\n')
    else:
        for r in finished_runs:
            out.write(','.join(str(x) for x in parameters[r, :])+'\n')
    out.close()

    gr = len(finished_runs)
    if gr >= 1:
        vars_to_graph = [_F, _Yi, _S, _Ii, _Ri, _Iij, _R]
        num_vars = len(vars_to_graph)
        f, axarr = plt.subplots(num_vars)
        graph_start = 0#int(control_start - 300)
        graph_end = int(simulation_end)

        for i, v in enumerate(vars_to_graph):
            yy = np.array([y[graph_start:graph_end, v] for y in yys])
            axarr[i].plot(tt[graph_start:graph_end], yy.T)
            axarr[i].set_xlim(graph_start, graph_end)
            y_lim = axarr[i].get_ybound()
            axarr[i].set_ylim(0, y_lim[1])
            axarr[i].axvline(control_start, color='black', linestyle='--')
            axarr[i].axvline(control_end, color='black', linestyle='--')
            if i+1 != num_vars:
                axarr[i].set_xticklabels([])
            else:
                axarr[i].set_xlabel('Days')

            axarr[i].set_ylabel(s_names[v], rotation='horizontal')
            axarr[i].yaxis.set_label_position("right")

        plt.figlegend(labels=('Run_{}'.format(x) for x in finished_runs),
                      handles= axarr[-1].get_lines(), loc=1)
        plt.show()

if __name__ == '__main__':
    main()
