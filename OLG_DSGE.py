#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 2017

Updated on Fri May 18 2018

@author: klp4

This is a simple OLG model with S-period-lived agents and an aggregate shock
to productivity.  Agents have fixed labor supply and no retirement.

This model is meant soley as a proof-of-concept and works with S=400, i.e. a
quarterly time period.

"""

# Overlapping Generations Model
import numpy as np
import matplotlib.pyplot as plt

# create a definitions function
def Modeldefs(Xp, X, Z, params):
    '''
    This function takes vectors of endogenous and exogenous state variables
    along with a vector of 'jump' variables and returns explicitly defined
    values for consumption, gdp, wages, real interest rates, and transfers
    
    Inputs are:
        Xp: value of capital holdings in next period
        X: value of capital holdings this period
        Z: value of productivity this period
        params: list of parameter values
    
    Output are:
        Y: GDP
        w: wage rate
        r: rental rate on capital
        T: transfer payments
        c: consumption
        u: utiity
    '''
    
    # unpack input vectors
    kp = np.concatenate([Xp, np.zeros(1)])
    k = np.concatenate([np.zeros(1), X])
    z = Z
    
    # find definintion values
    K = np.sum(k)
    Y = K**alpha*(np.exp(z))**(1-alpha)
    w = (1-alpha)*Y
    r = alpha*Y/K
    T = tau*(w + (r - delta)*K)
    c = (1-tau)*(w + (r - delta)*k) + k + T - kp
    # check to make sure no consumptions are too low
    c = np.maximum(c, .0001*np.ones(S))
    u = c**(1-gamma)/(1-gamma)
    
    return K, Y, w, r, T, c, u


def Modeldyn(theta0, params):
    '''
    This function takes vectors of endogenous and exogenous state variables
    along with a vector of 'jump' variables and returns values from the
    characterizing Euler equations.
    
    Inputs are:
        theta: a vector containng (Xpp, Xp, X, Yp, Y, Zp, Z) where:
            Xpp: value of capital in two periods
            Xp: value of capital in next period
            X: value of capital this period
            Yp: value of labor in next period
            Y: value of labor this period
            Zp: value of productivity in next period
            Z: value of productivity this period
        params: list of parameter values
    
    Output are:
        Euler: a vector of Euler equations written so that they are zero at the
            steady state values of X, Y & Z.  This is a 2x1 numpy array. 
    '''
    
    # unpack theat0
    Xpp = theta0[0 : nx]
    Xp = theta0[nx : 2*nx]
    X = theta0[2*nx : 3*nx]
    Yp = theta0[3*nx : 3*nx + ny]
    Y = theta0[3*nx + ny : 3*nx + 2*ny]
    Zp = theta0[3*nx + 2*ny : 3*nx + 2*ny + nz]
    Z = theta0[3*nx + 2*ny+ nz : 3*nx + 2*ny + 2*nz]
    
    # find definitions for now and next period
    K, Y, w, r, T, c, u = Modeldefs(Xp, X, Z, params)
    Kp, Yp, wp, rp, Tp, cp, up = Modeldefs(Xpp, Xp, Zp, params)

    # truncate c vectors for intertemporal Euler equation
    c1 = c[0 : S-1]
    c1p = cp[1 : S]
    E = (c1**(-gamma)) / (beta*c1p**(-gamma)*(1 + (1-tau)*(rp - delta))) - 1.

    return E

'''
Now the main program
'''
# import the modules from LinApp
from LinApp_FindSS import LinApp_FindSS
from LinApp_Deriv import LinApp_Deriv
from LinApp_Solve import LinApp_Solve
from LinApp_SSL import LinApp_SSL

# set parameter values
S = 80
alpha = .35
beta = .99**(80/S)
gamma = 2.5
delta_ann = .08
delta = 1. - (1. - delta_ann)**(80/S)
tau = .05
rho = .9**(80/S)
sigma = .02

# make parameter list to pass to functions
params = (S, alpha, beta, gamma, delta, tau, rho, sigma)

# set LinApp parameters
Zbar = np.array([0.])
nx = S-1
ny = 0
nz = 1
logX = 0
Sylv = 0

# take a guess for steady state values of k
kguess = 0.001
guessXY = kguess*np.ones(nx)

# find the steady state values using LinApp_FindSS
XYbar = LinApp_FindSS(Modeldyn, params, guessXY, Zbar, nx, ny)
kbar = XYbar[0:nx]
# print ('XYbar: ', XYbar)

Kbar, Ybar, wbar, rbar, Tbar, cbar, ubar = Modeldefs(kbar, kbar, Zbar, params) 
Cbar = np.sum(cbar)

plt.subplot(3, 1, 1)
plt.plot(kbar)
plt.xticks([])
plt.title('SS Capital by Age', y=.92)

plt.subplot(3, 1, 2)
plt.plot(cbar)
plt.xticks([])
plt.title('SS Consumption by Age', y=.92)

plt.subplot(3, 1, 3)
plt.plot(ubar)
plt.title('SS Utility by Age', y=.92)

plt.show()


# set up steady state input vector
theta0 = np.concatenate([kbar, kbar, kbar, Zbar, Zbar])

# check SS solution
check = Modeldyn(theta0, params)
print ('check: ', np.max(np.abs(check)))
if np.max(np.abs(check)) > 1.E-6:
    print ('Have NOT found steady state')

# find the derivatives matrices
[AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM] = \
    LinApp_Deriv(Modeldyn, params, theta0, nx, ny, nz, logX)

# set value for NN    
NN = rho
    
# find the policy and jump function coefficients
PP, QQ, RR, SS = \
    LinApp_Solve(AA,BB,CC,DD,FF,GG,HH,JJ,KK,LL,MM,NN,Zbar,Sylv)
#print ('P: ', PP)
#print ('Q: ', QQ)
#print ('R: ', RR)
#print ('S: ', SS)


# generate a history of Z's
nobs = 250
Zhist = np.zeros((nobs+1,1))
for t in range(1, nobs+1):
    Zhist[t,0] = rho*Zhist[t,0] + sigma*np.random.normal(0., 1.)
    
# put SS values and starting values into numpy vectors
XYbar = kbar
X0 = kbar

# simulate the model
Xhist, Yhist = \
    LinApp_SSL(X0, Zhist, XYbar, logX, PP, QQ, RR, SS)
    
# generate non-state variables
Khist = np.zeros(nobs)
Yhist = np.zeros(nobs)
whist = np.zeros(nobs)
rhist = np.zeros(nobs)
Thist = np.zeros(nobs)
chist = np.zeros((nobs,S))
Chist = np.zeros(nobs)
uhist = np.zeros((nobs,S))

for t in range(0, nobs):
    Khist[t], Yhist[t], whist[t], rhist[t], Thist[t], chist[t,:], uhist[t,:] \
        = Modeldefs(Xhist[t+1,:], Xhist[t,:], Zhist[t,:], params) 
    Chist[t] = np.sum(chist[t,:])

    
# plot
plt.subplot(3, 2, 1)
plt.plot(Khist)
plt.xticks([])
plt.title('Capital', y=.92)

plt.subplot(3, 2, 2)
plt.plot(Yhist)
plt.xticks([])
plt.title('GDP', y=.92)

plt.subplot(3, 2, 3)
plt.plot(whist)
plt.xticks([])
plt.title('Wage', y=.92)

plt.subplot(3, 2, 4)
plt.plot(rhist)
plt.xticks([])
plt.title('Interest', y=.92)

plt.subplot(3, 2, 5)
plt.plot(Thist)
plt.title('Taxes', y=.92)

plt.subplot(3, 2, 6)
plt.plot(Chist)
plt.title('Consumption', y=.92)

plt.show()