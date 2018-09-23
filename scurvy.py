import scipy
from scipy.integrate import *
from scipy.interpolate import *
import matplotlib
from matplotlib import rc
from matplotlib import axes
from numpy import *
from pylab import *
from scipy.integrate import cumtrapz
from scipy.optimize import minimize

import numpy.random
# import time
import os
import os.path

import opalreader as op

#Uncomment the following if you want to use LaTeX in figures 
rc('font',**{'family':'serif'})
rc('mathtext',fontset='cm')
rc('mathtext',rm='stix')
rc('text', usetex=True)
# #add amsmath to the preamble
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"] 

close('all')
ioff()

thetamin = 0.01
alpha=0.1

def kappa_OPAL(temp, n15, kappafun):
    '''
    wrapper for opalreader's table; but kappafun needs to be defined first with op.opalread()
    '''
    lgT, lgR = op.lgTR(temp, n15)
    return 10.**(kappafun(lgT, lgR))

def verte(piar, mdot, omega, sigma, kappafun):
    # Tc in kK, nc in 1e15

    s=0. ; p=1.; z=0.; q=0. ; theta=1. # boundary conditions at the midplane

    ds=1e-3 # what should define ds?
   
    pi1, pi2, pi3, pi4 = piar

    # mdot in 1e-11 Msun/yr
    # sigma in g/cm^2
    # omega in seconds
    Tc = pi3 / alpha * 1215.33 * mdot * omega / sigma 
    nc = 2080.93 * omega * sigma / sqrt(Tc)  / sqrt(pi1) / pi2
    
    #    thetasurface = (16./3.*pi4/tau0)**0.25 *0.
    #    psurface = sqrt(3./16./2.**0.125*pi1*pi2/pi4*thetasurface**(17./2.))*0.
    
    while((s<1.) & (theta > 0.) & (p>0.)):
        # half-step:
        p1 = p - pi1 * pi2 * z * (ds/2.)
        z1 = z + pi2 * theta / p * (ds/2.)
        q1 = q + pi3 * theta * (ds/2.)
        thetapos = (theta+abs(theta-thetamin*2.))/2.+thetamin
        kappa = kappa_OPAL(Tc*thetapos, nc*p/thetapos, kappafun)
        theta1 = theta - pi4 * q * kappa * (ds/2.)
        theta1pos = (theta1+abs(theta1-thetamin*2.))/2.+thetamin
        kappa1 = kappa_OPAL(Tc*theta1pos, nc*p1/theta1pos, kappafun)
        # full step:
        p -= pi1 * pi2 * z1 * ds
        z += pi2 * theta1 / p1 * ds
        q += pi3 * theta1 * ds
        theta -= pi4 * q1 * kappa1 *ds
        s += ds
        #    print(str(s)+" "+str(p)+" "+str(z)+" "+str(q)+" "+str(theta)+"\n")
    
    return (p-0.)**2 + (z-1.)**2 +  (q-1.)**2 +  (theta-0.)**2 + (s-1.)**2

def pisolve(mdot=1., omega=0.1, sigma=10., pistart=[7.6, 0.47, 1.1, 0.4], kappafun = None):

    if(kappafun == None):
        # linking an OPAL table
        kappafun = op.opalread(infile='GN93hz.txt', tableno = 73)

        
    res = scipy.optimize.minimize(verte, pistart, args = (mdot, omega, sigma, kappafun),
                                  method='Nelder-Mead', options={'maxiter': 1e3})
    return res

def curverestore():
    # calculates Pi1-4 and global disc parameters in the Sigma-mdot plane

    sigma1 = 1. ; sigma2 = 100. ; nsig = 10
    sig = (sigma2/sigma1)**(arange(nsig)/double(nsig-1))*sigma1
    mdot1 = 0.01 ; mdot2 = 10. ; nmdot = 15
    mdot = (mdot2/mdot1)**(arange(nmdot)/double(nmdot-1))*mdot1

    r9 = 10.
    omega = 0.446191 *r9**(-1.5) # for M=1.5Msun

    pistore_sig = [8.0, 0.47, 1.1, 0.4]

    deltaT = zeros([nsig, nmdot], dtype=double)
    piar1 = zeros([nsig, nmdot], dtype=double)
    piar2 = zeros([nsig, nmdot], dtype=double)
    piar3 = zeros([nsig, nmdot], dtype=double)
    piar4 = zeros([nsig, nmdot], dtype=double)

    # linking an OPAL table
    kappafun = op.opalread(infile='GN93hz.txt', tableno = 73)

    fout = open('pitable.dat', 'w')
    for ksig in arange(nsig):
        pistore = pistore_sig
        for kmdot in arange(nmdot):
            res = pisolve(mdot=mdot[kmdot], omega=omega, sigma=sig[ksig], pistart=pistore, kappafun=kappafun)
            if(kmdot>0):
                pistore = res.x
            else:
                pistore_sig = res.x
            pi1, pi2, pi3, pi4 = res.x
            Tc = pi3 / alpha * 1215.33 * mdot[kmdot] * omega / sig[ksig]
            nc = 2080.93 * omega * sig[ksig] / sqrt(Tc)  / sqrt(pi1) / pi2
            kappa0 = kappa_OPAL(Tc, nc, kappafun)
            if(kappa0<=0.):
                print(kappa0)
            Teq = 15.7921 * (omega**2*kappa0*sig[ksig]/pi4*mdot[kmdot])**0.25
            deltaT[ksig, kmdot] = Tc-Teq
            piar1[ksig, kmdot]= pi1
            piar2[ksig, kmdot]= pi2
            piar3[ksig, kmdot]= pi3
            piar4[ksig, kmdot]= pi4
            fout.write(str(sig[ksig])+" "+str(mdot[kmdot])+" "+str(deltaT[ksig, kmdot])+" "
                       +str(pi1)+" "+str(pi2)+" "+str(pi3)+" "+str(pi4)+"\n")
            print(str(sig[ksig])+" "+str(mdot[kmdot])+" "+str(deltaT[ksig, kmdot])+" "
                       +str(pi1)+" "+str(pi2)+" "+str(pi3)+" "+str(pi4))
    fout.close()

    clf()
    contourf(sig, mdot, transpose(deltaT))
    xscale('log') ; yscale('log')
    savefig('deltat.png')
    close('all')
