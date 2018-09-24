import scipy
from scipy.integrate import *
from scipy.interpolate import *
from numpy import *
from scipy.integrate import cumtrapz
from scipy.optimize import minimize
import time 

import numpy.random
# import time
import os
import os.path

import opalreader as op

thetamin = 0.01 # floor for temperature
alpha=0.1 # viscosity parameter
ifplot = False # if we are going to plot figures immediately

if(ifplot):
    import splot as pt

def kappa_OPAL(temp, n15, kappafun):
    '''
    wrapper for opalreader's table; but kappafun needs to be defined first with op.opalread()
    '''
    lgT, lgR = op.lgTR(temp, n15)
    return 10.**(kappafun(lgT, lgR))

def verte(piar, mdot, omega, sigma, kappafun):
    '''
    solves the vertical structure equations for given Pi1-4, mdot [1e-11 Msun/yr], omega [s^-1], sigma [g/cm^2], and opacity function
    returns a residual function equal to zero iff the BCs at the surface are satisfied
    '''
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
    '''
    searches the set of Pi parameters (Ketsaris&Shakura 1998) that zeros the verte function (see above)
    mdot in 1e-11 Msun/yr, omega in 1/s, sigma in g/cm^2
    '''
    
    if(kappafun == None): 
        # linking an OPAL table if not linked by the calling routine
        kappafun = op.opalread(infile='GN93hz.txt', tableno = 73)

    res = scipy.optimize.minimize(verte, pistart, args = (mdot, omega, sigma, kappafun),
                                  method='Nelder-Mead', options={'maxiter': 1e3})
    # Nelder-Mead is a simplex; probably, best-suited for the problem
    
    return res

def dtfun(res, sig, mdot, omega, kappafun):
    '''
    calculated temperature difference: temperature from pressure and from heat balance should be equal. dt=0 defines the S-curve
    Inputs:
    res structure produced by scipy.optimize.minimize, sigma in g/cm^2, mdot in 1e-11 Msun/yr, omega (Keplerian, in 1/s), and opacity function
    '''
    pi1, pi2, pi3, pi4 = res.x
    Tc = pi3 / alpha * 1215.33 * mdot * omega / sig
    nc = 2080.93 * omega * sig / sqrt(Tc)  / sqrt(pi1) / pi2
    kappa0 = kappa_OPAL(Tc, nc, kappafun)
    if(kappa0<=0.):
        print(kappa0)
    Teq = 15.7921 * (omega**2*kappa0*sig/pi4*mdot)**0.25
    return Tc-Teq # in kK

def curverestore():
    # calculates Pi1-4 and global disc parameters in the Sigma-mdot plane

    sigma1 = 1. ; sigma2 = 100. ; nsig = 20
    sig = (sigma2/sigma1)**(arange(nsig)/double(nsig-1))*sigma1
    mdot1 = 0.01 ; mdot2 = 10. ; nmdot = 35
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
            #            Tc = pi3 / alpha * 1215.33 * mdot[kmdot] * omega / sig[ksig]
            #            nc = 2080.93 * omega * sig[ksig] / sqrt(Tc)  / sqrt(pi1) / pi2
            #            kappa0 = kappa_OPAL(Tc, nc, kappafun)
            #            if(kappa0<=0.):
            #                print(kappa0)
            #            Teq = 15.7921 * (omega**2*kappa0*sig[ksig]/pi4*mdot[kmdot])**0.25
            deltaT[ksig, kmdot] = dtfun(res, sig[ksig], mdot[kmdot], omega, kappafun)
            piar1[ksig, kmdot]= pi1
            piar2[ksig, kmdot]= pi2
            piar3[ksig, kmdot]= pi3
            piar4[ksig, kmdot]= pi4
            fout.write(str(sig[ksig])+" "+str(mdot[kmdot])+" "+str(deltaT[ksig, kmdot])+" "
                       +str(pi1)+" "+str(pi2)+" "+str(pi3)+" "+str(pi4)+"\n")
            print(str(sig[ksig])+" "+str(mdot[kmdot])+" "+str(deltaT[ksig, kmdot])+" "
                       +str(pi1)+" "+str(pi2)+" "+str(pi3)+" "+str(pi4))
    fout.close()

############################################################
def findsig(mdot, omega, kappa=None, pistart=[8.0, 0.47, 1.1, 0.4], chord=False):

    if(kappa == None):
        # linking an OPAL table
        kappafun = op.opalread(infile='GN93hz.txt', tableno = 73)
    else:
        kappafun = kappa

    tstart = time.time()
        
    sigma1 = 0.1; sigma2 = 100. ; stol = 1e-3
    res1 = pisolve(mdot=mdot, omega=omega, sigma=sigma1, pistart=pistart, kappafun=kappafun)
    dt1 = dtfun(res1, sigma1, mdot, omega, kappafun)
    pistore1 = res1.x
    res2 = pisolve(mdot=mdot, omega=omega, sigma=sigma2, pistart=pistart, kappafun=kappafun)
    dt2 = dtfun(res2, sigma2, mdot, omega, kappafun)
    pistore2 = res2.x

    while(abs((sigma1-sigma2)/(sigma1+sigma2)) > stol):

        if(chord):
            sigma = exp((log(sigma1)*dt2 - log(sigma2)*dt1)/(dt2-dt1))
        else:
            sigma = sqrt(sigma1*sigma2)
        # sqrt(sigma1*sigma2)
        piest = (pistore1+pistore2)/2.
        res = pisolve(mdot=mdot, omega=omega, sigma=sigma, pistart=piest, kappafun=kappafun)
        dt = dtfun(res, sigma, mdot, omega, kappafun)
        if((dt*dt1) < 0.):
            sigma2 = sigma
            dt2 = dt
            pistore2 = res.x
        else:
            sigma1 = sigma
            dt1 = dt
            pistore1 = res.x
        print(sigma)
        
    tend = time.time()
    print("Search for sigma took "+str(tend-tstart)+"s = "+str((tend-tstart)/60.)+"min")
    
    return (sigma1+sigma2)/2., piest
    
def searchforsigma(r9 = 10.):
    '''
    finds the values of sigma for a range of mdot, for fixed alpha and radius [10^9cm]. 
    '''

    mdot1 = 0.01 ; mdot2 = 10. ; nmdot = 15
    mdot = (mdot2/mdot1)**(arange(nmdot)/double(nmdot-1))*mdot1

    omega = 0.446191 *r9**(-1.5) # for M=1.5Msun
    pistore = [8.0, 0.47, 1.1, 0.4]

    # linking an OPAL table
    kappafun = op.opalread(infile='GN93hz.txt', tableno = 73)

    sigma = zeros(nmdot, dtype=double)
    fout = open('sigtable.dat', 'w')
    
    for kmdot in arange(nmdot):
        sigmatmp, piout = findsig(mdot[kmdot], omega, kappa=kappafun, pistart=pistore)
        pistart = piout ; sigma[kmdot] = sigmatmp
        print(str(mdot[kmdot])+" "+str(sigma[kmdot]))
        fout.write(str(mdot[kmdot])+" "+str(sigma[kmdot])+"\n")
        fout.flush()

    teff = 22.6708 * mdot**0.25 / r9**0.75

    if(ifplot):
        pt.splot(sigma, teff)
    
