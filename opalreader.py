from scipy.integrate import *
from scipy.interpolate import *
# import matplotlib
# from matplotlib import rc
# from matplotlib import axes
from numpy import *
# from pylab import *
from scipy.integrate import cumtrapz

import numpy.random
# import time
import os
import os.path
from sys import exit

# Uncomment the following if you want to use LaTeX in figures 
# rc('font',**{'family':'serif'})
# rc('mathtext',fontset='cm')
# rc('mathtext',rm='stix')
# rc('text', usetex=True)
# #add amsmath to the preamble
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"] 

# close('all')
# ioff()

def lgTR(temp, n15):
    '''
    converts temperature in kK and density in 1e15cm-3 to lg(T,K) and lg(R) variables
    '''
    lgT = log10(temp)+3.
    lgR = -3.*log10(temp)+log10(n15)+0.223398
    return lgT, lgR

def opalread(infile='GN93hz.txt', tableno = 73):
    '''
    reading an ascii opal file
    makes a diagnostic plot;
    returns a function of lgT, lgR
    '''

    nheader = 240 # number of header lines
    nentry = 77 # the number of lines per table
    nt = 70 # number of temperature values
    nr = 19 # number of R values
    
    temprow = 240 + nentry * (tableno-1) + 7

    fin = open(infile, 'r')

    # skipping the header and previous tables:
    
    for k in arange(temprow-1):
        l=fin.readline()
        #        print(l)
    # now, l contains the grid in lgR
    s=l.split()
    lgR = asarray(s[1:], dtype=double)
    if(size(lgR)!=nr):
        print(size(lgR))
        exit("R grid has different dimensions")
    tlist = [] # temperature list
    kappa = zeros([nr, nt], dtype=double)
    kt = 0 # counter for temperature value
    while(size(tlist)<nt):
        l=fin.readline()
        s=l.split()
        ss = size(s)
        if(ss>1):
            tlist.append(s[0]) # temperature            
            kappa[:(ss-1), kt] = s[1:] # opacities
            #            print("lgT ="+str(s[0]))
            kt+=1
    lgT = asarray(tlist, dtype=double)

    # interpolation
    kappafun = interp2d(lgT, lgR, kappa, kind='linear', bounds_error = False)
    return kappafun

def bookplot(rho=1e-8):

    t1=0.5 ; t2=500. # kK
    nt=100
    temp = (t2/t1)**(arange(nt)/double(nt-1))*t1
    n15 = 5.97863e8 * rho
    
    lgt, lgr = lgTR(temp, n15)
    kappafun = opalread()

    kappa = zeros(nt, dtype=double)
    
    for k in arange(nt):
        kappa[k] = (10.)**kappafun(lgt[k], lgr[k])
    
#    clf()
#    plot(temp, kappa, '-k')
#    xscale('log') ; yscale('log')
#    savefig('bookplot.png')#
#    close()
