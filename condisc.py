from scipy.integrate import *
from scipy.interpolate import *
import matplotlib
from matplotlib import rc
from matplotlib import axes
from numpy import *
from pylab import *
from scipy.integrate import cumtrapz

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

# atomic data:

# names:
# names = ['hydrogen', 'helium', 'lithium', 'beryllium', 'boron',
#         'carbon', 'nitrogen', 'oxygen', 'fluorine', 'neon',
#         'sodium', 'magnesium', 'alimunium', 'silicon', '']
el = ['H', 'He',
      'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
      'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
      'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
nel=size(el)
# then goes a huge drop in abundance

#abundances: (Grevesse & Sauval 1996 standard abundances)
abund = asarray([12., 10.99,
         1.16, 1.15, 2.6, 8.55, 7.97, 8.87, 4.56, 8.08,
         6.44, 7.58, 6.47, 7.55, 5.45, 7.33, 5.5, 6.52,
         5.12, 6.36, 3.17, 5.02, 4.0, 5.67, 5.39, 7.5, 4.92, 6.25, 4.21, 4.6])

# ionization potentials:
iop = asarray([13.59844, 24.58741,
       5.39172, 9.3227, 8.29803, 11.26030, 14.53414, 13.61806, 17.42282, 21.5646,
       5.13908, 7.64624, 5.98577, 8.15169, 10.48669, 10.36001, 12.96764, 15.75962,
       4.34066, 6.11316, 6.5615, 6.8281, 6.7462, 6.7665, 7.43402, 7.9024, 7.8810, 7.6398, 7.72638, 9.3942])
# from H to Zn
# borrowed from: David R. Lide (ed), CRC Handbook of Chemistry and Physics, 84th Edition. CRC Press. Boca Raton, Florida, 2003; Section 10, Atomic, Molecular, and Optical Physics; Ionization Potentials of Atoms and Atomic Ions

sahacoeff = 1.30959e-5
ioconvert = 11.6046 # converting eV to kK
xtol = 1e-5
ttol = 1e-5
sigman = 1. # (in 1e-16 cm^2 units; neutral collision cross-section)
alpha = 0.1
Pi1 = 5.
Pi2 = 0.5
Pi3 = 1.
Pi4 = 0.4
lame = 0.0 # mixing length parameter (set lame=0 for no convection)
mass1 = 1.2

xrenorm = ((10.)**(abund-12.)).sum()  # maximal concentration of neutral atoms with respect to nH
print("X renormalization "+str(xrenorm))

# for k in arange(nel):
#     print(el[k]+": Z="+str(abund[k])+"; IP="+str(iop[k])+"\n")

###############################################################
# Pi-s as functions of optical depth
def pifunctions():
    '''
    interpolates the results of Ketsaris&Shakura 1998 for the coefficients Pi2 and Pi4
    returns two interpolation functions
    '''
    nlnt=31
    lnt = linspace(0., 6., nlnt, endpoint=True)
    pi2data = asarray([0.908, 0.876, 0.838, 0.798, 0.756,
                       0.716, 0.679, 0.647, 0.619, 0.596,
                       0.576, 0.56, 0.546, 0.534, 0.524,
                       0.515, 0.508, 0.501, 0.496, 0.491,
                       0.487, 0.483, 0.48, 0.477, 0.475,
                       0.473, 0.471, 0.469, 0.468, 0.466,
                       0.465])
    pi4data = zeros(nlnt, dtype=double)+0.399
    pi4data_start =  asarray([0.136, 0.185, 0.237, 0.286, 0.326,
                              0.354, 0.371, 0.383, 0.389, 0.393,
                              0.395, 0.397, 0.398, 0.398, 0.398,
                              0.398])
    pi4data[:size(pi4data_start)] = pi4data_start[:]

    pi2fun = interp1d(lnt, pi2data, bounds_error=False,fill_value='extrapolate')
    pi4fun = interp1d(lnt, pi4data, bounds_error=False,fill_value='extrapolate')

    return pi2fun, pi4fun

##
# let us define the two functions in global scope:
pi2fun, pi4fun = pifunctions()
##

####################################################
#       ionization fraction (may be >1 as normalized by hydrogen)

def abundfun(temp, n15, x): # temperature in kK, n15 = nH/10^{15}cm^{-3}
    z = (10.)**(abund-12.) # physical abundances with respect to hydrogen

    return (z/(1.+sahacoeff*x*n15/temp**1.5*exp(ioconvert * iop/temp))).sum()

def findiofr(temp, n15, xseed = 0.5):
    '''
    finds ionization fraction by direct iterations
    '''
    x1=0. ; x2=1.
    x=xseed
    while((abs(x1/x2-1.)>xtol) & ((x1+x2)>1e-12)):
        x2 = abundfun(temp, n15, x1)
        #        print("x = "+str(x1)+" = "+str(x2))
        x1 = abundfun(temp, n15, x2)

    return (x1+x2)/2.

def allx(temp, n15, x):
    # calculates the abundances of individual elements
    xi=zeros(nel, dtype=double)
    for k in arange(nel):
        xi=1./(1.+sahacoeff*x*n15/temp**1.5*exp(ioconvert * iop/temp))
    z = (10.)**(abund-12.) # physical abundances with respect to hydrogen
    xchecksum = (z*xi).sum()
    print("T = "+str(temp)+"kK; n = "+str(n15*1e15)+" cm^{-3}")
    print("total ionization fraction "+str(x)+" = "+str(xchecksum))
    return xi

################################################################
def condy(temp, n15, x):
    '''
    calculates conductivity as a function of temperature (in kK) and density (10^{15} cm^{-3})
    '''

    # cross-sections in 1e-16 cm^2 units
    sigmaC = 1e5 / temp**2
    #    x = findiofr(temp, n15)
    o13 = 20571.9 / sqrt(temp) * x/ (sigman * (xrenorm - x) + sigmaC * x)
    o13_C = 20571.9 / sqrt(temp) / sigmaC
    o13_n = 20571.9 / sqrt(temp) * x / (xrenorm-x) / sigman
    
    return o13, o13_C, o13_n 
#######################################################################
# making a disc: opacity and S-curve
def kappa_Kr(temp, n15):
    return 1978.32*temp**(-3.5)*n15 # Kramers's opacity (cm^2/g) law for X=0.7, from wikipedia; to be replaced by OPAL

def kappa_OPAL(temp, n15, kappafun):
    '''
    wrapper for opalreader's table; but kappafun needs to be defined first with op.opalread()
    '''
    lgT, lgR = op.lgTR(temp, n15)
    return 10.**(kappafun(lgT, lgR))

def taufun(temp, n15, r9=1., mdot11=1., opacity = None):
    '''
    estimates the optical depth for given central temperature and density
    ignores mu!
    "opacity " is the function that calculates the opacities from temp and n15
    should be kappafun if we use OPAL tables; if None, Kramers approximation is taken
    '''
    if (opacity == None):
        return 442.761*kappa_Kr(temp, n15)*mdot11*sqrt(mass1/r9**3)/temp/alpha*Pi3
    else:
        return 442.761*kappa_OPAL(temp, n15, opacity)*mdot11*sqrt(mass1/r9**3)/temp/alpha*Pi3

def sigfun(temp, n15, r9=1., mdot11=1.):
    # surface density, g/cm^2
    return 442.761*Pi3*mdot11*sqrt(mass1/r9**3)/temp/alpha

def nc(temp, r9=1., mdot11=1.):
    # provides nH in 1e15 units
    #  return 0.208757*mdot11/alpha/2./Pi2/r9**2/sqrt(temp)
    return 167831. /alpha / temp**1.5 *mdot11*mass1/r9**3 * (Pi3/Pi2/sqrt(Pi1))
    #    return 3.35663e5 * mass1 * mdot11 /alpha/Pi2/temp**1.5/r9**3

def tempfun(temp, r9=1., mdot11=1., opacity = None):
    # should be zero at proper Tc
    n15 = nc(temp, r9=r9, mdot11=mdot11)
    tau = taufun(temp, n15, r9=r9, mdot11=mdot11, opacity = opacity)
    fluxratio = 5.678e-6*temp**4/mass1/mdot11*r9**3
    csc = 9.58348e-06 * sqrt(temp) # speed of sound in light units
    prat = 54745.2 * n15 / temp**3
    return fluxratio * (32./3.*Pi4/tau + lame * csc * prat)-1.
    
def tempsolve(r9=1., mdot11=0.1, opacity = None):
    '''
    searches the optimal value of Tc
    "opacity " is the name of the function that calculates the opacities from temp and n15
    should be kappa_OPAL if we use OPAL tables; if None, Kramers approximation is taken
    '''
    
    tc1=0.1 ; tc2=100. 
    f1=tempfun(tc1, opacity = opacity) ; f2=tempfun(tc2, opacity = opacity)
    
    while(abs(tc2/tc1-1.)>ttol):
        tc=sqrt(tc1*tc2)
        f=tempfun(tc, r9=r9, mdot11=mdot11, opacity = opacity)
        if((f1*f)>0.):
            tc1=tc
            f1=f
        else:
            tc2=tc
            f2=f
            #            print("new T = "+str(tc))

    return (tc1+tc2)/2.

#########################################################################
# making the pictures
def xicond():
    '''
    calculates ionization fractions and conductivities for different nH and temperatures
    '''
    nar = asarray([1., 1e3, 1e6])
    nn = size(nar)
    temp1 = 50. ; temp2 = .5 ; ntemp=30
    temp = (temp2/temp1)**(arange(ntemp, dtype=double)/double(ntemp))*temp1

    x = zeros([ntemp, nn], dtype=double)
    xH = zeros([ntemp, nn], dtype=double)
    xK = zeros([ntemp, nn], dtype=double)
    con = zeros([ntemp, nn], dtype=double)
    con_C = zeros([ntemp, nn], dtype=double)
    con_n = zeros([ntemp, nn], dtype=double)
    
    for kn in arange(nn):
        for kt in arange(ntemp):
            if(kt>0):
                xtmp = findiofr(temp[kt], nar[kn], xseed=xtmp)
            else:
                xtmp = findiofr(temp[kt], nar[kn], xseed=1.)
            x[kt,kn] = xtmp
            xels = allx(temp[kt], nar[kn], xtmp)
            xH[kt,kn] = xels[0] # ionization fraction of hydrogen
            xK[kt,kn] = xels[18] # ionization fraction of potassium
            contmp = condy(temp[kt], nar[kn], xtmp)
            con[kt,kn] = contmp[0] ; con_C[kt,kn] = contmp[1] ; con_n[kt,kn]=contmp[2]

    colorsequence = ['b', 'k', 'r', 'g', 'm']

    # ionization fraction plot:
    clf()
    fig=figure()
    for kn in arange(nn):
        plot(temp, x[:, kn], '-', color=colorsequence[kn])
        plot(temp, xK[:, kn]*(10.)**(abund[18]-12.), ':', color=colorsequence[kn])
        plot(temp, xH[:, kn], '--', color=colorsequence[kn])
    xscale('log')  ;  yscale('log')
    ylim(1e-12, 1.5)
    xlabel('$T$, kK', fontsize=18)
    ylabel(r'$n_{\rm e} / n_{\rm H}$', fontsize=20)
    plt.tick_params(labelsize=16, length=3, width=1., which='minor')
    plt.tick_params(labelsize=16, length=3, width=1., which='major')
    fig.set_size_inches(5, 4)
    fig.tight_layout()
    savefig('iofrele.eps')
    close()
    
    # conductivity plot:
    clf()
    fig=figure()
    for kn in arange(nn):
        plot(temp, con[:, kn], '-', linewidth = kn+1, color='k')
        plot(temp, con_C[:, kn], '--', linewidth = kn+1, color='b')
        plot(temp, con_n[:, kn], ':', linewidth = kn+1, color='r')
    ylim(1e-7, 1e3)
    xscale('log')  ;  yscale('log')
    xlabel('$T$, kK', fontsize=18)  ;  ylabel(r'$\omega, \,10^{13}{\rm \, s}^{-1}$', fontsize=18)
    tick_params(labelsize=16, length=3, width=1., which='minor')
    tick_params(labelsize=16, length=3, width=1., which='major')
    fig.set_size_inches(5, 4)
    fig.tight_layout()
    savefig('conde.eps')
    close()
    
#######################################################################
# disc model plots
def scurve():
    r9 = 1. # radius in 10^9 cm (fixed)
    # corotation radius is 3.5\times 10^9 cm for GRO10
    mdot1 = 2. ; mdot2 = 2e-4 # 1e-11 Msun/yr units
    nmdot=30
    mdot = (mdot2 / mdot1)**(arange(nmdot, dtype=double)/double(nmdot)) * mdot1

    temp = zeros(nmdot, dtype=double)
    temp_Kr = zeros(nmdot, dtype=double)
    teff = zeros(nmdot, dtype=double)
    sig = zeros(nmdot, dtype=double)
    sig_Kr = zeros(nmdot, dtype=double)
    iof = zeros(nmdot, dtype=double)
    xstore=0.5
    
    # linking an OPAL table
    kappafun = op.opalread(infile='GN93hz.txt', tableno = 66)
    
    for k in arange(nmdot):
        # OPAL:
        temptmp = tempsolve(r9=r9, mdot11=mdot[k], opacity = kappafun)
        n15 = nc(temptmp, r9=r9, mdot11=mdot[k])
        xstore=findiofr(temptmp, n15, xseed=xstore)
        iof[k]=xstore
        sig[k] = sigfun(temptmp, n15, r9=r9, mdot11=mdot[k])
        temp[k] = temptmp
        # Kramers:
        temptmp = tempsolve(r9=r9, mdot11=mdot[k])
        n15 = nc(temptmp, r9=r9, mdot11=mdot[k])
        sig_Kr[k] = sigfun(temptmp, n15, r9=r9, mdot11=mdot[k])
        temp_Kr[k] = temptmp
        teff[k] = 20.4853 * (mass1 * mdot[k] / r9**3)**0.25
        print("Teff = "+str(teff[k]))

    # Lasota's points:
    sig1=39.9*(alpha/0.1)**(-0.8)*(r9*0.1)**1.11*mass1**(-0.37) # g/cm^2
    sig2=74.6*(alpha/0.1)**(-0.83)*(r9*0.1)**1.18*mass1**(-0.4) # g/cm^2
    teff1=6.890*(r9*0.1)**(-0.09)*mass1**0.03
    teff2=5.210*(r9*0.1)**(-0.1)*mass1**0.04
    
    clf()
    plot(sig, teff, '.k')
    plot(sig_Kr, teff, '-g')
    #    plot(teff**4*(sig.mean()/(teff**4).mean()), teff, 'b')
    #    plot(teff**3*(sig.mean()/(teff**3).mean()), teff, 'r')
    plot([sig1, sig2], [teff1, teff2], 'ob')
    xscale('log') #  ;  yscale('log')
    ylabel(r'$T_{\rm eff}$, kK')  ;  xlabel(r'$\Sigma$, g\,cm$^{-2}$')
    savefig('scurve.eps')
    savefig('scurve.png')
    close()
    clf()
    plot(sig, iof, '.k')
    xscale('log')   ;  yscale('log')
    ylabel(r'$x$')  ;  xlabel(r'$\Sigma$, g\,cm$^{-2}$')
    savefig('scurve_iof.eps')
    savefig('scurve_iof.png')
    close()
    
