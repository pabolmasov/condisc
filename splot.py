import matplotlib
from matplotlib import rc
from matplotlib import axes
from pylab import *

#Uncomment the following if you want to use LaTeX in figures 
rc('font',**{'family':'serif'})
rc('mathtext',fontset='cm')
rc('mathtext',rm='stix')
rc('text', usetex=True)
# #add amsmath to the preamble
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"] 

mass1 = 1.5

close('all')
ioff()

def splot(sigma, teff):
    '''
    Simple 2D image of an S-curve in sigma[g/cm^2]-Teff[kK] space 
    '''
    clf()
    plot(sigma, teff, 'k')
    xscale('log')
    ylabel(r'$T_{\rm eff}$, kK') ;   xlabel(r'$\Sigma$, ${\rm g\,cm^{-2}}$')
    savefig('scure.png')
    close('all')
        
def deltatshow(r9 = 10., alpha=0.1):
    # contour of an S-curve based on dT (temperature balance) overplotted with the splot-style S-curve
    
    lines = loadtxt('pitable_titan.dat', comments="#", delimiter=" ", unpack=False)

    sigma = lines[:,0]
    mdot = lines[:,1]
    dt = lines[:,2]
    sun = unique(sigma) ; mun = unique(mdot)
    ns = size(sun);  nm = size(mun)
    dt = transpose(reshape(dt, [ns, nm]))

    lines = loadtxt('sigtable_alpha0.1.dat', comments="#", delimiter=" ", unpack=False)
    sigline = lines[:,1]
    mdotline = lines[:,0]    

    teff = 22.6708 *mun**0.25 / r9**0.75
    teff_line = 22.6708 *mdotline**0.25 / r9**0.75
    
    # Lasota's points:
    sig1=39.9*(alpha/0.1)**(-0.8)*(r9*0.1)**1.11*mass1**(-0.37) # g/cm^2
    sig2=74.6*(alpha/0.1)**(-0.83)*(r9*0.1)**1.18*mass1**(-0.4) # g/cm^2
    teff1=6.890*(r9*0.1)**(-0.09)*mass1**0.03 #*alpha**(1./6.)
    teff2=5.210*(r9*0.1)**(-0.1)*mass1**0.04 #*alpha**(1./6.)

    clf()
    contourf(sun, teff, dt)
    colorbar()
    contour(sun, teff, dt, levels=[0.], colors='w')
    plot(sigline, teff_line, 'k', linewidth=2)
    plot([sig1, sig2], [teff1, teff2], 'ow')
    xscale('log')
    ylabel(r'$T_{\rm eff}$, kK') ;   xlabel(r'$\Sigma$, ${\rm g\,cm^{-2}}$')
    savefig('deltat.png')
    close('all')

def scurvecompare(r9=10.):

    lines = loadtxt('sigtable_alpha1.dat', comments="#", delimiter=" ", unpack=False)
    sigline0 = lines[:,1]
    mdotline0 = lines[:,0]    
    lines = loadtxt('sigtable_alpha0.1.dat', comments="#", delimiter=" ", unpack=False)
    sigline1 = lines[:,1]
    mdotline1 = lines[:,0]    
    lines = loadtxt('sigtable_alpha0.01.dat', comments="#", delimiter=" ", unpack=False)
    sigline2 = lines[:,1]
    mdotline2 = lines[:,0]    

    teff0 = 22.6708 *mdotline0**0.25 / r9**0.75
    teff1 = 22.6708 *mdotline1**0.25 / r9**0.75
    teff2 = 22.6708 *mdotline2**0.25 / r9**0.75

    # Lasota's points:
    Lsig0_1=39.9*(1./0.1)**(-0.8)*(r9*0.1)**1.11*mass1**(-0.37) # g/cm^2
    Lsig0_2=74.6*(1./0.1)**(-0.83)*(r9*0.1)**1.18*mass1**(-0.4) # g/cm^2
    Lsig1_1=39.9*(0.1/0.1)**(-0.8)*(r9*0.1)**1.11*mass1**(-0.37) # g/cm^2
    Lsig1_2=74.6*(0.1/0.1)**(-0.83)*(r9*0.1)**1.18*mass1**(-0.4) # g/cm^2
    Lsig2_1=39.9*(0.01/0.1)**(-0.8)*(r9*0.1)**1.11*mass1**(-0.37) # g/cm^2
    Lsig2_2=74.6*(0.01/0.1)**(-0.83)*(r9*0.1)**1.18*mass1**(-0.4) # g/cm^2
    Lteff1=6.890*(r9*0.1)**(-0.09)*mass1**0.03 #*alpha**(1./6.)
    Lteff2=5.210*(r9*0.1)**(-0.1)*mass1**0.04 #*alpha**(1./6.)

    clf()
    plot(sigline0, teff0, 'k', linewidth=2, label=r"$\alpha = 1$")
    plot(sigline1, teff1, 'b', linewidth=2, label=r"$\alpha = 0.1$")
    plot(sigline2, teff2, 'g', linewidth=2, label=r"$\alpha = 0.01$")
    plot([Lsig0_1, Lsig0_2], [Lteff1, Lteff2], 'ok')
    plot([Lsig1_1, Lsig1_2], [Lteff1, Lteff2], 'ob')
    plot([Lsig2_1, Lsig2_2], [Lteff1, Lteff2], 'og')
    xscale('log')
    ylabel(r'$T_{\rm eff}$, kK') ;   xlabel(r'$\Sigma$, ${\rm g\,cm^{-2}}$')
    savefig('sigcompare.png')
    close('all')
