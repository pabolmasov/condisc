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

    lines = loadtxt('sigtable.dat', comments="#", delimiter=" ", unpack=False)
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
