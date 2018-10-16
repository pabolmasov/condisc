from scipy.integrate import *
from scipy.interpolate import *
import matplotlib
from matplotlib import rc
from matplotlib import axes
from numpy import *
from pylab import *
from scipy.integrate import cumtrapz
from scipy.ndimage import zoom, gaussian_filter

import numpy.random
# import time
import os
import os.path

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
