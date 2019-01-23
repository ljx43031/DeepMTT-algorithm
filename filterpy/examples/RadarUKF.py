# -*- coding: utf-8 -*-
"""Copyright 2015 Roger R Labbe Jr.

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
#from GetRadar import GetRadar
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import JulierSigmaPoints as SP



def my_fx(x, dt):
    """ state transition function for sstate [downrange, vel, altitude]"""
    F = np.array([[1., dt, 0.],
                  [0., 1., 0.],
                  [0., 0., 1.]])

    return np.dot(F, x)


def my_hx(x):
    """ returns slant range based on downrange distance and altitude"""

    return (x[0]**2 + x[2]**2)**.5


if __name__ == "__main__":

    dt = 0.05

    my_SP = SP(3,kappa=0.)
    radarUKF = UKF(dim_x=3, dim_z=1, dt=dt, hx=my_hx, fx=my_fx, points=my_SP)
    radarUKF.Q *= Q_discrete_white_noise(3, 1, .01)
    radarUKF.R *= 10
    radarUKF.x = np.array([0., 90., 1100.])
    radarUKF.P *= 100.

    t = np.arange(0, 20+dt, dt)
    n = len(t)
    xs = []
    rs = []
    radarUKF.predict()
    radarUKF.update(10)
    
    #    for i in range(n):
#        r = GetRadar(dt)
#        rs.append(r)
#
#        radarUKF.update(r, my_hx, my_fx)
#
#        xs.append(radarUKF.x)
#
#    xs = np.asarray(xs)
#
#    plt.subplot(311)
#    plt.plot(t, xs[:, 0])
#    plt.title('distance')
#
#    plt.subplot(312)
#    plt.plot(t, xs[:, 1])
#    plt.title('velocity')
#
#    plt.subplot(313)
#    plt.plot(t, xs[:, 2])
#    plt.title('altitude')
