#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 21:26:40 2018

@author: ljx
"""

#==============================================================================
#Randomly select one batch and derive sample data to train the network
#==============================================================================

#from Trajectory_data_generator import*

import scipy.io as scio
import numpy as np
import random as rd
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import JulierSigmaPoints as SP

#using the data with small range of distances (60->20)
from Trajectory_data_generator2 import*
data_len = 50
#define UKF
dim_state = 4
dim_obser = 2
#transition noise
#Sampling time
sT = 0.1
state_n = 10.0
s_var = np.square(state_n)
T2 = np.power(sT,2)
T3 = np.power(sT,3)
T4 = np.power(sT,4)
var_m = np.array([[T4/4,0,T3/2,0],[0,T4/4,0,T3/2],[T3/2,0,T2,0],[0,T3/2,0,T2]]) * s_var
#var_m = np.array([[T4/4,0,0,0],[0,T4/4,0,0],[0,0,T2,0],[0,0,0,T2]]) * s_var
#observation noise
dis_n = 10.0   #distance
dis_var = np.square(dis_n)
azi_n = 8.0     #azimuth
azi_var = np.square(azi_n/1000)

#Modefy observations, make them continuous
def data_refine(Obser):
    bs,dl,_ = np.shape(Obser)
    new_obser = np.copy(Obser)
    for j in range(bs):
        for i in range(dl-1):
            a = new_obser[j,i,0]
            b = new_obser[j,i+1,0]
            c = a-b
            if c > 6:
                new_obser[j,i+1:,0] = new_obser[j,i+1:,0] + 2*np.pi
            if c < -6:
                new_obser[j,i+1:,0] = new_obser[j,i+1:,0] - 2*np.pi
            
    return new_obser

#Creat batch for training

#State transition function
def my_fx(x, sT):
    """ state transition function for sstate [downrange, vel, altitude]"""
    F = np.array([[1,0,sT,0],[0,1,0,sT],[0,0,1,0],[0,0,0,1]],'float64') #F_cv

    return np.dot(F, x)
#Observation function
def my_hx(x):
    """ returns slant range = np.array([[0],[0]]) based on downrange distance and altitude"""
    r = np.array([0,0],'float64')
    r[0] = np.arctan2(x[1],x[0])
    r[1] = np.sqrt(np.square(x[0])+np.square(x[1]))
    return r

#Batch creating
def creat_batch3(min_dn,max_dn,pos_noise,vel_noise,BN):
    batch_size = int(100*BN)
    my_traj, my_obser, Tran_m = trajectory_batch_generator(batch_size,data_len)
    Traj_r = my_traj
    Obser = my_obser
    
    Traj_s = np.array([[[0 for i in range(4)] for j in range(data_len)] for k in range(batch_size)],'float64')
    for i in range(batch_size): 
        my_SP = SP(dim_state,kappa=0.)
        my_UKF = UKF(dim_x=dim_state, dim_z=dim_obser, dt=sT, hx=my_hx, fx=my_fx, points=my_SP)
        my_UKF.Q *= var_m
        my_UKF.R *= np.array([[azi_var,0],[0,dis_var]])
        x0_noise = np.array([np.random.normal(0,pos_noise,2),np.random.normal(0,vel_noise,2)])
        my_UKF.x = Traj_r[i,0,:] + np.reshape(x0_noise,[4,])
        my_UKF.P *= 1.
        
        #tracking results of UKF
        xs = []
        xs.append(my_UKF.x)
        for j in range(data_len-1):
            my_UKF.predict()
            my_UKF.update(Obser[i,j+1,:])
            xs.append(my_UKF.x.copy())    
        Traj_s[i] = np.asarray(xs)
        
    return Traj_r, Obser, Traj_s, Traj_r-Traj_s