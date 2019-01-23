#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 11:07:17 2017

@author: ljx
"""

#==============================================================================
#Maneuvering Target Trajectory Data Generator
#For common Airport Surveillance Radar
#The detection coverage: 0.5~60 nautical miles (1.852km)
#For common civil aircraft (including helicopter)
#The velocity coverage: -340~340m/s
#The turn rate coverage: -10~10°  changing interval = 0.1°
#==============================================================================

import numpy as np
import random as rd
from numpy.linalg import cholesky
#Sampling interval
sT = 0.1

#The distance coverage of starting point:
#modify!!!!!
Dist_max = 20 * 1.852 * 1000
Dist_min = 0.5 * 1.852 * 1000
#The velocity coverage of starting point:
Velo_max = 340
Velo_min = -340
##The turn rate coverage:
TR_max = 10
TR_min = -10
TR_intv = 0.1

#Trajectory generator
def trajectory_creat(data_len, T_matrix):
    #Transition noise（Acceleration noise m/s^2）
    state_n = rd.uniform(8.0,13.0)
    s_var = np.square(state_n)
    T2 = np.power(sT,2)
    T3 = np.power(sT,3)
    T4 = np.power(sT,4)
    #var_m = np.array([[T4/4,0,T3/2,0],[0,T4/4,0,T3/2],[T3/2,0,T2,0],[0,T3/2,0,T2]]) * s_var
    var_m = np.array([[T4/4,0,0,0],[0,T4/4,0,0],[0,0,T2,0],[0,0,0,T2]]) * s_var
    chol_var = cholesky(var_m)
    
    
    data = np.array([[0 for i in range(4)] for j in range(data_len)],'float64')
    Dist_change = Velo_max * sT * data_len
    #Starting point
    sp_distance = rd.uniform(Dist_min + Dist_change, Dist_max - Dist_change)
    sp_direction = (rd.random()-0.5) * 2 * np.pi
    d_x = sp_distance * np.cos(sp_direction)   #Target X dirction position
    d_y = sp_distance * np.sin(sp_direction)   #Target Y dirction position
    
#    d_x = -500
#    d_y = 200
    
    sp_velocity = rd.uniform(Velo_min,Velo_max)
    vel_direction = (rd.random()-0.5) * 2 * np.pi
    
#    sp_velocity = 200
#    vel_direction = -np.pi/4
    
    v_x = sp_velocity * np.cos(vel_direction)  #Target X dirction velocity
    v_y = sp_velocity * np.sin(vel_direction)  #Target Y dirction velocity
    
    X_a = np.array([[d_x, d_y, v_x, v_y]],'float64')
    for i in range(data_len):
        data[i,:] = X_a
        X_a = np.dot(X_a, T_matrix)
    data_n = data + np.dot(np.random.randn(data_len, 4), chol_var)  #加噪声
    return data, data_n

def trajectory_batch_generator(batch_size, data_len):
    #Initialization
    Traj_r = np.array([[[0 for i in range(4)] for j in range(data_len)] for k in range(batch_size)],'float64')    #Initialization of trajectory
    Obser = np.array([[[0 for i in range(2)] for j in range(data_len)] for k in range(batch_size)],'float64') #Initialization of observation
    Tran_m = np.array([[[0 for i in range(4)] for j in range(4)] for k in range(batch_size)],'float64')
    for i in range(batch_size):
        
        turn_rate = rd.randint(TR_min/TR_intv,TR_max/TR_intv)*TR_intv
        if turn_rate == 0:
            F_c = np.array([[1,0,sT,0],[0,1,0,sT],[0,0,1,0],[0,0,0,1]])
        else:    
            w = turn_rate*np.pi/180  #turn rate
            F_c = np.array([[1,0,np.sin(w*sT)/w,(np.cos(w*sT)-1)/w],
                            [0,1,-(np.cos(w*sT)-1)/w,np.sin(w*sT)/w],
                            [0,0,np.cos(w*sT),-np.sin(w*sT)],
                            [0,0,np.sin(w*sT),np.cos(w*sT)]])
    
        F_c = np.transpose(F_c,[1,0])
        dt, dtn = trajectory_creat(data_len, F_c)
        Traj_r[i]= dt
        
        #Observations
        #observation noise
        dis_n = rd.uniform(8.0,13.0)  #Distance noise
        dis_var = np.square(dis_n)
        azi_n = rd.uniform(7.0,9.0)     #Azimuth noise
        azi_var = np.square(azi_n/1000)

        Obser[i,:,0] = np.arctan2(dtn[:,1],dtn[:,0]) + np.random.normal(0,azi_var,data_len) #Azimuth
        Obser[i,:,1] = np.sqrt(np.square(dtn[:,0])+np.square(dtn[:,1])) + np.random.normal(0,dis_var,data_len)   #Distance
        Tran_m[i,:,:] = F_c
    return Traj_r, Obser, Tran_m

def trajectory_batch_generator2(batch_size, data_len):
    #Initialization
    #observaiton noise in X Y direction
    Traj_r = np.array([[[0 for i in range(4)] for j in range(data_len)] for k in range(batch_size)],'float64')    #Initialization of trajectory
    Obser = np.array([[[0 for i in range(2)] for j in range(data_len)] for k in range(batch_size)],'float64') #Initialization of observation
    Tran_m = np.array([[[0 for i in range(4)] for j in range(4)] for k in range(batch_size)],'float64')
    for i in range(batch_size):
        
        turn_rate = rd.randint(TR_min/TR_intv,TR_max/TR_intv)*TR_intv
        if turn_rate == 0:
            F_c = np.array([[1,0,sT,0],[0,1,0,sT],[0,0,1,0],[0,0,0,1]])
        else:    
            w = turn_rate*np.pi/180  #turn rate
            F_c = np.array([[1,0,np.sin(w*sT)/w,(np.cos(w*sT)-1)/w],
                            [0,1,-(np.cos(w*sT)-1)/w,np.sin(w*sT)/w],
                            [0,0,np.cos(w*sT),-np.sin(w*sT)],
                            [0,0,np.sin(w*sT),np.cos(w*sT)]])
    
        F_c = np.transpose(F_c,[1,0])
        dt, dtn = trajectory_creat(data_len, F_c)
        Traj_r[i]= dt
        
        #Observations
        #observation noise in X Y direction
        X_n = rd.uniform(10.0,50.0)  #X direction noise
        X_var = np.square(X_n)
        dtn[:,0] = dtn[:,0] + np.random.normal(0,X_var,data_len)
        
        Y_n = rd.uniform(10.0,50.0)  #X direction noise
        Y_var = np.square(Y_n)
        dtn[:,1] = dtn[:,1] + np.random.normal(0,Y_var,data_len)
        

        Obser[i,:,0] = np.arctan2(dtn[:,1],dtn[:,0]) #Azimuth
        Obser[i,:,1] = np.sqrt(np.square(dtn[:,0])+np.square(dtn[:,1]))   #Distance
        Tran_m[i,:,:] = F_c
    return Traj_r, Obser, Tran_m

##==============================================================================
##保存成mat数据
#import scipy.io as scio
#mydata = {'my_traj':my_traj, 'my_obser':my_obser}
#scio.savemat('Traj_sample', mydata)
