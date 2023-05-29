#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 10:42:21 2016

@author: flavio.pignone@cimafoundation.org
@author: mirko.dandrea@cimafoundation.org
"""

import numpy as np
import scipy.fftpack as fft
import scipy.io as io


power = 3.0
T = 5
dt = 1
sigma = 0.02
tendency_vel = 0



def __get_k2l2(shape):
    '''
    get the symmetric weight matrix for the phase modulation
    :param shape: shape of the matrix
    :return: the weight matrix
    '''
    n_rows = shape[0]
    n_cols = shape[1]
    k2l2_pow = np.zeros((n_rows, n_cols))
    
    for r in range(n_rows):
        # calcolo lunghezza fourier k
        length_f_k = r
        if length_f_k >= n_rows/2:
            length_f_k = length_f_k - n_rows+1
    
        # per ogni pixel y
        for c in range(n_cols):
            # calcolo lunghezza fourier l
            length_f_l = c
            if length_f_l >= n_cols/2:
                length_f_l = length_f_l - n_cols+1
    
            # velocit‡ angolari
            k2l2_pow[r, c] = (length_f_k**2+length_f_l**2)**(1/power)

    return k2l2_pow

    
    

def __realeB_Rect(phi):
    '''
    a = realeB(g)
     
    Restituisce la matrice simmetrica delle fasi per una reale 
    antitrasformata di Fourier utilizzando la prima metà della 
    matrice (vettore) in ingresso.
    
    'size(g)=[N,1]' :  non verranno toccate le posizioni g(1) e g(Nmp1)
    'size(g)=[N,N]' :  non verranno toccate le posizioni g(1,1), g(1,Nmp1), g(Nmp1,1) e g(Nmp1,Nmp1)
    '''
    
    ret_phi = phi.copy()
    
    N = phi.shape[0];
    Nm = int(N/2);

    Nmp1 = Nm;
    
    if len(phi.shape) == 1:
        for i in range(1, Nm):
            m = N - i + 1
            ret_phi[m] = -ret_phi[i]

    else:
        M = phi.shape[1]
        for i in range(1, Nm):                  #da 1 a Nm-1
            for j in range(1, M):               #da 1 a M-1
                m = N - i                       #
                n = M - j
                ret_phi[m, n] = -ret_phi[i, j]    #a
        for j in range(1, Nm):
            n = N - j
            ret_phi[0, n] = -ret_phi[0, j]          
            ret_phi[Nmp1, n] = -ret_phi[Nmp1, j]    #b

        for i in range(1, Nm):
            m = N - i
            ret_phi[m, 0] = -ret_phi[i,0]   #c
 
    
    return ret_phi
    

def phast(r_now, r_pre, n_times=12, n_ensamble=10, enable_norm=True):
    '''
    generates a ensamble of nowcasted rainfall fields based on r_now and r_pre
    :param r_now: rainfall field at timestep t
    :param r_pre: rainfall field at timestep t-1    
    :param n_times: number of forecasted timesteps
    :param n_ensamble: number of ensambles
    :param enable_norm: enable normalization of the field
    :return: 4d matrix [rows, columns, n_times, n_ensambles]
    '''

    
    n_rows, n_cols = r_now.shape[0], r_now.shape[1]
    k2l2_pow = __get_k2l2(r_now.shape)
    langevin = np.sqrt((2*dt)/T) * np.sqrt(1-dt/(2*T));
    
    
    r_now_sorted = np.sort(r_now.flatten())
    
    f_r_pre = fft.fft2(r_pre)
    f_r_now = fft.fft2(r_now)
    
    
    
    angle_r_pre = np.angle(f_r_pre)
    initial_phase = np.angle(f_r_now)
    initial_amplitude = np.abs(f_r_now)
    
    #initial angular velocity
    initial_angular = initial_phase - angle_r_pre  
    
    
    #phasting
    phast_output = np.zeros((n_rows, n_cols, n_times, n_ensamble))

    for e in range(n_ensamble):
        r_vect = np.random.randn(n_times)    
    
        nowcasted_phase = initial_phase.copy()
        for t, r in enumerate(r_vect):
            print(e, t)
            eps = np.zeros(t+1)
            for i in range(t+1):
                eps[i] = langevin * sigma * r_vect[i] * (1-(dt/T))**(t-i);
    
            epsilon = k2l2_pow * sum(eps)
            
            #nowcast angular speed
            nowcasted_angular = \
                initial_angular * (1-(dt/T))**t + \
                tendency_vel * (1-(1-(dt/T))**t)+ \
                epsilon
    
            #
            nowcasted_phase = nowcasted_phase + nowcasted_angular * dt;
            if enable_norm:
                nowcasted_values = np.real( \
                   fft.ifft2(initial_amplitude *\
                                 np.exp(1j * __realeB_Rect(nowcasted_phase))))
            else:
                nowcasted_values = np.real( \
                   fft.ifft2(initial_amplitude *\
                                 np.exp(1j * nowcasted_phase)))                
               
    
    
            sort_idx_nowcast = np.argsort(nowcasted_values.flatten())
            v = np.zeros(sort_idx_nowcast.shape)
            v[sort_idx_nowcast] = r_now_sorted
    
            phast_output[:,:,t,e] = np.reshape(v, (n_rows, n_cols))
    

    return phast_output

    
    
if __name__ == '__main__':    
    mat_file_rain = io.loadmat('test.mat')
    R = mat_file_rain['a3dRadarRainT']    
    phast(R[:,:,-1], R[:,:,-2])