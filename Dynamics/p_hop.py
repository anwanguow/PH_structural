#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def p_hop(traj, t, t_R_2):
    
    N_mols = len(traj[0])

    p_hop_i = np.zeros((N_mols), dtype = "float32")

    A = np.zeros((t_R_2+1), dtype = "int")
    B = np.zeros((t_R_2+1), dtype = "int")

    for i in range(t_R_2+1):
        A[i] = t-t_R_2+i
        B[i] = t+i

    r_avg_A = 0
    for i in range(t_R_2+1):
        r_avg_A = r_avg_A + traj[A[i]]
    r_avg_A = r_avg_A/(t_R_2+1)

    r_avg_B = 0
    for i in range(t_R_2+1):
        r_avg_B = r_avg_B + traj[B[i]]
    r_avg_B = r_avg_B/(t_R_2+1)

    r_sqr_A = 0
    for i in range(t_R_2+1):
        m_vec = traj[A[i]] - r_avg_B
        sum_sqr = np.power(np.linalg.norm(m_vec, axis = 1),2)
        r_sqr_A = r_sqr_A + sum_sqr
    r_sqr_A = r_sqr_A/(t_R_2+1)

    r_sqr_B = 0
    for i in range(t_R_2+1):
        m_vec = traj[B[i]] - r_avg_A
        sum_sqr = np.power(np.linalg.norm(m_vec, axis = 1),2)
        r_sqr_B = r_sqr_B + sum_sqr
    r_sqr_B = r_sqr_B/(t_R_2+1)
        
    p_hop_i = np.sqrt(r_sqr_A * r_sqr_B)

    return p_hop_i

def p_hop_i(traj, i, t, t_R_2):
    vec = p_hop(traj, t, t_R_2)
    return vec[i]


