# -*- coding: utf-8 -*-
"""
Created on Tue May  1 14:07:07 2018

@author: l_cry
"""

import copy
import numpy as np
from math import log
def v_value(t,w,v,param,g,h,unit):
    vals = []
    index = 0
    exceed_vals = []
    temp = {}
    temp['t']=t[0]
    temp['w'] = w[0]
    temp['v'] = v[0]
    temp['w_pre_value'] = 0
    temp['v_pre_value'] = 0
    temp['pre_value'] = temp['w_pre_value']+temp['v_pre_value']
    if w[0]>param['u']:
        temp['w_value'] = g(w[0],0)
        exceed_temp=temp
        exceed_vals.append(exceed_temp)
    else:
        temp['w_value'] = 0
    
    temp['v_value']=0
    temp['value'] = temp['w_value'] + temp['v_value']
    temp['w_back_value'] = temp['w_pre_value'] + temp['w_value']
    temp['v_back_value'] = temp['v_pre_value'] + temp['v_value']
    temp['back_value'] = temp['pre_value'] + temp['value']
    
    vals.append(temp)
    
    
    for i in np.arange(0,len(t)-1,1):
        temp={}
        temp['t'] = t[i+1]
        temp['w'] = w[i+1]
        temp['v'] = v[i+1]
        temp['w_pre_value'] = vals[i]['w_back_value']*np.exp(-param['gama']*(t[i+1]-t[i])/unit)
        temp['v_pre_value'] = vals[i]['v_back_value']*np.exp(-param['v_gama']*(t[i+1]-t[i])/unit)
        temp['pre_value'] = temp['w_pre_value']+temp['v_pre_value']
        
        if w[i+1]>param['u']:
            temp['w_value'] = g(w[i+1],temp['pre_value'])
            exceed_temp=temp
            exceed_vals.append(exceed_temp)
        else:
            temp['w_value'] = 0
            
        
            #temp_sum_v = sum(v[(i+1-param['ge']):(i+1)]) and np.array(v[(i+1-param['ge']):(i+1)])>
        if i+1-index>=param['ge']:
            temp_sum_v = sum(v[(i+1-param['ge']):(i+1)])
        else:
            temp_sum_v = sum(v[(index):(i+1)])
        if temp_sum_v > param['vu']: 
            temp['v_value'] = h(temp_sum_v,temp['pre_value'])
            index = copy.deepcopy(i+1)
        else:
            temp['v_value'] = 0
        
            
        temp['value'] = temp['w_value'] + temp['v_value']
        temp['w_back_value'] = temp['w_pre_value'] + temp['w_value']
        temp['v_back_value'] = temp['v_pre_value'] + temp['v_value']
        temp['back_value'] = temp['pre_value'] + temp['value']
        vals.append(temp)
        
    return vals,exceed_vals

def max_likelihood(t,w,v,param):
    one_day = np.timedelta64(1,'D')
    #g = lambda x,z:1+param['delta']*(x-param['u'])/(param['beta']+param['alpha']*z)
    #g = lambda x,z:1+param['delta']/param['kesi']*log(1+param['kesi']*(x-param['u'])/(param['beta']+param['alpha']*z))
    g = lambda x,z:param['delta']*(x-param['u'])/(param['beta']+param['alpha']*z)+1
    h = lambda x,z:param['v_delta']*(1+10*(x-param['vu'])/(param['v_beta']+param['v_alpha']*z))
    #h = lambda x,z:1+param['v_delta']*(x-param['vu'])
    vals,exceed_vals = v_value(t,w,v,param,g,h,one_day)
    t_max = t[-1]
    integral_sum = param['fi']*sum(map(lambda x: x['w_value']/param['gama']*(1-np.exp(-(t_max-x['t'])/one_day*param['gama']))+x['v_value']/param['v_gama']*(1-np.exp(-(t_max-x['t'])/one_day*param['v_gama'])) , vals))
    const_sum = param['tao']*((t_max-t[0])/one_day)
    #acc_sum = sum(map(lambda x:log((param['tao']+param['fi']*x['pre_value'])/(param['beta']+param['alpha']*x['pre_value']))-(x['w']-param['u'])/(param['beta']+param['alpha']*x['pre_value']),exceed_vals))
    acc_sum = sum(map(lambda x:log((param['tao']+param['fi']*x['pre_value'])/(param['beta']+param['alpha']*x['pre_value']))+(-1/param['kesi']-1)*log(1+param['kesi']*(x['w']-param['u'])/(param['beta']+param['alpha']*x['pre_value'])),exceed_vals))

    return -integral_sum-const_sum+acc_sum
    