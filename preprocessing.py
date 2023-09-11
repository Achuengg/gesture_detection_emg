#/bin/python
"""
Created on Thu May 12 23:02:50 2022

@author: Akshaya
"""

import os
import numpy as np
from scipy import io
from scipy.signal import butter,lfilter




def preprocessing(raw_file):     
   
    lowcut,highcut = [int(x) for x in input("Enter low cutoff & high cutoff frequency: ").split()]
    fs = int(input("Enter Sampling frequency: "))
    order =  int(input("Enter Digital filter order: "))
    
    for i in raw_file:
    
        raw_data = io.loadmat(raw_file)
        prep_data=[]
    ###### Implementing butterworth filter to remove baseline interfernce #########  
      
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
    
        b, a = butter(order, [low, high], btype='bandstop')
        raw_filter = lfilter(b,a,raw_data['data'],axis=0)
        
    ################# Dividing raw data into 10 trial files #######################   
     
        start = 0
        end = 0
        counter = 0
        
        for i in range(0,(np.size(raw_data['gesture'],1))-1):
            if raw_data['gesture'][0,i+1]- raw_data['gesture'][0,i] > 0:
                start = i+1          
            elif raw_data['gesture'][0,i+1]- raw_data['gesture'][0,i] < 0:
                end = i+1 
            if start>0 and end >0:
                counter += 1
                temp = raw_filter[start:end,:]
                length = np.size(temp,0)
                m = int(length/2)
                tr = {'data':temp[m-500:m+500,:],'gesture':raw_data['gesture'][:,start],'subject':raw_data['subject'],'trial': counter } 
                prep_data.append(tr)
                start = 0
                end = 0
    return prep_data 






# def butter_bandstop_filter(data, lowcut, highcut, fs, order):


#         nyq = 0.5 * fs
#         low = lowcut / nyq
#         high = highcut / nyq

#         b, a = butter(order, [low, high], btype='bandstop')
#         y = lfilter(b,a,data,axis=0)
#         return y

# ################# Dividing raw data into 10 trial files #######################
# def split_trl(raw_data,raw_filter):
#     start = 0
#     end = 0
#     counter = 0
#     prep_data=[]
#     for i in range(0,(np.size(raw_data['gesture'],1))-1):
#         if raw_data['gesture'][0,i+1]- raw_data['gesture'][0,i] > 0:
#             start = i+1          
#         elif raw_data['gesture'][0,i+1]- raw_data['gesture'][0,i] < 0:
#             end = i+1 
#         if start>0 and end >0:
#             counter += 1
#             temp = raw_filter[start:end,:]
#             length = np.size(temp,0)
#             m = int(length/2)
#             tr = {'data':temp[m-500:m+500,:],'gesture':raw_data['gesture'][:,start],'subject':raw_data['subject'],'trial': counter } 
#             prep_data.append(tr)
#             start = 0
#             end = 0
#     return prep_data  


    




