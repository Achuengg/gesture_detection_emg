#/bin/python
"""
Created on Wed May  4 15:01:29 2022

@author: Akshaya
"""

import os
import shutil


base_path = "C:/Users/vjoth/Documents/MyData/Akshaya/MS_E_CS/NMI/PROJECT/Input_Data/raw/"

final_path = "C:/Users/vjoth/Documents/MyData/Akshaya/MS_E_CS/NMI/PROJECT/Input_Data/gest_removed/"


original_path = [base_path+i for i in os.listdir(base_path) if not '.zip' in i ]
for ls in original_path:
    list_files = [ls+'/'+i for i in os.listdir(ls)]
    for i in list_files:
        if i[-7:-4]== '100' or i[-7:-4]== '101' :
            shutil.move(i,final_path)
        else:
            continue
    
