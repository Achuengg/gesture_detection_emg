#/bin/python
"""
Created on Wed May  4 01:46:12 2022

@author: Akshaya
"""
import os
import zipfile
def un_zip(file_name):
    """decompression zip"""
    zip_file = zipfile.ZipFile(file_name)
    file_name = file_name[:-4]
    if os.path.isdir(file_name):
        return
    else:
        os.mkdir(file_name)
    for names in zip_file.namelist():
        zip_file.extract(names,file_name)
    zip_file.close()
 # enter the folder path in which zip files are located  
folder_path = "C:/Users/vjoth/Documents/MyData/Akshaya/MS_E_CS/NMI/PROJECT/Input_Data/"
list_dir = [folder_path+'/'+i for i in os.listdir(folder_path) if '.zip' in i ]  
  
# calling unzip function in a loop to extract multiple zip files kept in above folder

for i in list_dir:
    un_zip(i)
 