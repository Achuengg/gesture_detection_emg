#/bin/python
"""
Created on Tue May 10 14:10:23 2022

@author: Akshaya
"""

# Importing required packages & Modules 
import os
import sys
from scipy import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from scipy.signal import butter,lfilter
from sklearn.preprocessing import StandardScaler as std, MinMaxScaler as mm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as qda
from sklearn import svm
from sklearn.model_selection import KFold , cross_val_score
from sklearn.metrics import accuracy_score,plot_confusion_matrix,confusion_matrix,ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
###############################################################################
# Functions to load files
###############################################################################

def loadfiles(folder_path):
        
    if folder_path == "./Input_Data/preprocessed/":    
        list_folder = [folder_path+i for i in os.listdir(folder_path)]
        list_dir = []
        for name in list_folder:
            list_dir.extend([name+'/'+i for i in os.listdir(name) if '.mat' in i ])
        base_data = []
        for i in list_dir:
                mydata = io.loadmat(i)
                base_data.append(mydata)
        return base_data
    else:
       folder_path == "./Input_Data/raw/"
       
       list_folder = [folder_path+i for i in os.listdir(folder_path)]
       raw_file = []
       for name in list_folder:
           raw_file.extend([name+'/'+i for i in os.listdir(name) if '.mat' in i ])
       return raw_file 

###############################################################################
# Functions for raw data preprocessing 
###############################################################################
    
def preprocessing_data(raw_file,trial=[]):     
   
    lowcut = 45
    highcut = 55
    fs = 1000
    order =  2
    prep_data=[]
    
    for i in raw_file:
    
        raw_data = io.loadmat(i)
        
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
                if length<1000:
                    start = 0
                    end = 0
                    continue
                m = int(length/2)
                tr = {'data':temp[m-500:m+500,:],'gesture':[[raw_data['gesture'][0,start]]],'subject':raw_data['subject'],'trial': [[counter]] }
                if not len(trial):
                    prep_data.append(tr)
                else:
                    if counter in trial:
                        prep_data.append(tr)
                start = 0
                end = 0
    return prep_data    

###############################################################################
# Functions for feature extraction  
###############################################################################
def feat_mav(data,wl,win):
    
    samp_len = len(data[:,1])
    features = []
    st=0
    end=wl    
    while end<= samp_len:
        temp = np.mean(np.abs(data[st:end,:]),axis=0)
        features.append(temp)
        st = st + win
        end = end + win
    return features 

def feat_zc(data,wl,win):
    
    sign_ch = np.zeros((np.size(data,0)-1,np.size(data,1)))

    for j in range (np.size(data,1)):
        x = data[:,j]
        
        for i in range(len(x)-1):
            if (x[i]>=0 and x[i+1]>=0) or (x[i]<=0 and x[i+1]<=0):
                pass
            else:
                sign_ch[i,j]=1
       
    samp_len = len(sign_ch[:,1])
    features = []
    st=0
    end=wl-1
    
    while end<= samp_len:
        temp = np.sum(sign_ch[st:end,:],0)
        features.append(temp)
        st = st + win
        end = end + win
    return features

###############################################################################
# Feature Extraction for all dataset
###############################################################################   
def feature_extraction(base_data,wl,win):
    out={}
    
    for i in range(0,len(base_data)):
        cur_data = base_data[i]['data']
        try:
            temp0 = np.array(feat_mav(cur_data,wl,win))
            temp1 = np.array(feat_zc(cur_data,wl,win))
            out["feat_mav"] = np.append(out["feat_mav"],temp0,0)        
            out["feat_zc"] = np.append(out["feat_zc"],temp1,0)
            out["gesture"] = np.append(out["gesture"],base_data[i]['gesture']*np.ones((len(temp1),1)),0) 
            out["subject"] = np.append(out["subject"],base_data[i]['subject']*np.ones((len(temp1),1)),0)
            out["trial"] = np.append(out["trial"],base_data[i]['trial']*np.ones((len(temp1),1)),0)
                   
        except KeyError as e:
                
            out["feat_mav"] = temp0
            out["feat_zc"] = temp1
            out["gesture"] = base_data[i]['gesture']*np.ones((len(out['feat_zc']),1))  
            out["subject"] = base_data[i]['subject']*np.ones((len(out['feat_zc']),1))
            out["trial"] = base_data[i]['trial']*np.ones((len(out['feat_zc']),1))       
        
    
    return out


###############################################################################
# Normalizing Functions
###############################################################################

def log(data):       
    data_log = np.log(data)
    return data_log

def norm_minmax(data):
    
    m = mm(feature_range=(-1,1))
    m.fit(data)
    pickle.dump(m, open('./model_ref/minmax.sav', 'wb'))
    data_minmax = m.transform(data)
    return data_minmax

def norm(data):    
    s=std()
    s.fit(data)
    pickle.dump(s, open('./model_ref/norm.sav', 'wb'))
    data_norm = s.transform(data)
    return data_norm 

###############################################################################
#Classifiers selection
###############################################################################
def classifiers(clf,data,label):
    model = clf  
    #Define method to evaluate model   
    cv = KFold(n_splits=5,shuffle=False)
    #evaluate model
    scores = cross_val_score(model,data, label.ravel(), scoring='accuracy', cv=cv, n_jobs=3)
    m_score = np.mean(scores)
    return m_score
###############################################################################
#Training the model
############################################################################### 
def training(op_class,PCA_flag=0):   
    result = {}
    for i in range(1,10):
        for j in range(i+1,11):        
            train_df = in_df.loc[(in_df['trial']!= i) & (in_df['trial']!= j)]           
            data = np.array(train_df.loc[:,0:255])         
            label= train_df.loc[:,op_class]
            # feature Normalization 
            data[:,0:128] = log(data[:,0:128])
            data[:,128:] = norm_minmax(data[:,128:])
            data = norm(data)
            # PCA
            if PCA_flag:
                pca = PCA(n_components = 32)
                data = pca.fit_transform(data)

            for clf in class_list:
                  val = classifiers(class_list[clf],data,label)
                  try:
                      result["test_trials"].append([i,j])
                      result["classifier"].append(clf)
                      result['train_accuracy'].append(val)
                  except KeyError as e:
                      result["test_trials"] = [[i,j]]
                      result["classifier"] = [clf]
                      result['train_accuracy'] = [val]
    return result
###############################################################################
# Testing the model
############################################################################### 
def testing(train_data,train_label,test_data,test_label,model):    
    model.fit(train_data,train_label.ravel())
    ypred = model.predict(test_data)
    
    print(accuracy_score(test_label,ypred))
    disp = plot_confusion_matrix(model,test_data,test_label,cmap = 'OrRd')
    
    return disp ,model
###############################################################################
# Main code starts here
###############################################################################
if __name__ == '__main__':
    opt = input("Use Preprocessed data or Raw data ?")
    wl=int(input('Enter Window length:'))
    win=int(input('Enter Window increment:'))
    out_path =  './out_matlab/'
    
    if opt == 'Preprocessed data':
        folder_path = "./Input_Data/preprocessed/"
        base_data = loadfiles(folder_path)
        print("Input data loading completed")
        out = feature_extraction(base_data,wl,win)     
        print(f"Feature Extraction of {opt} completed & saved")
        
    elif opt == 'Raw data':
        folder_path = "./Input_Data/raw/"
        raw_file = loadfiles(folder_path)
        prep_data = preprocessing_data(raw_file) 
        print("Input data loading completed")
        out = feature_extraction(prep_data,wl,win)
        print(f"Feature Extraction of {opt} completed & saved")
              
    ###########################################################################
    # Combining all features
    ###########################################################################
    final = np.concatenate((out['feat_mav'],out["feat_zc"]),axis=1)
    final={"feat_comb":final,"gesture":out["gesture"],"subject":out["subject"],"trial":out["trial"]}
    print("All features are combined successfully!!!")
    io.savemat('./Output_Data/'+"feat_all"+'.mat',final)
    print("Combined features stored successfully!!!")
    
    ###########################################################################
    # Model slection using 3 classifiers for all data set
    ########################################################################### 
    input_data = io.loadmat("./Output_Data/feat_all.mat")
    indata = pd.DataFrame(np.hstack((input_data['feat_comb'],input_data['gesture'],input_data['subject'],input_data['trial'])))
    temp_in_df = indata.rename(columns={256: 'gesture',257:"subject",258:"trial"})
    in_df = temp_in_df.sort_values('trial')
    class_list = {"lda":lda(),"qda":qda(),"svm":svm.SVC(C = 5,kernel='poly', decision_function_shape='ovo')}
    
    for i in ['gesture',"subject"]:    
        op_class = i
        if i == "gesture":
            result = training(op_class,PCA_flag=0)
        else:    
            result = training(op_class,PCA_flag=1) 
            
        max_accuracy = max(result["train_accuracy"])
        max_index = result["train_accuracy"].index(max_accuracy)
        final_model = result["classifier"][max_index]
        trials_not_used = result["test_trials"][max_index]
        
        io.savemat("./Output_Data/"+"result_"+op_class+'.mat',result)
        print(f'The best model for {op_class} recognition from all(Train-Test) trials is {final_model} with train accuracy of {max_accuracy}')
        print(f'The trials not used for above training are {trials_not_used}')
        
        # Train the selected model    
        tr1 =  trials_not_used[0]
        tr2 =  trials_not_used[1]        
        test = in_df.loc[in_df['trial'].isin([tr1,tr2])]
        train = in_df.loc[(in_df['trial']!= tr1) & (in_df['trial']!= tr2)]       
        train_data  = np.array(train.loc[:,:255])
        train_label = train.loc[:,op_class]
        test_data  = np.array(test.loc[:,:255])
        test_label = test.loc[:,op_class]
        
        # feature Normalization        
        train_data[:,0:128] = log(train_data[:,0:128])              
        train_data[:,128:] = norm_minmax(train_data[:,128:])     
        train_data = norm(train_data)      
        minmax_fit = pickle.load(open('./model_ref/minmax.sav', 'rb'))
        norm_fit = pickle.load(open('./model_ref/norm.sav', 'rb'))
        test_data[:,0:128] = log(test_data[:,0:128])
        test_data[:,128:] = minmax_fit.transform(test_data[:,128:])
        test_data = norm_fit.transform(test_data)
        
        # PCA for subject 
        if i == "subject":            
            pca = PCA(n_components = 32)
            train_data = pca.fit_transform(train_data)
            pickle.dump(pca, open('./model_ref/pca_fit.sav', 'wb')) 
            test_data = pca.transform(test_data)
        
        model = class_list[final_model]    
        # Testing the model
        print('Test accuracy for '+op_class+':' )
        cf_matrix, fitted_model = testing(train_data,train_label,test_data,test_label,model)
    
        # Save the model to disk
        filename = './model_ref/finalized_model_'+op_class+'.sav'
        pickle.dump(model, open(filename, 'wb'))    
        cf_matrix.figure_.savefig('./Output_Data/Confusion_Matrix_'+op_class+'.jpg')
       
    
    


 


