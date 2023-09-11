#/bin/python
"""
Created on Sat May 14 22:52:01 2022

@author: Akshaya
"""
import os
from main import *

print("Best Model for Gesture Recognition: Quadratic Discriminant Analysis")
print("Best Model for Person Recognition:Linear Discriminant Analysis")
print("Trials for testing : 1 2")
filename_gest = "./model_ref/finalized_model_gesture.sav"
filename_subj = "./model_ref/finalized_model_subject.sav"
minmax_fit = pickle.load(open('./model_ref/minmax.sav','rb'))
norm_fit = pickle.load(open('./model_ref/norm.sav','rb'))
pca_fit = pickle.load(open('./model_ref/pca_fit.sav','rb'))

# Load raw data
folder_path = './Input_data/raw/dba-raw-' + input("Enter the file name (001,002..,018):") + '/'
raw_file = [folder_path+i for i in os.listdir(folder_path)]
pre_process = preprocessing_data(raw_file,trial=[1,2])

wl=200
win=50
out = feature_extraction(pre_process,wl,win)

data_log = log(out["feat_mav"])     
data_minmax = minmax_fit.transform(out["feat_zc"]) 

###############################################################################
# Combining all features
###############################################################################
final = np.concatenate((data_log,data_minmax),axis=1)

# Normalising combined features before training (required for SVM)

data_norm = norm_fit.transform(final) 
start , end = [int(x) for x in input("Enter start and end data points for testing (12-20): ").split('-')]
test_data = data_norm[start:end,:]
test_actual_gesture = out['gesture'][start:end,:]
test_actual_subject = out['subject'][start:end,:]

# load gesture identification model from disk

loaded_gest_model = pickle.load(open(filename_gest, 'rb'))
y_pred_gesture = loaded_gest_model.predict(test_data)

print('Accuracy for Gesture: ')
print(accuracy_score(test_actual_gesture,y_pred_gesture))
disp = plot_confusion_matrix(loaded_gest_model,test_data,test_actual_gesture,cmap = 'OrRd') 

# load subject  identification model from disk
# PCA 
test_data = pca_fit.transform(test_data)
loaded_subj_model = pickle.load(open(filename_subj, 'rb'))
y_pred_subject = loaded_subj_model.predict(test_data)

print('Accuracy for Subject: ')
print(accuracy_score(test_actual_subject,y_pred_subject))
disp = plot_confusion_matrix(loaded_subj_model,test_data,test_actual_subject,cmap = 'Blues')  

print("Result:")
gesture_dict = {1:'Thumb up', \
                2:'Extension of index and middle, flexion of the others',\
                3:'Flexion of ring and little finger, extension of the others',\
                4:'Thumb opposing base of little finger',\
                5:'Abduction of all fingers',\
                6:'Fingers flexed together in fist',\
                7:'Pointing index',\
                8:'Adduction of extended fingers'}
for i in range(len(y_pred_gesture)):
    print("Person - {} did the gesture: {}".format(y_pred_subject[i],gesture_dict[y_pred_gesture[i]])) 
  

  