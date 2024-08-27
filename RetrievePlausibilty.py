# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:28:16 2024

@author: RubenSilva
"""

import nibabel as nib
import os
import matplotlib.pyplot as plt 
from AnatomicalPlausibilityVfinal import *

"""Choose Path where the segmentations are"""

#Predictions
SofiaSegPath='C:/Users/RubenSilva/Desktop/CAMUS Dataset/SofiaSeg'
#Manual
CAMUSPath='E:/RubenSilva/POCUS/Resources/database_nifti'

"""
/DATA
-----/patientXXX
---------patientXXX_2CH_ES.nii.gz

"""

list_patients=os.listdir(SofiaSegPath)

""" Extract patient names (if there are not in the a specific patient folder)"""
# list_patients = {file.split('_')[0] for file in list_patients}
# list_patients=sorted(list_patients)

"""Choose path where do you want to save the results.
Choose the cardiac_ph: ES= Systole and ED= Diastole"""

cardiac_ph='ES'
path_save_info='C:/Users/RubenSilva/Desktop/THOR/ResultsAP/'+cardiac_ph

"""Do yow want to see the plots?"""
retrieve_all_plots=True

"""If you just have the predictions"""
for patient in list_patients:
    
    seg_sofia= nib.load(os.path.join(SofiaSegPath,str(patient)+'_2CH_'+cardiac_ph+'.nii.gz'))
    
    data_sofia= seg_sofia.get_fdata()
    data_sofia = np.hstack((data_sofia, np.zeros((data_sofia.shape[0], 1))))
    data_sofia = np.hstack(( np.zeros((data_sofia.shape[0], 1)),data_sofia))

    # Display data_sofia in the third subplot
    plt.imshow(data_sofia, cmap='gray')
    plt.title(f'Prediction ({patient})')
    plt.axis('off')  # Hide axis ticks
    
    # Show the plot
    plt.show()
    
    plaus=CheckPlausibility(data_sofia,3,cardiac_ph,path_save_info,patient,retrieve_all_plots)
    print('Patient: ', patient,', plausibility:', plaus )
    ExcelRowResult(path_save_info,plaus,patient)    


"""If you have GT to compare use this code"""

# for patient in list_patients:
    
#     path_patient= os.path.join(CAMUSPath,patient)
#     img_camus = nib.load(os.path.join(path_patient,str(patient)+'_2CH_'+cardiac_ph+'.nii.gz'))
#     seg_camus= nib.load(os.path.join(path_patient,str(patient)+'_2CH_'+cardiac_ph+'_gt.nii.gz'))
#     try:
#         seg_sofia= nib.load(os.path.join(SofiaSegPath,str(patient)+'_2CH_'+cardiac_ph+'.nii.gz'))
#     except:    
#         print(patient,'does not exist')
#         continue
    
#     data_camus= img_camus.get_fdata()
#     data_GT= seg_camus.get_fdata()
#     data_GT = np.hstack((data_GT, np.zeros((data_GT.shape[0], 1))))
    
#     data_sofia= seg_sofia.get_fdata()
#     data_sofia = np.hstack((data_sofia, np.zeros((data_sofia.shape[0], 1))))
#     data_sofia = np.hstack(( np.zeros((data_sofia.shape[0], 1)),data_sofia))

#     # Create a figure with 1 row and 3 columns
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
#     # Display data_camus in the first subplot
#     axes[0].imshow(data_camus, cmap='gray')
#     axes[0].set_title('Data Camus')
#     axes[0].axis('off')  # Hide axis ticks
    
#     # Display data_GT in the second subplot
#     axes[1].imshow(data_GT, cmap='gray')
#     axes[1].set_title('Data GT')
#     axes[1].axis('off')  # Hide axis ticks
    
#     # Display data_sofia in the third subplot
#     axes[2].imshow(data_sofia, cmap='gray')
#     axes[2].set_title(f'Data Sofia ({patient})')
#     axes[2].axis('off')  # Hide axis ticks
    
#     # Show the plot
#     plt.show()
    
#     plaus=CheckPlausibility(data_sofia,3,cardiac_ph,path_save_info,patient,retrieve_all_plots)
#     print('Patient: ', patient,', plausibility:', plaus )
#     ExcelRowResult(path_save_info,plaus,patient)
    

    
"""Criteria:
    1-(3 criteria) Presence of holes in the Left Ventricle (LV), Myocardium (MYO), and Left Atrium (LA).
    2-(2 criteria) Presence of holes between the LV and MYO, or between the LV and LA.
    3-(3 criteria) Presence of more than one LV, MYO, or LA. 
    4-(2 criteria) Size of the perimeter by which the LV touches the background or the MYO touches the LA exceeds a certain threshold.
    5-(1 criterion) Ratio between the minimal and maximal thickness of the MYO is below a given threshold
    6-(1 criterion) Ratio between the width of the LV and the average thickness of the MYO exceeds a certain threshold. Both width and thickness are computed as the total width of the structure at the middle-point of the embedded bounding box. The goal is to identify situations where the MYO is too thin with respect to the size of the LV.
    7-Curvature(3 criterion)
"""    
