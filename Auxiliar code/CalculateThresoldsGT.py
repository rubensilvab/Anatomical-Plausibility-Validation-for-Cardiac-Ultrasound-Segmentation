# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:57:17 2024

@author: RubenSilva
"""

import nibabel as nib
import os
import matplotlib.pyplot as plt 
from AnatomicalPlausibility import *

SofiaSegPath='C:/Users/RubenSilva/Desktop/CAMUS Dataset/SofiaSeg'
CAMUSPath='E:/RubenSilva/POCUS/Resources/database_nifti'
list_patients=os.listdir(CAMUSPath)

#img_sofia = nib.load(os.path.join(SofiaSegPath,'patient0089_2CH_ES.nii.gz'))
"""Load images DiAstole"""
path_save_info='C:/Users/RubenSilva/Desktop/THOR/findThresolds'

#list_patients=['patient0296']#,'patient0057','patient0181','patient0201']#,'patient0007','patient0017']#'patient0001',]#'patient0017',] 
for patient in list_patients:
    
    path_patient= os.path.join(CAMUSPath,patient)
    img_camus = nib.load(os.path.join(path_patient,str(patient)+'_2CH_ES.nii.gz'))
    seg_camus= nib.load(os.path.join(path_patient,str(patient)+'_2CH_ES_gt.nii.gz'))

    data_camus= img_camus.get_fdata()
    data_GT= seg_camus.get_fdata()
    #data_GT=data_GT[:,45:]
    data_GT = np.hstack((data_GT, np.zeros((data_GT.shape[0], 1))))
    data_GT = np.hstack(( np.zeros((data_GT.shape[0], 1)),data_GT))

    plt.imshow(data_camus, cmap='gray')
    plt.show()
    plt.imshow(data_GT, cmap='gray')
    plt.show()
    
    plaus=CheckPlausibility(data_GT,3,path_save_info,patient)
    print('Patient: ', patient,'Plausibility:', plaus )

import numpy as np 
import random
import cv2
import cc3d
from skimage import measure
