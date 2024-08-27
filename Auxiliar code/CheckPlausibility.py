# -*- coding: utf-8 -*-
"""
Created on Mon May 20 12:19:31 2024

@author: RubenSilva
"""

import nibabel as nib
import os
import matplotlib.pyplot as plt 

"""Descomentar para testar"""
# SofiaSegPath='C:/Users/RubenSilva/Desktop/CAMUS Dataset/SofiaSeg'
# CAMUSPath='C:/Users/RubenSilva/Desktop/CAMUS Dataset/CAMUS/patient0089'

# img_sofia = nib.load(os.path.join(SofiaSegPath,'patient0089_2CH_ES.nii.gz'))
# img_camus = nib.load(os.path.join(CAMUSPath,'patient0089_2CH_ES.nii.gz'))
# seg_camus= nib.load(os.path.join(CAMUSPath,'patient0089_2CH_ES_gt.nii.gz'))

# data_sofia= img_sofia.get_fdata()
# data_camus= img_camus.get_fdata()
# data_GT= seg_camus.get_fdata()


# plt.imshow(data_sofia, cmap='gray')
# plt.show()
# plt.imshow(data_camus, cmap='gray')
# plt.show()
# plt.imshow(data_GT, cmap='gray')
# plt.show()

#import cc3d
import numpy as np 
import random
import cv2
import cc3d
from skimage import measure

Name_classes={0: 'Background', 1: 'Endocardium', 2:'Myocardium', 3: 'left atrium'}

def SeeSpecificStructure(structure, data):
    # Create a boolean mask where the elements are equal to the structure
    mask = (data == structure)
    
    if structure ==0:
        mask=mask.astype(int)
        return np.logical_not(mask).astype(int)
    else:
        # Use the mask to zero out elements that are not part of the structure
        result = np.where(mask, data, 0)
        return result

def BinaryzeAllMask(data):
    result=np.copy(data)
    mask = (data >=1)
    result[np.where(mask)] = 1
    
    return result

#Create Segmentations with holes

def extract_random_integers(start, end, count):
    # Generate a list of random integers between start and end (inclusive)
    random_integers = random.sample(range(start, end + 1),count)
    
    return random_integers


def create_holes(data,n_holes):
    data1=np.copy(data)
    indx=np.where(data1>=1)
    #print(indx,len(indx[0]))
    randint=extract_random_integers(0, len(indx[0]), n_holes)
    center_y,center_x=indx[0][randint],indx[1][randint]

    height,width=data.shape
    
    for i in range(n_holes):
    
        Y, X = np.ogrid[:height, :width]
        dist= np.sqrt((Y-center_y[i])**2+(X-center_x[i])**2)
        
        data1[dist<18]=0
    
    return data1

def create_disconnected_components(data,label,n_comp):
    data1=np.copy(data)
    height,width=data1.shape
    randint_height=extract_random_integers(0, height-100, n_comp)
    randint_width=extract_random_integers(0, width-100, n_comp)
    
   
    center_y,center_x=randint_height,randint_width

    for i in range(n_comp):
    
        Y, X = np.ogrid[:height, :width]
        dist= np.sqrt((Y-center_y[i])**2+(X-center_x[i])**2)
        
        data1[dist<18]=label
    
    return data1
    

def detect_holes(image):
    # Convert the image to grayscale if it is not already
    image = np.transpose(image, (1, 0))
    image = BinaryzeAllMask(image).astype(np.uint8)
    #print(np.unique(image))
    mask_fill=image.copy()
    filled_img = np.zeros_like(mask_fill)
    holes_img = np.zeros_like(mask_fill)
    
    
    # Find contours
    contours, hierarchy= cv2.findContours(mask_fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on black image
    contour_img = np.zeros_like(mask_fill)
    
    cv2.drawContours(contour_img, contours, -1, 1, 2)
    
    for contour in contours:
                cv2.fillPoly(filled_img, contours, 1)
    
    #print(np.unique(filled_img))
    holes_img=filled_img-image
    
    contours = measure.find_contours(holes_img, 0.5)
    n_holes=len(contours)
    
    if np.sum(holes_img)>1:
        #print('There are holes in the segmentation, no plausible anatomy')
        holes=True

    else: 
        holes=False
        n_holes=0
        
    return filled_img,contour_img,holes_img,holes,n_holes


def connected_components(image,n):
     # Get a labeling of the k largest objects in the image.
     # The output will be relabeled from 1 to N.
     labels_out, N = cc3d.largest_k(
       image, k=n, 
       connectivity=4, delta=0,
       return_N=True,
     )
    
     labels_out=labels_out.astype(np.uint8)
     # labels_out[labels_out==2]=0
     # plt.imshow(labels_out)
     # plt.show()
     # print(np.unique(labels_out), np.sum(labels_out))
     for i in range(len(np.unique(labels_out))-1):
         n_points=len(np.where(labels_out==i+1)[0])
         #print(n_points, 'classe',i+1,np.unique(labels_out))
         if n_points<15:
             N=N-1

     return labels_out, N
 
def search_disconnected_components(image,n_classes):
     # Get a labeling of the k largest objects in the image.
     # The output will be relabeled from 1 to N.
     for i in range(n_classes):
         
        data_n=SeeSpecificStructure(i+1, image)
        data_b_bi=BinaryzeAllMask(data_n)
        
        plt.imshow(data_b_bi, cmap='gray')
        plt.show()

        _,N=connected_components(data_b_bi)
        
        if N>1:
            print('There are disconnected componentes in the class', i+1,', no anatomically plausible')
         
     return N    
 
def DefineContour(structure):
    # Convert the image to grayscale if it is not already
    image = np.transpose(structure, (1, 0))
    image = BinaryzeAllMask(image).astype(np.uint8)
    #print(np.unique(image))
    mask_fill=image.copy()
       
    # Find contours
    contours, hierarchy= cv2.findContours(mask_fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on black image
    contour_img = np.zeros_like(mask_fill)
    
    cv2.drawContours(contour_img, contours, -1, 1, 2)
    
    return contour_img  



def detect_holes_new(image):
    mask_fill=image.copy()
    
    mask_fill,_=connected_components(mask_fill)
    # plt.imshow(mask_fill, cmap='gray')
    # plt.show()
    
    #Find biggest
    contour_img=DefineContour(mask_fill)
    plt.imshow(contour_img, cmap='gray')
    plt.show()
    contour_img=np.where(contour_img==1)

    # Crop to see only inside (ignoring disconnected componenents)
    x_max,x_min=np.max(contour_img[0]),np.min(contour_img[0])
    y_max,y_min=np.max(contour_img[1]),np.min(contour_img[1])
    
    # Convert the image to grayscale if it is not already
    mask_fill = np.transpose(mask_fill, (1, 0))
    mask_fill = BinaryzeAllMask(mask_fill).astype(np.uint8)
    
    #print(np.unique(image))
    
     
    mask_fill= mask_fill[x_min-5:x_max+5,y_min-5:y_max+5]
    
    plt.imshow(mask_fill, cmap='gray')
    plt.show()
    
    contours = measure.find_contours(mask_fill, 0.8)
    n_holes=len(contours)-1
    
    # plt.imshow(mask_fill, cmap='gray')
    # plt.show()
    #Compute the contours of the mask to be able to analyze each part individually
    
   
    #print(np.unique(image))
    #plt.imshow(mask_fill, cmap='gray')
    #plt.show()
    
    return n_holes   
    
#AllSeg=BinaryzeAllMask(data_GT).astype(np.uint8)

#SpecificSeg=SeeSpecificStructure(1, data_GT)

"""Create holes and disconnected Components"""

###SegWHoles=create_holes(data_GT,2).astype(np.uint8)

###AllSeg_with_di=create_disconnected_components(SegWHoles,3 ,1)

# plt.imshow(AllSeg_with_di, cmap='gray')
# plt.show()

"""Search for holes"""

#image_wo_holes,contour_img,holes,_=detect_holes(AllSeg_with_di)

#detect_holes_new(AllSeg_with_di)

# plt.imshow(image_wo_holes, cmap='gray')
# plt.show()

# plt.imshow(contour_img, cmap='gray')
# plt.show()

# plt.imshow(holes, cmap='gray')
# plt.show()

"""Disconected Components"""

#N=search_disconnected_components(AllSeg_with_di,3)


#points=contours[0].reshape(-1,2)
#plt.scatter(points[:,1],669-points[:,0])

#plt.show()

""" 4 criteria calculate ratio between structures (perimeters)"""

    
def RatioPerInt(structure1,structure2):
    contours_1=DefineContour(structure1)
    contours_2=DefineContour(structure2)
    
    """Calculate intersection"""

    intersection=contours_1*contours_2

    per_1=np.sum(contours_1)
    per_2=np.sum(contours_2)
    Pxinter=np.sum(intersection)
    
    plt.imshow(contours_1)
    plt.show()

    plt.imshow(contours_2)
    plt.show()
    
    plt.imshow(intersection)
    plt.show()

    
    ratio_1=Pxinter/per_1
    ratio_2=Pxinter/per_2
    
    return ratio_1,ratio_2

def RatioPerInt_new(contours_1,contours_2,inter12):
    
    """Perimeter in number of points"""
    
    per_1=len(contours_1)
    per_2=len(contours_2)
    Pxinter=len(inter12)
    
    ratio_1=Pxinter/per_1
    ratio_2=Pxinter/per_2
    
    return ratio_1,ratio_2

def DefineContourSkimage(mask):
    
   # Compute the contours of the mask to be able to analyze each part individually
   contours = measure.find_contours(mask, 0.5)
   
   # Initialize arrays to store the curvature information for each edge pixel
   edge_pixels = []
   
   # Iterate over each contour
   for contour in contours:
       # Iterate over each point in the contour
       if (contour.shape[0]>30):
           for i, point in enumerate(contour):    
            # Store curvature information and corresponding edge pixel
            edge_pixels.append(point)
     
   # Convert lists to numpy arrays for further processing
   edge_pixels = np.array(edge_pixels)            
   
   return edge_pixels  

def dist2points(p1,p2):
    diff=p2-p1
    #print(diff)
    power=np.sum(diff**2)
    #print(power)
    dist=np.sqrt(power)
    
    return dist


def FindCommonPoints(c1,c2,delta):
    CommonPointsc1=[]
    CommonPointsc2=[]
    
    for i in range(c1.shape[0]):
        for l in range(c2.shape[0]):
            if dist2points(c1[i],c2[l])<delta:
                CommonPointsc1.append(c1[i])
                CommonPointsc2.append(c2[l])
    return np.array(CommonPointsc1), np.array(CommonPointsc2)

from scipy.spatial import KDTree

def FindCommonPoints_new(c1, c2, delta):
    # Build KD-tree for the second set of points
    tree = KDTree(c2)
    
    CommonPointsc1 = []
    CommonPointsc2 = []
    
    for point in c1:
        # Query the KD-tree for neighbors within delta
        indices = tree.query_ball_point(point, delta)
        for idx in indices:
            CommonPointsc1.append(point)
            CommonPointsc2.append(c2[idx])
    
    return np.array(CommonPointsc1), np.array(CommonPointsc2)

def RemovePoints(c,inter):
    #c=list(c)
    #inter=list(inter)
    # Create a boolean mask for points to be removed
    mask = np.ones(len(c), dtype=bool)
    
    for point in inter:
        mask &= ~np.all(c == point, axis=1)
    
    # Apply the mask to filter out points
    filtered_points = c[mask]
    #plt.scatter(filtered_points[:, 1], filtered_points[:,0])
    #plt.show()
    
    return filtered_points         
      

# SpecificSeg0=SeeSpecificStructure(0, data_GT) #Background
# SpecificSeg1=SeeSpecificStructure(1, data_GT) # Endocardium
#SpecificSeg2=SeeSpecificStructure(2, data_GT) #Myocardium
# SpecificSeg3=SeeSpecificStructure(3, data_GT) # LA

# contour_0=DefineContourSkimage(SpecificSeg0)
# contour_1=DefineContourSkimage(SpecificSeg1)
#contour_2=DefineContourSkimage(SpecificSeg2)
# contour_3=DefineContourSkimage(SpecificSeg3)

# plt.scatter(contour_2[:,1],contour_2[:,0])

# # Set equal scaling
# plt.axis('equal')
# plt.show()

import numpy as np
from sklearn.decomposition import PCA

def derivative(x):
    derivative=[]
    for i in range(len(x)-1):
        derivative.append(x[i+1]-x[i])
    return derivative

def CropX(c, th):
    points_remove=[]
    for point in c:   
        if point[1]< th:
            points_remove.append(point)
            
    filtered_c=RemovePoints(c, points_remove)

    plt.scatter(filtered_c[:,1],filtered_c[:,0])

    # Set equal scaling
    # plt.axis('equal')
    # plt.show()

    return filtered_c  

def CropY(c, th):
    points_remove=[]
    for point in c:   
        if point[0]< th:
            points_remove.append(point)
            
    filtered_c=RemovePoints(c, points_remove)

    # plt.scatter(filtered_c[:,1],filtered_c[:,0])

    # # Set equal scaling
    # plt.axis('equal')
    # plt.show()

    return filtered_c        
            
# pca = PCA(n_components=2)
# pca.fit(contour_2-np.mean(contour_2,axis=0))

# print(pca.explained_variance_ratio_)

# contour_rotation= contour_2-np.mean(contour_2,axis=0) 
# contour_rotation=np.matmul(contour_rotation,pca.components_.T)

# plt.scatter(contour_rotation[:,1],contour_rotation[:,0])

# # Set equal scaling
# plt.axis('equal')
# plt.show()


# contour_rotation=CropY(contour_rotation, 100)

# contour_rotation=np.matmul(contour_rotation,pca.components_)

# contour_rotation=contour_rotation+ np.mean(contour_2,axis=0)

# plt.scatter(contour_rotation[:,1],contour_rotation[:,0])

# # Set equal scaling
# plt.axis('equal')
# plt.show()



from scipy.spatial import distance as dist


def FindFirstPoint(contour,first_ind):
    #first_ind: first indice of the point which we want to find if this is the first in the open contour 
    ind=first_ind
    distance=0
    while distance<3:
        distance=dist2points(contour[ind],contour[ind-1])
        ind=ind-1 
    return ind+1    

def order_contour_points(points,n_class=1):
    
    if n_class==2:
        #top righ index
        first_index = FindFirstPoint(points,0)
        first_point=points[first_index]
    
    elif n_class=='centerline':
        
        first_index=np.argmax(points[:,1])
        first_point = points[first_index]
    
    else:
        # Find the bottom-left point
        first_index = np.argmin(points[:, 0] + points[:, 1])
        first_index = FindFirstPoint(points,first_index)
        first_point = points[first_index]
    
    # Initialize ordered points with the top-left point
    ordered_points = [first_point]
    
    # Remove the top-left point from the list of points
    points = np.delete(points, first_index, axis=0)
    
    while points.shape[0] > 0:
        # Find the closest point to the last point in ordered_points
        last_point = ordered_points[-1]
        distances = np.linalg.norm(points - last_point, axis=1)
        nearest_index = np.argmin(distances)
        nearest_point = points[nearest_index]
        
        # Add the nearest point to ordered_points and remove it from points
        ordered_points.append(nearest_point)
        points = np.delete(points, nearest_index, axis=0)
    
    return np.array(ordered_points)

def AddDistancePoint(contour):

 distance=np.zeros(contour.shape[0])    
 for i in range(contour.shape[0]):
     if i==0:
         distance[i]=0
     else:
         distance[i]=dist2points(contour[i],contour[i-1])+distance[i-1] 
         
 return distance    


from scipy import interpolate
"""Make interpolation in order to put the points equidistance"""

def ContourInterpolation(contour):
      
  distance=AddDistancePoint(contour)
  
  fx = interpolate.interp1d(distance, contour[:,1])
  fy= interpolate.interp1d(distance, contour[:,0])
  
  # Fit a polynomial of degree 2 to the rotated coordinates
  #coeffs_x = np.polyfit(distance, contour[:,1], degree)
  # Fit a polynomial of degree 2 to the rotated coordinates
  #coeffs_y = np.polyfit(distance, contour[:,0], degree)
  dist_max=distance[-1]
  dist_p_points=1
  
  #plt.plot(distance,contour[:,1])
  #plt.show()

  distances= np.arange(0,dist_max,dist_p_points)
  #print(distances)
  #x_values=np.polyval(coeffs_x, distances)
  #y_values=np.polyval(coeffs_y, distances)
  
  xnew = fx(distances)
  ynew=  fy(distances)
  
  #contourfit=np.transpose(np.array([y_values,x_values]))
  contourfit=np.transpose(np.array([ynew,xnew]))
  
  plt.scatter(contourfit[:, 1], contourfit[:, 0])
  plt.show()
  
  
  return contourfit


# contour_1_fit=ContourInterpolation(contour_1)
# contour_2_fit=ContourInterpolation(contour_2)
# contour_3_fit=ContourInterpolation(contour_3)

# plt.scatter(contour_3[:, 1][0:20], contour_3[:, 0][0:20], cmap='gray')
# plt.show()

# plt.scatter(contour_1_fit[:, 1][0:700], contour_1_fit[:, 0][0:700])
# plt.show()

# inter_13,inter_31=FindCommonPoints_new(contour_1_fit,contour_3_fit,1)
# inter_21,_=FindCommonPoints_new(contour_2_fit,contour_1_fit,1)

# new_contour_1=RemovePoints(contour_1_fit,inter_13)
# new_contour_2=RemovePoints(contour_2_fit,inter_21)
# new_contour_3=RemovePoints(contour_3_fit,inter_31)


# new_contour_3=order_contour_points(new_contour_3)
# new_contour_2=order_contour_points(new_contour_2,2)
# new_contour_1=order_contour_points(new_contour_1,2)

# plt.scatter(new_contour_1[:, 1], new_contour_1[:, 0], cmap='gray')
# plt.show()

# plt.scatter(new_contour_2[:, 1], new_contour_2[:, 0], cmap='gray')
# plt.show()

from scipy.signal import find_peaks

def RemovePointsMyo(points,x=1):
    
    """Remove the undesired points for the Myocardium"""
    
    TopRightIndex=np.argmax(points[:, x-1] + points[:, x])
    PointsToRemove_1=points[0:TopRightIndex]
    
    #distance=AddDistancePoint(points)
    
    #plt.plot(distance,points[:,1])
    
    # Find peaks (local maxima)
    #peaks, _ = find_peaks(points[:,1])
    
    """Points have to be in order along the contour, find peaks in X (because the there is always a difference) """
    # Find peaks with a minimum height of 2 and minimum distance of 1
    peaks, properties = find_peaks(points[:,x], height=2, distance=3)
    PointsToRemove_2=points[peaks[-1]:]
    
    # Print the indices of the peaks
    #print("Indices of local maxima:", peaks)
    
    # # Print the values of the peaks
    # print("Values of local maxima:",points[:,1][peaks])
    
    # Plot the data and the peaks
    plt.plot(points[:,x])
    plt.plot(peaks, points[:,x][peaks], "x")
    plt.title('Local Maxima in Data')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()
    filtered_1=RemovePoints(points,PointsToRemove_1)
    filtered_2=RemovePoints(filtered_1,PointsToRemove_2)
    #print(PointsToRemove_1)
    #print(PointsToRemove_2)
    
    return  filtered_2  

# new_contour_2=RemovePointsMyo(new_contour_2,x=1)    

# teste=order_contour_points(new_contour_2,2)

# plt.scatter(teste[:, 1][0:700], teste[:, 0][0:700], cmap='gray')
# plt.show()


# """Intersecion endo bckg"""

# inter_01,_=FindCommonPoints_new(contour_1,contour_0,1)
# ratio1,ratio2=RatioPerInt_new(contour_0,contour_1,inter_01)

# """Intersecion MYO LA"""

# inter_23,_=FindCommonPoints_new(contour_2,contour_3,1)
# ratio1,ratio2=RatioPerInt_new(contour_2,contour_3,inter_23)


# plt.scatter(new_contour_1[:, 1], new_contour_1[:, 0], cmap='gray')
# plt.show()

# plt.scatter(new_contour_2[:, 1], new_contour_2[:, 0], cmap='gray')
# plt.show()

# plt.scatter(new_contour_3[:, 1][0:270], new_contour_3[:, 0][0:270], cmap='gray')
# plt.show()

   
    

def CurvatureValues_new(contour,window_size):
   
   # Initialize arrays to store the curvature information for each edge pixel
   curvature_values = []
   
    # Iterate over each point in the contour
   for i, point in enumerate(contour):
    # Compute the curvature for the point
    # We set the window size to 1/5 of the whole contour edge. Adjust this value according to your specific task
    #window_size=int(len(contour)/window_size_ratio)
    curvature = compute_curvature(point, i, contour, window_size)
     
     # Store curvature information and corresponding edge pixel
    curvature_values.append(curvature)
    
   # Convert lists to numpy arrays for further processing
   curvature_values = np.array(curvature_values)
   
   plt.scatter(contour[:, 1], contour[:, 0],c=curvature_values, cmap='jet',vmin=-0.12, vmax=0.12)

   #plt.scatter(edge[:, 1][0:2000], edge[:, 0][0:2000])
   plt.colorbar(label='Curvature')
   plt.title("Curvature of Edge Pixels")
   plt.show()
   

   return  curvature_values  

#curvature_values=CurvatureValues_new(new_contour_1,20)





# def draw_contour(structure,contours):
#     # Convert the image to grayscale if it is not already
#     image = np.transpose(structure, (1, 0))
    
#     mask_fill=image.copy()
    
#     new_contour=(contours.reshape(len(contours),1,2)).astype(np.int32)
#     new_countor=(new_contour,)
       
#     # Draw contours on black image
#     contour_img = np.zeros_like(mask_fill)
    
#     cv2.drawContours(contour_img, new_countor, -1, 1, 2)
    
#     return contour_img 






# ratio_1,ratio_2=RatioPerInt(SpecificSeg0,SpecificSeg1)


            


"""Critério 5: Espessura ??"""

"""Critério 7: Curvatura"""

 

def CurvatureValues(mask,min_contour_length, window_size_ratio):
    # Compute the contours of the mask to be able to analyze each part individually
   contours = measure.find_contours(mask, 0.5)
   #print(len(contours))
   # Initialize arrays to store the curvature information for each edge pixel
   curvature_values = []
   edge_pixels = []

   # Iterate over each contour
   for contour in contours:
       # Iterate over each point in the contour
       for i, point in enumerate(contour):
           # We set the minimum contour length to 20
           # You can change this minimum-value according to your specific requirements
           #print(contour.shape)
           if contour.shape[0] > min_contour_length:
               # Compute the curvature for the point
               # We set the window size to 1/5 of the whole contour edge. Adjust this value according to your specific task
               window_size = int(contour.shape[0]/window_size_ratio)
               curvature = compute_curvature(point, i, contour, window_size)
                
                # Store curvature information and corresponding edge pixel
               curvature_values.append(curvature)
               # Store curvature information and corresponding edge pixel
               edge_pixels.append(point)
   
    
   # Convert lists to numpy arrays for further processing
   curvature_values = np.array(curvature_values)
   edge_pixels = np.array(edge_pixels)            
   
   return edge_pixels, curvature_values     

def compute_curvature(point, i, contour, window_size):
    # Compute the curvature using polynomial fitting in a local coordinate system
    if (((i - window_size // 2)<0 )&(dist2points(contour[0], contour[-1])<1)):
        
        start=i - window_size // 2
        end=-1
        contour_before=contour[start:end]
        contour_after=contour[0:i + window_size // 2 + 1]
        neighborhood= np.concatenate((contour_before, contour_after))
     
    elif ((i + window_size // 2> len(contour)) &(dist2points(contour[0], contour[-1])<1)):
        
        start=i - window_size // 2
        end=-1
        contour_before=contour[start:end]
        contour_after=contour[0:i + window_size // 2 + 1 - len(contour)]
        neighborhood= np.concatenate((contour_before, contour_after))
    
        
    else:
            
        # Extract neighboring edge points
        start = max(0, i - window_size // 2)
        end = min(len(contour), i + window_size // 2 + 1)
        neighborhood = contour[start:end]
    
    # #Extract neighboring edge points
    # start = max(0, i - window_size // 2)
    # end = min(len(contour), i + window_size // 2 + 1)
    # neighborhood = contour[start:end]

    # Extract x and y coordinates from the neighborhood
    x_neighborhood = neighborhood[:, 1]
    y_neighborhood = neighborhood[:, 0]

    # Compute the tangent direction over the entire neighborhood and rotate the points
    tangent_direction_original = np.arctan2(np.gradient(y_neighborhood), np.gradient(x_neighborhood))
    tangent_direction_original.fill(tangent_direction_original[len(tangent_direction_original)//2])

    # Translate the neighborhood points to the central point
    translated_x = x_neighborhood - point[1]
    translated_y = y_neighborhood - point[0]


    # Apply rotation to the translated neighborhood points
    # We have to rotate the points to be able to compute the curvature independent of the local orientation of the curve
    rotated_x = translated_x * np.cos(-tangent_direction_original) - translated_y * np.sin(-tangent_direction_original)
    rotated_y = translated_x * np.sin(-tangent_direction_original) + translated_y * np.cos(-tangent_direction_original)

    # Fit a polynomial of degree 2 to the rotated coordinates
    coeffs = np.polyfit(rotated_x, rotated_y, 2)


    # You can compute the curvature using the formula: curvature = |d2y/dx2| / (1 + (dy/dx)^2)^(3/2)
    # dy_dx = np.polyval(np.polyder(coeffs), rotated_x)
    # d2y_dx2 = np.polyval(np.polyder(coeffs, 2), rotated_x)
    # curvature = np.abs(d2y_dx2) / np.power(1 + np.power(dy_dx, 2), 1.5)

    # We compute the 2nd derivative in order to determine whether the curve at the certain point is convex or concave
    curvature = np.polyval(np.polyder(coeffs, 2), rotated_x)

    # Return the mean curvature for the central point
    return np.mean(curvature)       

# # Set minimum length of the contours that should be analyzed
# min_contour_length = 20
# # Set the ratio of the window size (contour length / window_size_ratio) for local polynomial approximation
# window_size_ratio = 30

# edge,curvature_values=DefineContoursSkimage(SpecificSeg1,min_contour_length, window_size_ratio)

# plt.scatter(edge[:, 1], edge[:, 0],c=curvature_values, cmap='jet',vmin=-0.12, vmax=0.12)

# #plt.scatter(edge[:, 1][0:2000], edge[:, 0][0:2000])
# plt.colorbar(label='Curvature')
# plt.title("Curvature of Edge Pixels")
# plt.show()



def CheckPlausibility(image,n_classes):
    
    plt.imshow(image, cmap='gray')
    plt.show()
    n_holes_class=[]
    holes_imgs=[]
    clean_img=[]
    cr1,cr2,cr3,cr4,cr5,cr6, cr7= np.zeros((3,)),np.zeros((2,)),np.zeros((3,)),np.zeros((2,)),np.zeros((1,)),np.zeros((1,)),np.zeros((1,))
    
        
    for i in range(n_classes):
         
        data_n=SeeSpecificStructure(i+1, image)
        data_b_bi=BinaryzeAllMask(data_n)
       
        plt.imshow(data_b_bi, cmap='gray')
        plt.show()
        
        filled_img,_,holes,_,n_holes=detect_holes(data_b_bi)
        n_holes_class.append(n_holes)
        holes_imgs.append(holes)
        
        #print(n_holes)
        if n_holes>0:
            print('There are',n_holes,' holes in the',Name_classes[i+1],' segmentation, no plausible anatomy')
            cr1[i]=1
             
        _,Ni=connected_components(data_b_bi,3)
        pp_img,_=connected_components(filled_img,1)
        clean_img.append(pp_img)
        
        if Ni>1:
            print('There are', Ni,' disconnected componentes in the class', Name_classes[i+1],', no anatomically plausible')
            cr3[i]=1 
        
    """Common criteria (not need to iterate for each class)"""
    
    # Presence of holes between LV and MYO
    mask_myo_lv=np.copy(image)
    mask_myo_lv[mask_myo_lv==3]=0
    data_myo_lv=BinaryzeAllMask(mask_myo_lv)
    
    _,_,total_holes,_,holes_myo_lv=detect_holes(data_myo_lv)
    holes_int=holes_myo_lv-n_holes_class[0]-n_holes_class[1]
    
    if (holes_int)>0:
        print('There are',holes_int,' holes between LV and MYO, no plausible anatomy')
        cr2[0]=1
        #Add intersection holes to Myocardium to the contours be considered as intersection LV-MYO and be removed
        holes_myo_lv= total_holes-holes_imgs[0]-holes_imgs[1]
        clean_img[1]=clean_img[1]+holes_myo_lv
        
    
    # Presence of holes between LV and LA
    mask_la_lv=np.copy(image)
    mask_la_lv[mask_la_lv==2]=0
    data_la_lv=BinaryzeAllMask(mask_la_lv)
    
    _,_,total_holes_lalv,_,n_holes_la_lv=detect_holes(data_la_lv)
    holes_int_lav=n_holes_la_lv-n_holes_class[0]-n_holes_class[2]
    
    if holes_int_lav>0:
        print('There are',holes_int_lav,' holes between LV and LA, no plausible anatomy')
        cr2[1]=1
        #Add intersection holes to la to the contours be considered as intersection LV-LA and be removed
        holes_la_lv= total_holes_lalv-holes_imgs[0]-holes_imgs[2]
        clean_img[2]=clean_img[2]+holes_la_lv
    
    """ Critera 4:Size of the perimeter by which the LV touches the background or the MYO touches the LA exceeds a certain threshold."""
    
    """Image has to be clean, without holes or disconnected components"""
    backgrd=SeeSpecificStructure(0, image) # Background
    #clean background
    filled_img,_,_,_,_=detect_holes(backgrd)
    Seebackgrd,_=connected_components(filled_img,1)
    
    SpecificSeg0=np.transpose(Seebackgrd, (1, 0)) #Background
    SpecificSeg1=np.transpose(clean_img[0], (1, 0)) # Endocardium
    SpecificSeg2=np.transpose(clean_img[1], (1, 0)) # Myocardium
    SpecificSeg3=np.transpose(clean_img[2], (1, 0)) # LA
    
    """ Find Contours"""
    contour_0=DefineContourSkimage(SpecificSeg0)
    contour_1=DefineContourSkimage(SpecificSeg1)
    contour_2=DefineContourSkimage(SpecificSeg2)
    contour_3=DefineContourSkimage(SpecificSeg3)
    
    plt.scatter(contour_2[:,1],contour_2[:,0])
    plt.show()
    
    "Interpolation in order to make all the points equidistances"
    
    contour_0=ContourInterpolation(contour_0)
    contour_1=ContourInterpolation(contour_1)
    contour_2=ContourInterpolation(contour_2)
    contour_3=ContourInterpolation(contour_3)
    
    
    "Find intersections between contours"
    
    inter_13,inter_31=FindCommonPoints_new(contour_1,contour_3,3) # Endo and LA
    inter_21,_=FindCommonPoints_new(contour_2,contour_1,4) # Myo and Endo
    
    """Remove intersections from the contours"""
    new_contour_1=RemovePoints(contour_1,inter_13)
    new_contour_2=RemovePoints(contour_2,inter_21)
    new_contour_3=RemovePoints(contour_3,inter_31)
    
    """Order contour"""
    
    new_contour_1=order_contour_points(new_contour_1,2)
    new_contour_2=order_contour_points(new_contour_2,2) # fix problem with order Myo
    new_contour_3=order_contour_points(new_contour_3) # fix problem with order Atrium 

    """Intersecion endo bckg"""

    inter_01,_=FindCommonPoints_new(new_contour_1,contour_0,1)
    ratio_IntBckg,ratio_IntLV=RatioPerInt_new(contour_0,new_contour_1,inter_01)

    """Intersecion MYO LA"""

    inter_23,_=FindCommonPoints_new(new_contour_2,new_contour_3,1)
    ratio_IntMyo,ratio_IntLA=RatioPerInt_new(new_contour_2,new_contour_3,inter_23)
       
    
    print('Ratio (Intersetion between LV and background/ Background: ',ratio_IntBckg,' and ratio.../ Endocardium: ', ratio_IntLV)    
    print('Ratio (Intersetion between MYO and LA/ MYO: ',ratio_IntMyo,' and ratio.../ LA: ', ratio_IntLA)    
    
    # If ratio...> th: Falta calcular
    cr4[0],cr4[1]= 999,999 
    
    """Criterio Myocardium thickness (2)"""
    cr5[0],cr6[0]= 999,999 
    
        
    """Criteria 7: Curvature"""
    
    # Set the ratio of the window size (contour length / window_size_ratio) for local polynomial approximation
    window_size = 30
    
    #plt.scatter(new_contour_2[:,1][0:100],new_contour_2[:,0][0:100])
    #plt.show()
    new_contour_2=RemovePointsMyo(new_contour_2) # remove the 2 segments
    
    plt.scatter(new_contour_2[:,1],new_contour_2[:,0])
    plt.show()
    curvature_values_1=CurvatureValues_new(new_contour_1,window_size)
    curvature_values_2=CurvatureValues_new(new_contour_2,window_size)
    curvature_values_3=CurvatureValues_new(new_contour_3,window_size)
    
    #if curvature_values > th: ...
    
    cr7[0]=999
    
    
    plausibility=[cr1,cr2,cr3,cr4,cr5,cr6,cr7] 
    return plausibility   

#plaus=CheckPlausibility(AllSeg_with_di,3)

#print(plaus)

"""Critérios:
    1-(3 criteria) Presence of holes in the Left Ventricle (LV), Myocardium (MYO), and Left Atrium (LA).
    2-(2 criteria) Presence of holes between the LV and MYO, or between the LV and LA.
    3-(3 criteria) Presence of more than one LV, MYO, or LA. 
    4-(2 criteria) Size of the perimeter by which the LV touches the background or the MYO touches the LA exceeds a certain threshold.
    5-(1 criterion) Ratio between the minimal and maximal thickness of the MYO is below a given threshold
    6-(1 criterion) Ratio between the width of the LV and the average thickness of the MYO exceeds a certain threshold. Both width and thickness are computed as the total width of the structure at the middle-point of the embedded bounding box. The goal is to identify situations where the MYO is too thin with respect to the size of the LV.
    7-Curvature
"""
 
import cv2
import skimage.morphology, skimage.data

