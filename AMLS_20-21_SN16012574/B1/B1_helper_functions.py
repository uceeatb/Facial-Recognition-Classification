''' This module comprises three low-level functions that were utilised for Task B1.
    The functions are listed in order of their use during the preprocessing pipeline.
    Detailed commnets are shown to provide reader with low-level
    understanding of pipeline of each function. '''

import numpy as np
import cv2
from PIL import Image

'''# =========================================================================================================================================================='''
'''#   FUNCTION TO HISTOGRAM EQUALISE IMAGES TO MAKE EDGES MORE APPARENT   '''
'''# =========================================================================================================================================================='''

def B1_equalise_images(image_list):
     
    
    EQ_image_list = []
    
    for i, image in enumerate(image_list):
        
        CLAHE = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8,8))            # Contrast is enhanced in images, such that race of 
        EQ_image = CLAHE.apply(image)                                          # person is suppressed whilst face edges are amplified
        
        EQ_image_list.append(EQ_image)
        
        #if (i > 0 and i % 5000 == 0) or (i == 1):                              # Printing updates.   
            #print(i, 'images equalised') 
    
    EQ_image_list = np.asarray(EQ_image_list)
        
    return EQ_image_list

#%%


'''# =========================================================================================================================================================='''
'''#   FUNCTION TO ADD MASK TO CENTRE OF FACES: ONLY OUTER EDGES REQUIRED  '''
'''# =========================================================================================================================================================='''

def B1_mask_images(image_list):
     
    eye_coords = [(30, 30), (170, 90)]                                         # Edges of eyes, nose and mouth are not related 
    mouth_coords = [(60, 90), (140, 145)]                                      # to face shape, thus can be ignored by classifer. 
    colour = (0,0,0)                         #black                            # Variation in these 'inner edges' across the images  
    thickness = -1                                                             # Can be nullified by applying a common 'mask', which 
                                                                               # covers the general area of eyes, nose and mouth.
    masked_image_list = []                                                     # Mask is peuposely made to be small so that it does 
                                                                               # NOT encroach upon the outer edges of the faces 
    for i , image in enumerate(image_list):
        
        masked_image = image.copy()                                             
        
        masked_image = cv2.rectangle(masked_image, eye_coords[0], eye_coords[1], colour, thickness)        # Applying the mask to cover eyes 
                                                 
        masked_image = cv2.rectangle(masked_image, mouth_coords[0], mouth_coords[1], colour, thickness)                 
        
        #if (i > 0 and i % 5000 == 0) or (i == 1):                                          # Printing updates.   
            #print(i, 'images masked') 
        
        masked_image_list.append(masked_image)
    
    masked_image_list = np.asarray(masked_image_list)
        
    return masked_image_list

#%%


'''# =========================================================================================================================================================='''
'''#        FUNCTION TO RESIZE B1 FACES TO REDUCE FEATURE VECTOR SIZE      '''
'''# =========================================================================================================================================================='''

def B1_resize_images(image_list):
     
    resized_image_list = []
    
    for i, image in enumerate(image_list):
        
       image_object = Image.fromarray(image)                                   # Convert array to image object and resize it. Resize_value = 96 
       resized_image = image_object.resize((96,96), Image.ANTIALIAS)           # chosen because it more than quarters the 200x200 cropped faces
       resized_image = np.array(resized_image)                                 # and is a multiple of 8, whcih facilitates HOG feature extraction
    
       #if (i > 0 and i % 5000 == 0) or (i == 1):                              # Printing updates.   
            #print(i, 'images resized')

       resized_image_list.append(resized_image)    
            
    resized_image_list = np.asarray(resized_image_list)
        
    return resized_image_list



