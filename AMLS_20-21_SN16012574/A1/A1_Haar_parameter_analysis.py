

'''NOTE TO USER: THIS FILE IS FOR TRANSPARENCY PURPOSES ONLY.
   IT IS NOT DIRECTLY IMPLEMENTED WITHIN THE ASSIGNMENT. IT MERELY EXPLAINS
   HOW INPUT PARAMETERS WERE CHOSEN FOR THE PREPROCESSING STAGE OF TASK A1.
   
   THE USER CAN RUN THIS FILE TO UNDERSTAND THIS PROCESS. IT IS BEST TO RUN
   THIS FILE USING SPYDER, SO THAT PLOTS CAN BE IMAGED.
   
   PLEASE KEEP THIS FILE IN THE CURRENT FILE PATH. IT'S FUNCTIONS USE RELATIVE
   DIRECTORIES TO OPERATE, SO MOVING THE FILE WILL CAUSE ERRORS.
   
   
   
   CURRENT CELL: CELL 1
   
   IF OPENING THIS FILE DIRECTLY FROM ITS OWN DIRECTORY ( /A1 ) :
       - PLEASE RUN CELL 2 AND IGNORE CELL 3
   
   IF OPENING THIS FILE FROM THE SAME DIRECTORY AS MAIN.PY :
       - PLEASE RUN CELL 3 AND IGNORE CELL 2
                                                                            '''                                                              

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2                                                                           

#%%

''''IF OPENING THIS FILE DIRECTLY FROM ITS OWN DIRECTORY ( /A1 ) :'''

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
os.chdir(parent_directory)

from common_functions import load_images, load_labels, split_data

#%%

''''IF OPENING THIS FILE FROM THE SAME DIRECTORY AS MAIN.PY :'''

from common_functions import load_images, load_labels, split_data


#%%


'''# =========================================================================================================================================================='''
'''#    FUNCTION TO LOAD IMAGES, LABELS AND SPLIT DATA FOR TASK A1 or A2   '''
'''# =========================================================================================================================================================='''
    
def A1_load_data(task_name, random_int, image_set_request):                                                  
        
    image_list = load_images(task_name, image_set_request)                           # Loading A1 images from celeba directory.
    
    label_list = load_labels(task_name, image_set_request)
    
    Xtrain_img, Xtest_img, ytrain, ytest = split_data(image_list,                    # Data split before preprocess for unbiased parameter evaluation. 
                                                      label_list,                    # Training set: 80%.   Test set: 20%.
                                                      random_int = random_int) 
    
    return Xtrain_img, Xtest_img, ytrain, ytest

 
print('\nLoading images from A1 data directory...\n')
X_train_images_A1, X_test_images_A1, y_train, y_test = A1_load_data(task_name = 'A1',
                                                                    image_set_request = 'train_test',
                                                                    random_int = 0)
print("------------------------------------------------------------------")



#%%
'''# =========================================================================================================================================================='''
'''#       FUNCTION TO CALCULATE FACE SIZES OUTPUT BY HAAR CLASSIFER       '''
'''# =========================================================================================================================================================='''

def detect_faces_dimensions(image_collection, Min_neighbours = 2, scaleFactor = 1.05):
          
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    collection_copy = image_collection.copy()
    dimensions = [collection_copy.shape[1], collection_copy.shape[2]]
                                                                                    # Create list containing dimensions of faces output 
    all_faces_dimensions = np.zeros((collection_copy.shape[0], 2))                  # by the classifier.

    ''' LOOPING OVER EACH IMAGE... '''
    for i in range(len(collection_copy)):                                 

        grey_image = collection_copy[i]                                   
        faces_rect = cascade.detectMultiScale(grey_image, scaleFactor, Min_neighbours)     #     CALL THE CLASSSIFIER.
        
        # NO FACE DETECTED
        if len(faces_rect) == 0:                                                    # When NO face detected, shortest dimension of raw
                                                                                    # image output as the dimensions of the 'face'.
            all_faces_dimensions[i] = [min(dimensions), min(dimensions)]
                                                                              
        # FACE(S) DETECTED 
        else:    
            img_faces_regions = np.asarray(faces_rect).T                            # Haar may incorrectly detect multiple faces when
            img_faces_dimensions = img_faces_regions[2]                             # each image only contains one person. Safest approach   
            largest_face_index = np.argmax(img_faces_dimensions, axis = 0)          # Is to output ONLY the largest face from the classifier.   
            best_face_rect = faces_rect[largest_face_index]                         # These lines of code find this 'largest face'.

            h = best_face_rect[2]
            w = best_face_rect[3]
            
            all_faces_dimensions[i] = [h, w]

        if (i > 0 and i % 500 == 0) or (i == 1):                                    # Printing updates.
            print(i, 'faces detected')
    
    print("ALL FACES DETECTED")
    print("------------------------------------------------------------------")
        
    return all_faces_dimensions

#%%

dimensions = detect_faces_dimensions(X_train_images_A1)


#%%
'''The variation in dimensions of output faces is plot via a histogram
   to indicate the spread of sizes of faces detected by the classifier. '''

plt.subplots(1, 1, figsize=(5,4))
cm = plt.cm.get_cmap('gist_heat_r')

bins = np.linspace(40, 180, 30)

Y,X = np.histogram(dimensions, bins, density=False, range = (40, 180))              # Plotting a histogram to show distribution of face sizes
x_span = X.max()-X.min()
C = [cm(((x-X.min())/x_span)) for x in X]

plt.bar(X[:-1],Y,color=C,width=X[1]-X[0])   

plt.xlabel('Dimension of face (height or width) / pixels')
plt.ylabel('Frequency')

plt.show()

'''In order to determine the best size for resizing faces, the median 
   dimension for faces obtained from all training images was calculated. '''

median_dimension = np.median(np.median(dimensions, axis = 0)).astype(int)           # Since faces are squares, median across both axes taken.
mean_dimension = np.mean(np.mean(dimensions, axis = 0)).astype(int)

print('\nMedian dimension:', median_dimension)
print('Mean dimension:', mean_dimension)


print('\nExpected results: median_dimension = 106, mean_dimension = 107.\n')

'''Feature extraction uses 8x8 DCT transforms (standard for JPEG)').
   Thus, faces should be resized with dimensions that are a multiple of 8.
   Closest multiple of 8 to 106 is 104. All faces resized to 104x104. '''

print('All faces will be resized to 104x104 to faciliate feature extraction via DCT.')
print("------------------------------------------------------------------")


    #%%

''' The Haar classifier uses two key parameters for classification:
    minNeighbours and ScaleFactor. The accuracy of the classifier was
    tested using a range of values to determine the optimal parameters.'''
    

#%%
'''# =========================================================================================================================================================='''
'''#               FUNCTION TO TEST SUCCESS OF HAAR CLASSIFIER             '''
'''# =========================================================================================================================================================='''

def haar_success_rate(image_collection, Min_neighbours, scaleFactor):
    
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    collection_copy = image_collection.copy()
    
    unsuccessful = 0                                                                # Initialise couters for both successful and
    successful = 0                                                                  # unsuccesful face detections.

    ''' LOOPING OVER EACH IMAGE... '''
    for i in range(len(collection_copy)):    

        grey_image = collection_copy[i]
        
        faces_rect = cascade.detectMultiScale(grey_image, scaleFactor, Min_neighbours)      #     CALL THE CLASSIFIER.

        # NO FACE DETECTED
        if len(faces_rect) == 0:
            
            unsuccessful += 1                                                       # Increase counter for failed detection.
        
        # FACE(S) DETECTED
        else:
            
            successful += 1                                                         # Increase counter for successful detection.
                
        if (i > 0 and i % 2000 == 0) or (i == 1):                     
            print(i, 'faces detected')                                              # Printing updates every 2000 images.
    
    total = successful + unsuccessful
    rate = successful/total                                                         # Calculate rate of success.
    
    print("ALL FACES DETECTED")
    print("------------------------------------------------------------------")
    
    return rate
#%%

'''Below are a list of values for minNeighbours = N and ScaleFactor = SF. '''

N = [2, 3, 4, 5, 6]                                                                                     
SF = [1.05, 1.1, 1.15, 1.2]


#%%
'''# =========================================================================================================================================================='''
'''#  FUNCTION TO TEST HAAR SUCCESS IN VARYING minNEIGHBOURS & SCALEFACTOR '''
'''# ==========================================================================================================================================================''' 

def test_haar_parameters(training_images, Neighbours, ScaleFactors):
   
    array = np.zeros((len(Neighbours), len(ScaleFactors)))                          # Create empty dataframe for haar success rates
    haar_rates = pd.DataFrame(array, columns = ScaleFactors, index = Neighbours)    # when classifying with different parameters.      
    
    ''' LOOPING OVER BOTH NEIGHBOURS AND SCALEFACTORS... '''
    for a in Neighbours:
        for b in ScaleFactors:
            
            print('Testing: Neighbours = '+str(a)+', ScaleFactor = '+str(b))
            
            haar_rates.at[a, b] = haar_success_rate(training_images, a, b)          # CALL HAAR_SUCCESS_RATE FUNCTION CONTAINING 
                                                                                            
    print("ALL PERMUTATIONS TESTED")
    print("------------------------------------------------------------------")
    
    rates_array = np.array(haar_rates)                                               # Indexing the parameters that produce the 
    best_params = list(haar_rates.stack().index[rates_array.argmax()])               # highest score for Haar classification.
       
    return haar_rates, best_params

#%%

''' The input data was only the training images, for unbiased evaluation.  '''

haar_scores, haar_best_params = test_haar_parameters(X_train_images_A1, N, SF)         

#%%
print(haar_scores, '\n')

print('Maximum classification score with',
      'Neighbours =', str(haar_best_params[0]),
       ', Scale Factor =', str(haar_best_params[1]))                                # Printing results.

#%%
print('Expected best parameters: Neighbours = 2, Scale Factor = 1.05\n')            # Printing expected results.             

print("Little variation in accuracy of Haar classifier with minNeighbours. ")       # Printing justification to use these parameters.
print("Large classification time does not compensate using smaller ScaleFactor.")
                      
