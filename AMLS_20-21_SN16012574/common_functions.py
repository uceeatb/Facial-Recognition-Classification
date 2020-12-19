
''' This module comprises several functions that were utilised across more
    than one task. Some functions are connected to each other, whilst others
    are not. Detailed commnets are shown to provide reader with low-level
    understanding of pipeline of each function. '''


import numpy as np
import os
from natsort import natsorted
import cv2
import pandas as pd
import time

#%%

'''# =========================================================================================================================================================='''
'''#            FUNCTION TO LOAD RAW IMAGES FROM CHOSEN DATASET            '''
'''# ==========================================================================================================================================================''' 

def load_images(task_name, image_set_request):
    
    home_directory = os.getcwd()                                                        # Capturing home directory for safekeeping
    
    try:                                                                                # Error handling for kernel interuption.
                                                                                        # Since this function changes the environment's directory
                                                                                        # environment must be returned to original directory
        if image_set_request == 'train_test':     
            if task_name == 'A1' or task_name == 'A2':                                  # Changing directory to where images are stored.
                rel_path = home_directory +'/Datasets/celeba/img'
                new_directory = os.path.abspath(rel_path)
                os.chdir(new_directory)
                
            if task_name == 'B1' or task_name == 'B2':
                rel_path = home_directory +'/Datasets/cartoon_set/img'
                new_directory = os.path.abspath(rel_path)
                os.chdir(new_directory)
            
        if image_set_request == 'test_only': 
            if task_name == 'A1' or task_name == 'A2':                                  # Changing directory to where images are stored.
                rel_path = home_directory +'/Datasets/celeba_test/img'
                new_directory = os.path.abspath(rel_path)
                os.chdir(new_directory)
                
            if task_name == 'B1' or task_name == 'B2':
                rel_path = home_directory +'/Datasets/cartoon_set_test/img'
                new_directory = os.path.abspath(rel_path)
                os.chdir(new_directory)
            
        
        list_files = os.listdir('.')
        list_files = natsorted(list_files)                                              # Creating a list of filenames for each image.
        
        images_list = []
        
        if task_name == 'B2':                                                           # Flag = -1 retains all RBG pixel info.     
            flag = -1                                                                   # Flag = 0 converts images to greyscale
        else:                                                                           # Only task B2 (eye colour classification)
            flag = 0                                                                    # requires RGB information to be retained.-
        
        for i, filename in enumerate(list_files):
            
            image = cv2.imread(filename, flag)                                          # Loading the RGB or greyscaled images into an array.    
            
            if flag == -1:                                                              # For task B2, analysis resulted in the conversion of 
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)                          # RGBA images to RBG only, since A channel was deemed redundant
            
            images_list.append(image)
            
            #if (i > 0 and i % 1000 == 0) or (i == 1):                                  # Printing updates.
                #print(i, 'images loaded')
        
        images_list = np.asarray(images_list)
            
        if flag == 0:    
            print("...all images loaded and greyscaled")
            
        else:
            print("...all images loaded and colour (RGB retained)")
    
        os.chdir(home_directory)
    
    
    except KeyboardInterrupt:                                                           # if keyboard interrupt occurs
        os.chdir(home_directory)                                                        # Returning to home directory after accessing images
    
    return images_list

#%%


'''# =========================================================================================================================================================='''
'''#      FUNCTION TO LOAD LABELS OF IMAGES CORESPONDING TO EACH TASK      '''
'''# =========================================================================================================================================================='''

def load_labels(task_name, image_set_request):
    
    home_directory = os.getcwd()                                                         # Capturing home directory for safekeeping    
    
    try:                                                                                 # Error handling for kernel interuption.
                                                                                         # Since this function changes the environment's directory
                                                                                         # to locate images, the if keyboard interrupt occurs
                                                                                         # environment must be returned to original directory
        if image_set_request == 'train_test': 
            if task_name == 'A1' or task_name == 'A2':                                   # Changing directory to where images are stored.
                rel_path = home_directory + '/Datasets/celeba/'
                new_directory = os.path.abspath(rel_path)
                os.chdir(new_directory)
                
            if task_name == 'B1' or task_name == 'B2': 
                rel_path = home_directory + '/Datasets/cartoon_set/'
                new_directory = os.path.abspath(rel_path)
                os.chdir(new_directory)
        
        
        if image_set_request == 'test_only':        
            if task_name == 'A1' or task_name == 'A2':                                   # Changing directory to where images are stored.
                rel_path = home_directory + '/Datasets/celeba_test/'
                new_directory = os.path.abspath(rel_path)
                os.chdir(new_directory)
                
            if task_name == 'B1' or task_name == 'B2': 
                rel_path = home_directory + '/Datasets/cartoon_set_test/'
                new_directory = os.path.abspath(rel_path)
                os.chdir(new_directory)
        
        labels_file = 'labels.csv'
        df = pd.read_csv(labels_file, delimiter = '\t')                                  # Loading labels.csv into a dataframe                                                          
        
        if task_name == 'A1':
            Y = np.asarray(df['gender'])                                                 # Assigning Y to label associated with task data
        
        elif task_name == 'A2':
            Y = np.asarray(df['smiling'])
        
        elif task_name == 'B1':
            Y = np.asarray(df['face_shape'])
        
        elif task_name == 'B2':
            Y = np.asarray(df['eye_color'])    
        
        os.chdir(home_directory)                                                         # Returning to home directory after accessing labels
    
    
    except KeyboardInterrupt:                                                           # if keyboard interrupt occurs
        os.chdir(home_directory)                                                        # Returning to home directory after accessing images
    
    return Y


#%%


'''# =========================================================================================================================================================='''
'''#           FUNCTION TO SPLIT DATA INTO TRAINING AND TEST DATA          '''
'''# =========================================================================================================================================================='''


def split_data(X, Y, random_int = 0):
    
    from sklearn.model_selection import train_test_split
    
    X = np.asarray(X)    
    
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y,                           # Spliting dataset into training and test set
                                                    test_size=0.2,
                                                    random_state=random_int) 
    
    print("...data has been split into training and test sets.")
    
    ''' It is important to note that the training/test set are split with 
        80/20 ratio of the raw data correspondingly. This choice is justified by:
           - the size of the raw dataset 
           - the use of cross-validation within the training set when
             the learning algorithm is implemented.                         '''

    
    return Xtrain, Xtest, ytrain, ytest


#%%

'''# =========================================================================================================================================================='''
'''#            FUNCTION TO CREATE DATASET FOR MACHINE LEARNING            '''
'''# =========================================================================================================================================================='''

def dataframe_creation(task_name, params, train_features, test_features, train_labels, test_labels):
    
    home_directory = os.getcwd()
    
    try:                                                                                       # Error handling for kernel interuption.
        
        df_directory = home_directory + '/dataframes'                                          # Capturing home directory for safekeeping 
        if os.path.isdir(df_directory):
            os.chdir(df_directory)                  
        else:                                                                                  # Create folder in directory to save all dataframes         
            os.mkdir(df_directory)
            os.chdir(df_directory)
       
        task_directory =  df_directory + '/' + str(task_name)   
        if os.path.isdir(task_directory):                                       
            os.chdir(task_directory)                  
        else:                                                                                  # Create task-specific folder in directory to save       
            os.mkdir(task_directory)                                                           # dataframe with features 
            os.chdir(task_directory)         
    
        print('\nStoring data for chosen preprocessing parameters in /dataframes/' + task_name) 
        print('[Training and test set are stacked - to be indexed at ratio of chosen split]')
    
        features = np.vstack((train_features, test_features))                                  # Create a dataframe by rejoining training and test data, which
        labels = np.hstack((train_labels, test_labels))                                        # can now be separated once more easily through indexing.
        labels = np.reshape(labels, (len(labels), 1))
        
        dataframe = pd.DataFrame(np.hstack((features, labels)))
        
        if params == None:
            file = str(task_name)                                                              # Create a filename to save the dataset if it does not exist.
        else:                                                                                  # File name incorporates any chosen preprocessing parameters
            file = str(task_name)+'__'+str(params)
        
        #print(file + '.csv')    
        #print(os.path.isfile(file+'.csv'))
        
        if not os.path.isfile(file + '.csv'):                                                  # Filename is specific to the parameters. 
            dataframe.to_csv(file+'.csv', index=True, header=True)
            dataframe.to_pickle(file+'.pkl')
            print('\n...new dataframe created.')
            
        else:
            print('Dataframe exists. Overriding...')
            dataframe.to_csv(file+'.csv', index=True, header=True)
            dataframe.to_pickle(file+'.pkl')
            print('\n...new dataframe created.')
        
        print('\nFilename: '+ str(file + '.csv'))
        print('\nReady for training of learning algorithm.')
        print("------------------------------------------------------------------")
         
        os.chdir(home_directory)                                                               # Returning to home directory after accessing labels 
    
    except KeyboardInterrupt:                                                               # if keyboard interrupt occurs
        os.chdir(home_directory)                                                            # Returning to home directory after accessing images
    
    return dataframe, file


#%%


'''# =========================================================================================================================================================='''
'''#  FUNCTION TO EXTRACT FEATURES FROM AN IMAGE VIA HISTOGRAM OF GRADIENTS '''
'''# =========================================================================================================================================================='''

def HOG_feature_extraction(image_list, cell_height, cell_width, equalise):
    
    from skimage.feature import hog
    import cv2
    
    #HOG_image_list = []
    all_image_features_list = []
    
    for i, image in enumerate(image_list):
            
        if equalise == True:                                                            # Contrast Limited Adaptive Histogram Equalisation.    
            CLAHE = cv2.createCLAHE(clipLimit=5, tileGridSize=(8,8))                    # method to amplify contras in images, resulting in 
            image = CLAHE.apply(image)                                                  # potentially stronger classifier
            
        #Can change 'Visualize' argument to True to also return HOG transformed images
        features = hog(image,
                       orientations = 8,
                       pixels_per_cell = (cell_height, cell_width),
                       cells_per_block = (1, 1),
                       visualize = False,
                       transform_sqrt = False,
                       multichannel = False)
        
        #HOG_image_list.append(HOG_image)
        all_image_features_list.append(features)
        
        #if (i > 0 and i % 1000 == 0) or (i == 1):                                      # Printing updates.   
            #print('Features extracted for', i, 'images')
    
    #HOG_image_array = np.asarray(HOG_image_list)
    all_image_features_array = np.asarray(all_image_features_list)
        
    return all_image_features_array
        

#%%

'''# =========================================================================================================================================================='''
'''#   FUNCTION TO CROP IMAGES GIVEN A CHOSEN START POSITION AND CROP SIZE '''
'''# =========================================================================================================================================================='''

def crop_images(images, x_coord, y_coord, crop_height, crop_width):
    
    cropped_images = []
    
    x = x_coord                                             # column index of array for top left corner of desired region of interest
    y = y_coord                                             # row index of array for top left corner of desired region of interest   
    h = crop_height                                         # height of deisred region of interest
    w = crop_width                                          # width of desired region of interest
    
    for i, img in enumerate(images):
              
        img = img[y:y+h, x:x+w]                                                         # Image array sliced for region dictates by 
                                                                                        # coordinates and dimensions 
        cropped_images.append(img)
        
        #if (i > 0 and i % 2000 == 0) or (i == 1):                                      # Printing updates.   
            #print(i, 'images cropped') 
    
    cropped_images = np.asarray(cropped_images)
    
    return cropped_images


#%%


'''# =========================================================================================================================================================='''
'''#     FUNCTION TO PLOT LEARNING CURVE OF MODEL USING CROSS VALIDATION   '''
'''# =========================================================================================================================================================='''

def plot_learning_curve(estimator, title, X, y, train_sizes, axes=None, ylim=None, cv=None, n_jobs=None):
    
    import matplotlib.pyplot as plt
    from sklearn.model_selection import learning_curve
    
    axes.set_title(title)

    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator, 
                                                                          X, y, 
                                                                          cv=cv, 
                                                                          n_jobs=n_jobs,
                                                                          train_sizes=train_sizes,
                                                                          return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    
    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes, train_scores_mean, 'x-', color="r",
                 label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'x-', color="g",
                 label="Cross-validation score")
    axes.legend(loc="best")

   
    return plt
