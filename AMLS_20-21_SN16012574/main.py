
'''# =========================================================================================================================================================='''
'''#     MAIN.PY SCRIPT TO PREPROCESS IMAGES, AND TRAIN + TEST CLASSIFICATION MODELS FOR EACH TASK   '''
'''# =========================================================================================================================================================='''

''' This module accesses several other modules in a nested manner. To do this, relative 
    paths are employed when changing directories within functions that are called. As such, 
    it is important that the module is run from ITS OWN directory. This way, the relative
    paths to access images and labels in their specifc folders will operate correctly.
    
    If MAIN.py is not being run from its own directory, please find the file and set your
    environment's current directory to the directory in which MAIN.py can be found.
    If you encounter any errors regarding file paths  when calling functions, please 
    return the current environment to the directory of MAIN.py.

    
    MAIN.py is split into 4 sections, one for each task - A1, A2, B1 and B2.
    Each section follows the same format:
        - Data preprocessing  (using images from celeb_a or cartoon_set)
        - Data saving (optional)
        - Data training and testing using classification model
    For the above procedure, the request type == 'train_test'
    The random_seed argument enables the user to pass an integer as the seed
    for an RNG which randomly splits the images into a training and test set 
    with sizes 80% and 20% of the total dataset respectively. 
    
    In addition, the user has the option to load images from the provided test sets
    (celeb_a_test or cartoon_set_test). In this case, the 'current' test set will be
    reassigning as the validation set, and the images from the true test dataset, 
    (for assessment) will be used as the test set. 
    For the above procedure, the request type == 'test_only'
    
    Following each task, the user also has the option to clear all variables from the 
    module's environment to free up memory.
     
    
    The results of training and testing each module are printed at the end of MAIN.py  '''

#import os, sys

#%%
''' OPTIONAL '''
# Please uncomment the lines of code below and enter the directory of MAIN.py.
# Run this cell if you encounter path errors when calling functions/interrupting kernel.

#import os
#DIRECTORY = r'(...insert directory here and remove elipses and brackets...)
#os.chdir(DIRECTORY)

import os
DIRECTORY = r'C:\Users\prave\OneDrive - University College London\SEM 1\Applied ML\CW\AMLS_assignment_kit\AMLS_20-21_SN12345678'
os.chdir(DIRECTORY)


#%%
'''# =========================================================================================================================================================='''
'''#     TASK A1: GENDER CLASSIFICATION     '''
'''# =========================================================================================================================================================='''

from A1.A1_code import A1_data_preprocessing, A1_data_saving, A1_learning_model

#%%
#Loading and preprocessing images for model
X_train_A1, X_test_A1, y_train_A1, y_test_A1 = A1_data_preprocessing(request_type = 'train_test', random_seed = 3)

#%%
''' OPTIONAL '''
#Function to save training and test data in a task-specific folder
A1_dataframe = A1_data_saving(X_train_A1, X_test_A1, y_train_A1, y_test_A1)

#%%
#Training learning algorithm and testing model
A1_train_accuracy, A1_test_accuracy = A1_learning_model(X_train_A1, y_train_A1,
                                                        X_test_A1, y_test_A1,
                                                        request_type = 'train_test')

#%%
''' OPTIONAL '''
#Renaming current test set and test accuracy as validation set and validation accuracy:
X_validation_A1 = X_test_A1.copy()
y_validation_A1 = y_test_A1.copy()
A1_validation_accuracy = A1_test_accuracy.copy()

#Loading and preprocessing test images for model. Reassigning variable name for new test set
X_test_A1, y_test_A1 = A1_data_preprocessing(request_type = 'test_only')

#Training learning algorithm and testing model on new test set
A1_train_accuracy, A1_validation_accuracy, A1_test_accuracy = A1_learning_model(X_train_A1, y_train_A1, 
                                                                                X_validation_A1, y_validation_A1, 
                                                                                X_test = X_test_A1, 
                                                                                y_test = y_test_A1,
                                                                                request_type = 'test_only')
#%%
''' OPTIONAL '''
# Clear all variables (except accuracy metrics) from current module to free up memory
del X_train_A1, X_test_A1, y_train_A1, y_test_A1, X_validation_A1, y_validation_A1
del A1_data_preprocessing, A1_data_saving, A1_learning_model





#%%
'''# =========================================================================================================================================================='''
'''#     TASK A2: EMOTION DETECTION            '''
'''# =========================================================================================================================================================='''

from A2.A2_code import A2_data_preprocessing, A2_data_saving, A2_learning_model

#%%
#Loading and preprocessing images for model
X_train_A2, X_test_A2, y_train_A2, y_test_A2 = A2_data_preprocessing(request_type = 'train_test', random_seed = 0)

#%%
''' OPTIONAL '''
#Function to save training and test data in a task-specific folder
A2_dataframe = A2_data_saving(X_train_A2, X_test_A2, y_train_A2, y_test_A2)

#%%
#Training learning algorithm and testing model
A2_train_accuracy, A2_test_accuracy = A2_learning_model(X_train_A2, y_train_A2,
                                                        X_test_A2, y_test_A2,
                                                        request_type = 'train_test')
#%%
''' OPTIONAL '''
#Renaming current test set and test accuracy as validation set and validation accuracy:
X_validation_A2 = X_test_A2.copy()
y_validation_A2 = y_test_A2.copy()
A2_validation_accuracy = A2_test_accuracy.copy()

#Loading and preprocessing test images for model. Reassigning variable name for new test set
X_test_A2, y_test_A2 = A2_data_preprocessing(request_type = 'test_only')

#Training learning algorithm and testing model on new test set
A2_train_accuracy, A2_validation_accuracy, A2_test_accuracy = A2_learning_model(X_train_A2, y_train_A2, 
                                                                                X_validation_A2, y_validation_A2, 
                                                                                X_test = X_test_A2, 
                                                                                y_test = y_test_A2,
                                                                                request_type = 'test_only')
#%%
''' OPTIONAL '''
# Clear all variables (except accuracy metrics) from current module to free up memory
del X_train_A2, X_test_A2, y_train_A2, y_test_A2, X_validation_A2, y_validation_A2
del A2_data_preprocessing, A2_data_saving, A2_learning_model





#%%
'''# =========================================================================================================================================================='''
'''#     TASK B1: FACE-SHAPE CLASSIFICATION     '''
'''# =========================================================================================================================================================='''

from B1.B1_code import B1_data_preprocessing, B1_data_saving, B1_learning_model

#%%
#Loading and preprocessing images for model
X_train_B1, X_test_B1, y_train_B1, y_test_B1 = B1_data_preprocessing(request_type = 'train_test', random_seed = 0)

#%%
''' OPTIONAL '''
#Function to save training and test data in a task-specific folder
B1_dataframe = B1_data_saving(X_train_B1, X_test_B1, y_train_B1, y_test_B1)

#%%
#Training learning algorithm and testing model
B1_train_accuracy, B1_test_accuracy = B1_learning_model(X_train_B1, y_train_B1,
                                                        X_test_B1, y_test_B1,
                                                        request_type = 'train_test')

#%%
''' OPTIONAL '''
#Renaming current test set and test accuracy as validation set and validation accuracy:
X_validation_B1 = X_test_B1.copy()
y_validation_B1 = y_test_B1.copy()
B1_validation_accuracy = B1_test_accuracy.copy()

#Loading and preprocessing test images for model. Reassigning variable name for new test set
X_test_B1, y_test_B1 = B1_data_preprocessing(request_type = 'test_only')

#Training learning algorithm and testing model on new test set
B1_train_accuracy, B1_validation_accuracy, B1_test_accuracy = B1_learning_model(X_train_B1, y_train_B1, 
                                                                                X_validation_B1, y_validation_B1, 
                                                                                X_test = X_test_B1, 
                                                                                y_test = y_test_B1,
                                                                                request_type = 'test_only')
#%%
''' OPTIONAL '''
# Clear all variables (except accuracy metrics) from current module to free up memory
del X_train_B1, X_test_B1, y_train_B1, y_test_B1, X_validation_B1, y_validation_B1
del B1_data_preprocessing, B1_data_saving, B1_learning_model






#%%
'''# =========================================================================================================================================================='''
'''#     TASK B2: EYE-COLOUR CLASSIFICATION     '''
'''# =========================================================================================================================================================='''

from B2.B2_code import B2_data_preprocessing, B2_data_saving, B2_learning_model

#%%
#Loading and preprocessing images for model
X_train_B2, X_test_B2, y_train_B2, y_test_B2 = B2_data_preprocessing(request_type = 'train_test', random_seed = 0)

#%%
''' OPTIONAL '''
#Function to save training and test data in a task-specific folder
B2_dataframe = B2_data_saving(X_train_B2, X_test_B2, y_train_B2, y_test_B2)

#%%
#Training learning algorithm and testing model
B2_train_accuracy, B2_test_accuracy = B2_learning_model(X_train_B2, y_train_B2,
                                                        X_test_B2, y_test_B2,
                                                        request_type = 'train_test')

#%%
''' OPTIONAL '''
#Renaming current test set and test accuracy as validation set and validation accuracy:
X_validation_B2 = X_test_B2.copy()
y_validation_B2 = y_test_B2.copy()
B2_validation_accuracy = B2_test_accuracy.copy()

#Loading and preprocessing test images for model. Reassigning variable name for new test set
X_test_B2, y_test_B2 = B2_data_preprocessing(request_type = 'test_only')

#Training learning algorithm and testing model on new test set
B2_train_accuracy, B2_validation_accuracy, B2_test_accuracy = B2_learning_model(X_train_B2, y_train_B2, 
                                                                                X_validation_B2, y_validation_B2, 
                                                                                X_test = X_test_B2, 
                                                                                y_test = y_test_B2,
                                                                                request_type = 'test_only')
#%%
''' OPTIONAL '''
# Clear all variables (except accuracy metrics) from current module to free up memory
del X_train_B2, X_test_B2, y_train_B2, y_test_B2, X_validation_B2, y_validation_B2
del B2_data_preprocessing, B2_data_saving, B2_learning_model









#%%
'''# =========================================================================================================================================================='''
'''#            PRINTING OUT RESULTS            '''
'''# =========================================================================================================================================================='''

print('TA2: {}, {};   TA2: {}, {};   TB2: {}, {};   TB2: {}, {};'.format(A1_train_accuracy, A1_test_accuracy,
                                                                         A2_train_accuracy, A2_test_accuracy,
                                                                         B1_train_accuracy, B1_test_accuracy,
                                                                         B2_train_accuracy, B2_test_accuracy))

