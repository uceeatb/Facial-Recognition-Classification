
''' FOREWORD '''

''' This module comprisises three fuctions in the same format for all tasks:
        - Data preprocessing function
        - Data saving function (helpful for validation, but not necessary)
        - Model training & testing function
    
    The module access two key modules:
        - common_functions.py module (functions used across more than one task)
        - A2_functions.py (task-specific functions that reduce length of this script) 
        
    Comments have been provided for the user to understand the high-level processes     
    involved in each of the three functions. 
    For more detailed understanding of the functions access from each of the two 
    modules, please visit the .py files for the respective module. Here, further 
    low-level understanding of each individual function is provided.              '''
                                                                            
        
'''# =========================================================================================================================================================='''
'''# =========================================================================================================================================================='''
'''#     MAIN PREPROCESSING FUNCTION TO OBTAIN FEATURES FROM RAW IMAGES    '''
'''# =========================================================================================================================================================='''
'''# =========================================================================================================================================================='''

def B2_data_preprocessing(request_type, random_seed = None): # random_seed affects splitting of training and test set
                                                             # Default is None type, when request_type is "test_only"): # random_seed affects splitting of training and test set
    
    ###########################################################################
    ###########################################################################
    #                                                                         #
    #  Inner functions for: image loading, preprocessing & feature extraction #
    #  Contains functionality to load training and test set or test set alone #
    #  'request_type' argument determines this.                               #
    #  Former, request_type = 'train_test'. Latter, request_type = 'test_only'#
    #  Several if statments throughout the function depedning on request_type #
    #                                                                         #
    ###########################################################################
    ###########################################################################
    
    import time
    import numpy as np

    from common_functions import load_images, load_labels, split_data, crop_images      
    from B2.B2_helper_functions import B2_resize_images, B2_flatten_images
    
    task = 'B2'
          
    start_time = time.time()
    
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    
     
    ################################################################################################################################################################
    
    
    '''# =========================================================================================================================================================='''
    '''#       FUNCTION TO LOAD IMAGES, LABELS AND SPLIT DATA FOR TASK B2      '''
    '''# =========================================================================================================================================================='''
    
    def B2_load_data(task_name, image_set_request):                                                  
    
        image_list = load_images(task_name, image_set_request)                           # Loading B2 images from cartoon_set directory.
    
        return image_list
    

    ###################################
    #    CALLING TO FUNCTION ABOVE    
    ###################################

    print('\nLoading images from B2 data directory...')

    B2_images = B2_load_data(task_name = task,
                             image_set_request = request_type)
    print("------------------------------------------------------------------")
    
    
    ################################################################################################################################################################
    

        
    '''# =========================================================================================================================================================='''
    '''#  FUNCTION TO PREPROCESS B2 IMAGES BY CROPPING FOR REGION OF INTEREST '''
    '''# =========================================================================================================================================================='''
    
    def B2_image_preprocessing(image_list):
       
        x_corner = 175
        y_corner = 235
        eye_height = 50
        eye_width = 60
            
        #print('Cropping for left eye of each image...\n')
        left_eye_regions = crop_images(image_list,
                                       x_corner,
                                       y_corner,
                                       eye_height, 
                                       eye_width)

        resized_image_list = B2_resize_images(left_eye_regions)    
        
        preprocessed_image_list = np.asarray(resized_image_list)
        
        return preprocessed_image_list
    
    
    ###################################
    #    CALLING TO FUNCTION ABOVE    
    ###################################

    print('\nCropping images for region of interest (left eye) and resizing...')
    B2_images_processed = B2_image_preprocessing(image_list = B2_images)

    
    ################################################################################################################################################################
    
    
    
    '''# =========================================================================================================================================================='''
    '''#         FUNCTION TO EXTRACT FEATURE VECTORS AND SPLIT DATASET         '''
    '''# =========================================================================================================================================================='''
    
    def B2_feature_extraction(task_name, image_list, image_set_request, random_int = 0):
        
        print('Flattening eye-images into feature vectors...') 
        flattened_image_list = B2_flatten_images(image_list)    
        
        print('Loading labels for each image...')
        label_list = load_labels(task_name, image_set_request)
        
        if image_set_request == 'train_test':
            print('Splitting flattened eye-images into training and test sets......')
            Xtrain, Xtest, ytrain, ytest = split_data(flattened_image_list,
                                                      label_list,
                                                      random_int = random_int)
            return Xtrain, Xtest, ytrain, ytest
            

        elif image_set_request == 'test_only':
            Xtest = flattened_image_list
            ytest = label_list
            
            return Xtest, ytest
        
        print('... Images flattened and labels loaded.')
        
    
    ###################################
    #    CALLING TO FUNCTION ABOVE    
    ###################################
    
    if request_type == 'train_test':
        X_train, X_test, y_train, y_test = B2_feature_extraction(task_name = task,
                                                                 image_list = B2_images_processed,
                                                                 image_set_request = request_type,
                                                                 random_int = random_seed)
    elif request_type == 'test_only':
        X_test, y_test = B2_feature_extraction(task_name = task,
                                               image_list = B2_images_processed,
                                               image_set_request = request_type)
        
        
    print("------------------------------------------------------------------")
    
    
    total_time = round((time.time() - start_time), 1)
    
    print('\nTotal time taken', total_time, 'seconds.')
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    
    if request_type == 'train_test':
        return X_train, X_test, y_train, y_test                                             # FUNCTION RETURNS DIFFERENT NUMBER OF VALUES
                                                                                            # DEPENDING ON REQUEST TYPE 
    elif request_type == 'test_only':
        return X_test, y_test




#%%

###################################
#           OPTIONAL    
###################################

'''# =========================================================================================================================================================='''
'''#     FUNCTION TO SAVE TRAINING AND TEST DATA INTO A DATAFRAME FOR ML   '''
'''# =========================================================================================================================================================='''

def B2_data_saving(X_train, X_test, y_train, y_test):
    
    from common_functions import dataframe_creation
    
    task = 'B2'
    B2_chosen_params = None              # 'None' assigned since no parameters were used in preprocessing
    
    saved_dataframe = dataframe_creation(task, B2_chosen_params, X_train, X_test, y_train, y_test)     

    return saved_dataframe




#%%

'''# =========================================================================================================================================================='''
'''#  FUNCTION TO TRAIN AND TEST ML MODEL ON DATA EXTRACTED FOR GIVEN TASK '''
'''# =========================================================================================================================================================='''

def B2_learning_model(X_train, y_train, X_validation, y_validation, request_type, X_test=None, y_test=None):
        
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, plot_confusion_matrix
    from sklearn.decomposition import PCA
    import time
    
    ''' The following function operates differently depending on the 'request type'',
    i.e. whether:
        - OPTION 1: 'request type == train_test'
                    user wishes to use one image dataset to train and test the model
                    I.e, data training AND TESTING is done using the original cartoon_set 
                    or celeba datasets ALONE.
        
        - OPTION 2: 'request type == test_only'
                    user wishes to use one image dataset to train and validate the model
                    and another image dataset to test the model. In particular, this enables 
                    the user to:
                       * train AND VALIDATE the model using cartoon_set or celeba dataset.  
                       * TEST the model using cartoon_set_test or celeba_test images 
                         after they have been preprocessed and features have been extracted.
    
    A key thing worth mentioning is that two of the function arguments, X_test and y_test,
    are None objects by default. This facilitates both options, wherein:
        - OPTION 1: requires nothing to be passed.
        - OPTION 2: requires array for Xtest and ytest to be passed.
    
    Throughout the function the choice of request_type affects the operations that are 
    completed and the outputs that are returned.
    
    For OPTION 1, the 'test data' is passed as X_validation, y_validation
    Since the validation data is 'tested upon', the arguments have been assigned
    to variables as below.                                                                   '''     
    
    X_dummy_test = X_validation
    y_dummy_test = y_validation
    
    
    
    
    start_time = time.time()    
    
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    
    
    if request_type == 'train_test':
        print('\nTraining and testing model using training and test set')

    elif request_type == 'test_only':
        print('\nTraining and testing model using training, validation AND test set')
        
    
    print('\nScaling features for equal contribution to model')
    scaler = StandardScaler().fit(X_train)
    
    X_train_scaled = scaler.transform(X_train)
    X_dummy_test_scaled = scaler.transform(X_dummy_test)
    if request_type == 'test_only':
        X_test_scaled = scaler.transform(X_test)
    
    print('...data scaled.')
    
    
    
    print('\nApplying PCA...')
    pca = PCA(n_components = 0.98)                                                   # Determined from cross-validation procedures
    
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_dummy_test_pca = pca.transform(X_dummy_test_scaled)
    if request_type == 'test_only':
        X_test_pca = pca.transform(X_test_scaled)
    
    total_components = X_train_scaled.shape[1]
    principal_components = X_train_pca.shape[1]
    
    print('...PCA applied.', principal_components,'out of',
           total_components , 'components returned (98% explained variance).')    
    
    
    
    
    print('\nLogistic Regression model chosen for data training. Optimal hyperparameters determined from cross validation.')
    print('Hyperparameters: L2 Regularisation, C = 4.')
    LogReg = LogisticRegression(C = 4, tol = 1e-5, max_iter = 50000, multi_class = 'ovr')
    
    
    
    print('\nTraining model on training data...')
    ##########################################################################
    LogReg.fit(X_train_pca, y_train)
    ##########################################################################
    print('...training complete.')
    print("------------------------------------------------------------------")
    
   
    
    print('\nPredicting accuracy of model...')
    
    # Training accuracy
    y_train_pred = LogReg.predict(X_train_pca)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print('\nTRAINING ACCURACY for TASK B2       = ', train_accuracy)
    
    # Test accuracy (dummy test)
    y_dummy_test_pred = LogReg.predict(X_dummy_test_pca)
    dummy_test_accuracy = accuracy_score(y_dummy_test, y_dummy_test_pred)
    if request_type == 'test_only':
        print('VALIDATION ACCURACY for TASK B2     = ', dummy_test_accuracy)
    else:
        print('TEST ACCURACY for TASK B2           = ', dummy_test_accuracy)
    
    # Test accuracy (test, if neccesary)    
    if request_type == 'test_only':
        y_test_pred = LogReg.predict(X_test_pca)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print('TEST ACCURACY for TASK B2           = ', test_accuracy)    
    print("------------------------------------------------------------------")
    
    
    
    total_time = round((time.time() - start_time), 1)
    
    print('\nTotal time taken', total_time, 'seconds.')
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    
    
    if request_type == 'train_test':
        return train_accuracy, dummy_test_accuracy                                          # FUNCTION RETURNS DIFFERENT NUMBER OF VALUES
                                                                                            # DEPENDING ON REQUEST TYPE 
    
    elif request_type == 'test_only':
        return train_accuracy, dummy_test_accuracy, test_accuracy
