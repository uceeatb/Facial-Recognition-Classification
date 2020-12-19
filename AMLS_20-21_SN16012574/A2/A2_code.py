
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

def A2_data_preprocessing(request_type, random_seed = None): # random_seed affects splitting of training and test set
                                                             # Default is None type, when request_type is "test_only"
    
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
    
    from common_functions import load_images, load_labels, split_data, HOG_feature_extraction

    task = 'A2'

    start_time = time.time()
    
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    

    ################################################################################################################################################################
    
    
    '''# =========================================================================================================================================================='''
    '''#      FUNCTION TO LOAD IMAGES, LABELS AND SPLIT DATA FOR TASK B2      '''
    '''# =========================================================================================================================================================='''
    
    def A2_load_data(task_name, image_set_request, random_int = 0):                                                  
            
        image_list = load_images(task_name, image_set_request)                          # Loading A2 images from celeba directory.
        
        label_list = load_labels(task_name, image_set_request)
        
        if image_set_request == 'train_test':
            Xtrain_img, Xtest_img, ytrain, ytest = split_data(image_list,                               # Data split before preprocess for unbiased parameter evaluation.     
                                                              label_list,                               # Training set: 80%.   Test set: 20%.
                                                              random_int = random_int) 
            
            return Xtrain_img, Xtest_img, ytrain, ytest   
        
        elif image_set_request == 'test_only':
            Xtest_img = image_list
            ytest = label_list
            
            return Xtest_img, ytest                                                          
                                                                                               
       
        return Xtrain_img, Xtest_img, ytrain, ytest


    ###################################
    #    CALLING TO FUNCTION ABOVE    
    ###################################
     
    print('\nLoading images from A2 data directory...\n')
    
    if request_type == 'train_test':
        X_train_images_A2, X_test_images_A2, y_train, y_test = A2_load_data(task_name = task,
                                                                            image_set_request = request_type,
                                                                            random_int = random_seed)
    elif request_type == 'test_only':
        X_test_images_A2, y_test = A2_load_data(task_name = task,
                                                image_set_request = request_type,
                                                random_int = random_seed)    
    
    print("------------------------------------------------------------------")
    
    
    ################################################################################################################################################################
    
    
    ###################################
    #    PREPROCESSING PARAMETERS    
    ###################################
    
    A2_parameters = {
                    'Cell size': [[8, 8], [16, 16]]
                    }

    A2_chosen_params = [8, 8]           # chosen cell size for HOG feature extraction method
    
    
    
    '''# =========================================================================================================================================================='''
    '''#               FUNCTION TO EXTRACT FEATURES FOR A2 VIA HOG             '''
    '''# =========================================================================================================================================================='''
      
    def A2_feature_extraction(image_list, cell_height, cell_width):
    
        features_list = HOG_feature_extraction(image_list,
                                               cell_height,
                                               cell_width,
                                               equalise = True)                 # CLAHE has not yet been performed on images.
                                                                                # equalise = True to apply CLAHE
        return features_list
    

    ###################################
    #    CALLING TO FUNCTION ABOVE    
    ###################################
    
    print('\nApplying equalisation to images and extracting features via Histogram of Gradients...')
    
    if request_type == 'train_test':
        X_train = A2_feature_extraction(image_list = X_train_images_A2, 
                                       cell_height = A2_chosen_params[0], 
                                       cell_width = A2_chosen_params[1]) 
    
        X_test = A2_feature_extraction(image_list = X_test_images_A2,
                                      cell_height = A2_chosen_params[0],
                                      cell_width = A2_chosen_params[1]) 
    
    elif request_type == 'test_only':
        X_test = A2_feature_extraction(image_list = X_test_images_A2,
                                      cell_height = A2_chosen_params[0],
                                      cell_width = A2_chosen_params[1]) 
    print('...features extracted.')
    print("------------------------------------------------------------------\n")
    
    
    
    
    
    total_time = round((time.time() - start_time), 1)
    
    print('Total time taken', total_time, 'seconds.')
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

def A2_data_saving(X_train, X_test, y_train, y_test):
    
    from common_functions import dataframe_creation
    
    task = 'A2'
    A2_chosen_params = 'Cell_size_8x8'
    
    saved_dataframe = dataframe_creation(task, A2_chosen_params, X_train, X_test, y_train, y_test)

    return saved_dataframe




#%%

'''# =========================================================================================================================================================='''
'''#  FUNCTION TO TRAIN AND TEST ML MODEL ON DATA EXTRACTED FOR GIVEN TASK '''
'''# =========================================================================================================================================================='''

def A2_learning_model(X_train, y_train, X_validation, y_validation, request_type, X_test=None, y_test=None):
        
    from sklearn.svm import SVC
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, plot_confusion_matrix
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
    pca = PCA(n_components = 0.70)                                                   # Determined from cross-validation procedures
    
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_dummy_test_pca = pca.transform(X_dummy_test_scaled)
    if request_type == 'test_only':
        X_test_pca = pca.transform(X_test_scaled)
    
    total_components = X_train_scaled.shape[1]
    principal_components = X_train_pca.shape[1]
    
    print('...PCA applied.', principal_components,'out of',
           total_components , 'components returned (70% explained variance).')
    
    
    
    print('\nSVC model chosen for data training. Optimal hyper parameters determined from cross validation.')
    print('Hyperparameters: linear kernel, C = 0.001.')
    
    svm = SVC(C = 0.001, kernel = 'linear')
    
    
    
    print('\nTraining model on training data...')
    ##########################################################################
    svm.fit(X_train_pca, y_train)
    ##########################################################################
    print('...training complete.')
    print("------------------------------------------------------------------")
    
   
    
    print('\nPredicting accuracy of model...')
    
    # Training accuracy    
    y_train_pred = svm.predict(X_train_pca)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print('\nTRAINING ACCURACY for TASK A2       = ', train_accuracy)
    
    # Test accuracy (dummy test)    
    y_dummy_test_pred = svm.predict(X_dummy_test_pca)
    dummy_test_accuracy = accuracy_score(y_dummy_test, y_dummy_test_pred)
    if request_type == 'test_only':
        print('VALIDATION ACCURACY for TASK A2     = ', dummy_test_accuracy)
    else:
        print('TEST ACCURACY for TASK A2           = ', dummy_test_accuracy)
         
    # Test accuracy (test, if neccesary)   
    if request_type == 'test_only':
        y_test_pred = svm.predict(X_test_pca)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print('TEST ACCURACY for TASK A2           = ', test_accuracy)    
    print("------------------------------------------------------------------")
    
    

    
    total_time = round((time.time() - start_time), 1)
    
    print('\nTotal time taken', total_time, 'seconds.')
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    
    
    if request_type == 'train_test':
        return train_accuracy, dummy_test_accuracy                                          # FUNCTION RETURNS DIFFERENT NUMBER OF VALUES
                                                                                            # DEPENDING ON REQUEST TYPE 
    
    elif request_type == 'test_only':
        return train_accuracy, dummy_test_accuracy, test_accuracy

    