
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

def A1_data_preprocessing(request_type, random_seed = None): # random_seed affects splitting of training and test set
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
    
    from common_functions import load_images, load_labels, split_data
    from A1.A1_helper_functions import haar_face_detector, split_image, DCT_2D, create_huffman, quantisation, encoding

    task = 'A1'
       
    start_time = time.time()
    
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    
       
    ################################################################################################################################################################
    
 
    '''# =========================================================================================================================================================='''
    '''#       FUNCTION TO LOAD IMAGES, LABELS AND SPLIT DATA FOR TASK A1     1 '''
    '''# =========================================================================================================================================================='''
    
    def A1_load_data(task_name, image_set_request, random_int = 0):                                                  
            
        image_list = load_images(task_name, image_set_request)                                 # Loading A1 images from celeba directory.
        
        label_list = load_labels(task_name, image_set_request)
        
        if image_set_request == 'train_test':
            Xtrain_img, Xtest_img, ytrain, ytest = split_data(image_list,                      # Data split before preprocess for unbiased parameter evaluation.     
                                                              label_list,                      # Training set: 80%.   Test set: 20%.
                                                              random_int = random_int) 
            
            return Xtrain_img, Xtest_img, ytrain, ytest   
        
        elif image_set_request == 'test_only':
            Xtest_img = image_list
            ytest = label_list
            
            return Xtest_img, ytest   
    
    
    ###################################
    #    CALLING TO FUNCTION ABOVE    
    ###################################
    
    print('\nLoading images from A1 data directory...\n')
    
    if request_type == 'train_test':
        X_train_images_A1, X_test_images_A1, y_train, y_test = A1_load_data(task_name = task,
                                                                            image_set_request = request_type,
                                                                            random_int = random_seed)
    elif request_type == 'test_only':
        X_test_images_A1, y_test = A1_load_data(task_name = task,
                                                image_set_request = request_type,
                                                random_int = random_seed)    
    
    print("------------------------------------------------------------------")
    
    
    
    ################################################################################################################################################################
    
   
    
    '''# =========================================================================================================================================================='''
    '''#      FUNCTION TO PREPROCESS IMAGES FOR A1 VIA HAAR FACE DETECTION     '''
    '''# =========================================================================================================================================================='''
    
    def A1_image_preprocessing(image_list, resize_number):                                                       
           
        faces, fails = haar_face_detector(image_list, 
                                          Min_neighbours = 2, 
                                          scaleFactor = 1.05, 
                                          resize_value = resize_number)                                    # Haar (face) classifcation using optimal parameters.
            
        return faces, fails  
    
    
    ###################################
    #    CALLING TO FUNCTION ABOVE    
    ###################################

    print('\nDetecting faces on images using Haar face detection classifier...\n')
    
    if request_type == 'train_test':
        X_train_faces_A1, training_fails = A1_image_preprocessing(image_list = X_train_images_A1,       # 104 chosen based on mean output face size.)
                                                                  resize_number = 104)                  # see A1_Haar_parameter_analysis.py for further details.        
        print('Faces found on training images.')
        training_examples = len(y_train)
        print('Haar Face detection failed for', training_fails, 'out of', training_examples, 'images in training set')

        
        X_test_faces_A1, test_fails = A1_image_preprocessing(image_list = X_test_images_A1, 
                                                             resize_number = 104)
        print('Faces found on test images.')
        print("------------------------------------------------------------------")
        
    
    elif request_type == 'test_only':
        X_test_faces_A1, test_fails = A1_image_preprocessing(image_list = X_test_images_A1, 
                                                             resize_number = 104)
        print('Faces found on test images.')
        print("------------------------------------------------------------------")
        
      

    ################################################################################################################################################################
        
    ###################################
    #    PREPROCESSING PARAMETERS    
    ###################################
    
    A1_parameters = {
                    'Huffman_number': [15, 21, 28, 64]
                    }

    A1_chosen_params = [15]
    

    '''# =========================================================================================================================================================='''
    '''#    FUNCTION TO EXTRACT FEATURES FOR A1 VIA DCT AND HUFFMAN ENCODING   '''
    '''# =========================================================================================================================================================='''
    
    def A1_feature_extraction(faces, resize_number, reducing_number):                                          
            
        import numpy as np
                
        no_subimages = int((resize_number / 8)**2)                                             # Haar classifer found median face ouput of 105 pixels.
        
        all_image_features = np.zeros((faces.shape[0], reducing_number*no_subimages))          # Images are resized to closest multiple of 8, i.e. 104x104 
        #reconstructed_faces = np.zeros((faces.shape[0], faces.shape[1], faces.shape[2]))      # since DCT function processes 8x8 matrices for JPEG images.
        
        ''' LOOPING OVER EACH IMAGE... '''
        for i, image in enumerate(faces):
    
            subimages = split_image(image, [8,8])                                              # 8x8 subimage tiles created for Discrete Cosine Transform.
            DCT_subimages = DCT_2D(subimages)
            
            H_matrix = create_huffman(reducing_number)                                         # Upper-left trianglar Huffman matrix to quantise DCT matrices.
            compressed_subimages, quant_matrix = quantisation(100, DCT_subimages)              # Quantisation using Q_factor = 100 (... uncompressed!)
            ''' NB: Q_factor of 100 chosen as final parameter for compression...
                I.e. compression NOT made on DCT coefficients using traditional JPEG method.
                Instead, ONLY Huffman encoding (slicing) used as compression method. '''
            
            features, H_subimages = encoding(compressed_subimages, H_matrix)                   # Encoding feature vector for image by mapping with huffman matrix,
                                                                                               # i.e. slicing subimage DCT arrays for only the largest values. 
            
            #H_IDCT = IDCT_2D(H_subimages * quant_matrix)                                      # Each tile decompressed by multiplying by quant_matrix. Then, 
            #H_reconstructed = unsplit_tiles.(H_IDCT,[resize_number,resize_number])             # Inverse DCT performed and tile recompiled into reconstructed image.
            
            '''cv2.imwrite('compressed_'+str(i)+'.jpg', H_reconstructed)'''                    # (Option to save reconstructed images to local directory).
            
            all_image_features[i] = features                                                   # Image features and reconstruction added to 
            #reconstructed_faces[i] = H_reconstructed                                          # their corresponding arrays.
                        
            #if (i > 0 and i % 1000 == 0) or (i == 1):                                         # Printing updates.
                #print(i, 'images preprocessed')
                               
        #return reconstructed_faces, all_image_features                                        # (Option to return reconstructed images used during testing.) 
        return all_image_features
    
    
    ###################################
    #    CALLING TO FUNCTION ABOVE    
    ###################################
    
    print('\nExtracting features from images...\n')
    
    if request_type == 'train_test':
        X_train = A1_feature_extraction(X_train_faces_A1, 
                                        resize_number = 104, 
                                        reducing_number = A1_chosen_params[0])          # Applying huffman 15-matrix as compressor
        print('Features extracted from training set.')
        
        X_test = A1_feature_extraction(X_test_faces_A1, 
                                       resize_number = 104, 
                                       reducing_number = A1_chosen_params[0])
        print('Features extracted from test set.')    

    
    
    elif request_type == 'test_only':
        X_test = A1_feature_extraction(X_test_faces_A1, 
                                       resize_number = 104, 
                                       reducing_number = A1_chosen_params[0])
        print('Features extracted from test set.')    

    
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

def A1_data_saving(X_train, X_test, y_train, y_test):
    
    from common_functions import dataframe_creation
    
    task = 'A1'
    A1_chosen_params = 'Huffman_15'
    
    saved_dataframe = dataframe_creation(task, A1_chosen_params, X_train, X_test, y_train, y_test)

    return saved_dataframe




#%%

'''# =========================================================================================================================================================='''
'''#  FUNCTION TO TRAIN AND TEST ML MODEL ON DATA EXTRACTED FOR GIVEN TASK '''
'''# =========================================================================================================================================================='''

def A1_learning_model(X_train, y_train, X_validation, y_validation, request_type, X_test=None, y_test=None):
        
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
    #pca = PCA(n_components = X_train_scaled.shape[1])
    #pca = PCA(n_components = 0.875)                                                   # Determined from cross-validation procedures
    pca = PCA(n_components = 0.7)
    
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_dummy_test_pca = pca.transform(X_dummy_test_scaled)
    if request_type == 'test_only':
        X_test_pca = pca.transform(X_test_scaled)
    
    total_components = X_train_scaled.shape[1]
    principal_components = X_train_pca.shape[1]
    
    print('...PCA applied.', principal_components,'out of',
           total_components , 'components returned (70% explained variance).')
    
    
    
    print('\nSVC model chosen for data training. Optimal hyper parameters determined from cross validation.')
    print('Hyperparameters: rbf kernel, C = 10, gamma = 0.001.')
    
    #svm = SVC(C = 0.0003, kernel = 'linear', tol = 1e-3)
    svm = SVC(C = 10, gamma = 0.001, kernel = 'rbf', tol = 1e-3)                     # Other choice for SVM parameters (both gave similar accuracy)  
    
    
    
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
    print('\nTRAINING ACCURACY for TASK A1       = ', train_accuracy)
    
    # Test accuracy (dummy test)    
    y_dummy_test_pred = svm.predict(X_dummy_test_pca)
    dummy_test_accuracy = accuracy_score(y_dummy_test, y_dummy_test_pred)
    if request_type == 'test_only':
        print('VALIDATION ACCURACY for TASK A1     = ', dummy_test_accuracy)
    else:
        print('TEST ACCURACY for TASK A1           = ', dummy_test_accuracy)
        
    # Test accuracy (test, if neccesary)    
    if request_type == 'test_only':
        y_test_pred = svm.predict(X_test_pca)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print('TEST ACCURACY for TASK A1           = ', test_accuracy)    
    print("------------------------------------------------------------------")
    
    
        
    total_time = round((time.time() - start_time), 1)
    
    print('\nTotal time taken', total_time, 'seconds.')
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    
    
    if request_type == 'train_test':
        return train_accuracy, dummy_test_accuracy                                          # FUNCTION RETURNS DIFFERENT NUMBER OF VALUES
                                                                                            # DEPENDING ON REQUEST TYPE 
    
    elif request_type == 'test_only':
        return train_accuracy, dummy_test_accuracy, test_accuracy
    


#%%



    