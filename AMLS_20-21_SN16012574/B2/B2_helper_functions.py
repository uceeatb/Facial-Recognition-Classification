''' This module comprises a low-level function that was utilised for Task A2.
    Some functions are connected to each other, whilst others
    are not. Detailed commnets are shown to provide reader with low-level
    understanding of pipeline of each function. '''

#%%
'''# =========================================================================================================================================================='''
'''#        FUNCTION TO RESIZE B2 EYES TO REDUCE FEATURE VECTOR SIZE      '''
'''# =========================================================================================================================================================='''

def B2_resize_images(image_list):
    
    import numpy as np
    from PIL import Image    
        
    resized_image_list = []
    
    '''LOOPING OVER ALL IMAGES'''
    for i, image in enumerate(image_list):
            
        rgba_image = Image.fromarray(image)                             
        rgb_image = rgba_image.convert('RGB')                   # Feature dimensionality was further reduced by
                                                                # removing alpha component of images following PCA analysis                 
        resized_image = rgb_image.copy()
        resized_image.thumbnail((30, 25), Image.ANTIALIAS)      # .thumbnail function with ANTIALIASING method used to 
                                                                # resize images to maintain aspect ratio and minimise info loss
        rgb_array = np.asarray(resized_image) 
               
        resized_image_list += [rgb_array]
        
    resized_image_list = np.asarray(resized_image_list)         # converting list back into array.
        
    return resized_image_list

#%%
'''# =========================================================================================================================================================='''
'''#      FUNCTION TO FLATTEN N-D IMAGE ARRAYS INTO 1-D FEATURE VECTORS    '''
'''# =========================================================================================================================================================='''

def B2_flatten_images(images):
    
    import numpy as np
    
    flattened_images = []
    
    for i, img in enumerate(images):
              
        img = np.ravel(img)                                      # converting array of dimensions > 1 to a 1D vector
        
        flattened_images.append(img)
        
        #if (i > 0 and i % 2000 == 0) or (i == 1):               # Printing updates.   
            #print(i, 'images flattened') 
    
    flattened_images = np.asarray(flattened_images)
    
    return flattened_images