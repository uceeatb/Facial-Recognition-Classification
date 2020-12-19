''' This module comprises several low-level functions that were utilised for Task A1.
    Some functions are connected to each other, whilst others
    are not. Detailed commnets are shown to provide reader with low-level
    understanding of pipeline of each function. '''

import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image
import cv2


'''# =========================================================================================================================================================='''
'''#  FUNCTION TO DETECT FACES FROM GREYSCALE IMAGES USING HAAR CLASSIFIER '''
'''# =========================================================================================================================================================='''

def haar_face_detector(image_collection, Min_neighbours, scaleFactor, resize_value):
    
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
          
    collection_copy = image_collection.copy()
    dimensions =[collection_copy.shape[1], collection_copy.shape[2]]
    
    faces_dataset = np.zeros((collection_copy.shape[0], resize_value, resize_value))     # Create two lists to contain faces output by the classifier.
    
    unsuccessful = 0                                                                     # Initialise a counter for failed face detection  .
    
    ''' LOOPING OVER EACH IMAGE... '''
    for i in range(len(collection_copy)):    

        grey_image = collection_copy[i]
        
        faces_rect = cascade.detectMultiScale(grey_image, scaleFactor, Min_neighbours)             #     CALL THE CLASSSIFIER.
        
        # NO FACE DETECTED
        if len(faces_rect) == 0:
            
            offset = int((max(dimensions) - min(dimensions)) / 2)                        # If no face is detected, safest approach is to crop out   
            ROI = grey_image[0:min(dimensions), 0+offset:min(dimensions)+offset]         # middle square of image (likeliest to contain face).
                                                                                                    
            ROI = Image.fromarray(ROI)                                                   # Convert array to image and resize it. Resize_value = 104. 
            resized_ROI = ROI.resize((resize_value,resize_value), Image.ANTIALIAS)       # 104 chosen because it is close to median face dimensions                                                   
            resized_ROI = np.array(resized_ROI)                                          # (105) and multiple of 8 for DCT feature extraction process.
                       
            '''cv2.imwrite(str(i)+'.jpg', resized_ROI)'''
            faces_dataset[i] = resized_ROI                                               # Add to lists of faces.
            unsuccessful += 1                                                            # Increase counter for failed detection.
        
        # FACE(S) DETECTED 
        else:
            
            faces_regions = np.asarray(faces_rect).T                                     # Haar may incorrectly detect multiple faces when
            faces_dimensions = faces_regions[2]                                          # each image only contains one person. Safest approach   
            largest_face_index = np.argmax(faces_dimensions, axis = 0)                   # Is to output ONLY the largest face from the classifier.   
            best_face_rect = faces_rect[largest_face_index]                              # These lines of code find this 'largest face'.
                 
            x, y, h, w = best_face_rect

            ROI = grey_image[y:y+h, x:x+w]                                               # Crop out the region containing face.
            
            ROI = Image.fromarray(ROI)                                                   # Convert array to image and resize it. 'ANTIALIASING' achieved
            resized_ROI = ROI.resize((resize_value, resize_value), Image.ANTIALIAS)      # best results upon image visualisation. 'Resize_value' = 104 
            resized_ROI = np.array(resized_ROI)                                          # chosen as it is close to median of all face dimensions (105) 
                                                                                         # and multiple of 8, for DCT feature extraction process (8x8 tiles).
            '''cv2.imwrite(str(i)+'.jpg', resized_ROI)'''
            faces_dataset[i] = resized_ROI                                               # Add to list of faces.
                
        #if (i > 0 and i % 1000 == 0) or (i == 1):                                      # Printing updates.   
            #print(i, 'faces detected')
            
        del faces_rect
    
    #print("ALL FACES DETECTED\n")
    #print('Face detection failed for', unsuccessful, 'out of', len(faces_dataset), 'examples')

    return faces_dataset, unsuccessful

#%%

'''# =========================================================================================================================================================='''
'''#        FUNCTION TO SPLIT IMAGE INTO TILES OF CHOSEN DIMENSIONS        '''
'''# =========================================================================================================================================================='''


def split_image(original_image, tile_size):
    
    image_shape = np.shape(original_image)

    tile_rows = np.reshape(original_image, [image_shape[0], -1, tile_size[1]])

    serial_tiles = np.transpose(tile_rows, [1, 0, 2])
    
    tiles = np.ndarray.astype(np.reshape(serial_tiles, [-1, tile_size[1], tile_size[0]]), float)
    
    return tiles

#%%

'''# =========================================================================================================================================================='''
'''#    FUNCTION TO CALCULATE DISCRETE COSINE TRASNFORM OF EACH SUBIMAGE   '''
'''# =========================================================================================================================================================='''

''' NB: This function is not called in A1_code.py because unsplitting of tiles is
    not necessary for FEATURE EXTRACTION. It was used to help image the reconstructed
    images following inverse DCT.'''

def unsplit_tiles(tiles, image_size):
    
    tile_width = np.shape(tiles)[1]
    
    serialized_tiles = np.reshape(tiles, [-1, image_size[0], tile_width])
    
    rowwise_tiles = np.transpose(serialized_tiles, [1, 0, 2])
    
    reformed_image = np.reshape(rowwise_tiles, [image_size[0], image_size[1]]).astype(int)
    
    return reformed_image

#%%

'''# =========================================================================================================================================================='''
'''#  FUNCTION TO CALCULATE 2-D DISCRETE COSINE TRANSFORM OF EACH SUBIMAGE '''
'''# =========================================================================================================================================================='''

def DCT_2D(subimages):   
    
    subimages_DCT = subimages.copy()                                                # Array containing greyscale values of each subimage
                                                                                    # is copied. Values will be swapped with DCT coefficeints.
    ''' LOOPING OVER EACH TILE / SUBIMAGE... '''
    for i in range(len(subimages)):
        
        tile = subimages[i]                                                        # 2-D DCT is performed, i.e. vvertically and horizontally.
                                                                                    # 128 minused from greyscale values for normalisation around 0.
        subimages_DCT[i] = dct(dct((tile - 128).T, 2, norm = 'ortho').T, norm = 'ortho')
        
    return subimages_DCT

#%%

'''# =========================================================================================================================================================='''
'''#   FUNCTION TO CALCULATE 2-D INVERSE DCT OF SUBIMAGE DCT COEFFICIENTS   '''
'''# =========================================================================================================================================================='''

''' NB: This function is not called in A1_code.py because calculation of IDCT is
    not necessary for FEATURE EXTRACTION. It was used to help reconstruct the original
    images following the compression procedures outlined in later functions.'''

def IDCT_2D(subimages_DCT):   
    
    subimages_IDCT = subimages_DCT.copy()                                           # Array containing DCT coefficients of each subimage       
                                                                                    # is copied. Each value will be swapped with Inverse DCT.
    ''' LOOPING OVER EACH TILE / SUBIMAGE... '''
    for i in range(len(subimages_DCT)):
                                                                                    # Like the DCT function, 2-D IDCT is used to map the image.    
        tile = subimages_DCT[i]                                                     # The values are rounded to integers and 128 is added to each.
                                                                                    # one, yielding greyscale values in range 0:255.
        subimages_IDCT[i] = np.rint(idct(idct((tile).T, 2, norm = 'ortho').T, norm = 'ortho')) +128


    return subimages_IDCT

#%%

'''# =========================================================================================================================================================='''
'''#      FUNCTION TO COMPRESS IMAGES BY QUANTISING DCT COEFFICIENTS       '''
'''# =========================================================================================================================================================='''

def quantisation(Q_factor, subimagesDCT):
    
    quant_matrix = np.array([[16,  11,  10,  16,  24,  40,  51,  61],               # Standard matrix for JPEG image compression with 
                             [12,  12,  14,  19,  26,  58,  60,  55],              # (quality) Q-factor = 50.    
                             [14,  13,  16,  24,  40,  57,  69,  56],
                             [14,  17,  22,  29,  51,  87,  80,  62],              # DCT coefficients are divided by scaled versions
                             [18,  22,  37,  56,  68, 109, 103,  77],              # of this matrix to yield quantised coefficients that
                             [24,  35,  55,  64,  81, 104, 113,  92],              # compress the data. Larger values in bottom right
                             [49,  64,  78,  87, 103, 121, 120, 101],              # result in larger 'punishment' of DCT coefficients
                             [72,  92,  95,  98, 112, 100, 103,  99]])             # in corresponding position... they are reduced to 0.
        
    
    if Q_factor == 100:                                                             # For Q_factor == 100, no compression is made.
        quant_matrix = np.ones((8,8))
    else:
        if Q_factor >= 50:                                                          # For Q_factor < 100, DCT coefficients are penalized.
            quant_matrix = np.rint(quant_matrix * (100-Q_factor)/50)                # to certain degree depending on the Q_factor.
            quant_matrix[quant_matrix > 255] = 255
        if Q_factor < 50:                                                           # The larger the Q_factor the less data is lost.        
            quant_matrix = np.rint(quant_matrix * 50/Q_factor)
            quant_matrix[quant_matrix > 255] = 255
            
    quantised_DCT = np.rint(subimagesDCT / quant_matrix) 
    
    return quantised_DCT, quant_matrix

#%%

'''# =========================================================================================================================================================='''
'''#    FUNCTION TO CREATE HUFFMAN MATRIX THAT WILL ENCODE FEATURE VECTORS  '''
'''# =========================================================================================================================================================='''

def create_huffman(triangular_number):
    
    ''' Huffman encoding uses a zig-zig pattern to navigate through DCT matrix
        and flatten all values into a 1D vector. The nature of the DCT matrix 
        and encoding pattern is such that, as the vector continues, its values
        will decrease in size. Therefore, it was deemed possible to splice the
        DCT matrices so that only the largest values, which capture most of the
        variance in an image, are output.
        
        To achieve this, 'Huffman matrices' of varying encoding levels were 
        created. These matrices are 'upper-left triangular matrices' which 
        can be mapped onto DCT matrices to reduce smaller values to zero. 
        
        The combination of compression and encoding can result in smaller
        feature vectors which adequately represent the original image.'''
    
    if triangular_number == 64:                                                   # If input = 64, no encoding is done.
        matrix = np.ones((8, 8))
    
    if triangular_number == 10:                                                   # If input is 10, upper-left triangle of  
        matrix = np.array([[1,  1,  1,  1,  0,  0,  0,  0],                       # size 10 is assigned to 1. All other values
                           [1,  1,  1,  0,  0,  0,  0,  0],                       # assigned to 0, result in DCT coefficients of   
                           [1,  1,  0,  0,  0,  0,  0,  0],                       # in the same position to be zeroes when encoding 
                           [1,  0,  0,  0,  0,  0,  0,  0],                       # is carried out.  
                           [0,  0,  0,  0,  0,  0,  0,  0],
                           [0,  0,  0,  0,  0,  0,  0,  0],
                           [0,  0,  0,  0,  0,  0,  0,  0],
                           [0,  0,  0,  0,  0,  0,  0,  0]])
        
    if triangular_number == 15:                                                   # Same as above, but with larger triangle
        matrix = np.array([[1,  1,  1,  1,  1,  0,  0,  0],                       # with size 15                   
                           [1,  1,  1,  1,  0,  0,  0,  0],
                           [1,  1,  1,  0,  0,  0,  0,  0],
                           [1,  1,  0,  0,  0,  0,  0,  0],
                           [1,  0,  0,  0,  0,  0,  0,  0],
                           [0,  0,  0,  0,  0,  0,  0,  0],
                           [0,  0,  0,  0,  0,  0,  0,  0],
                           [0,  0,  0,  0,  0,  0,  0,  0]])
    
    if triangular_number == 21:                                                   # Same as above, but with larger triangle  
        matrix = np.array([[1,  1,  1,  1,  1,  1,  0,  0],                       # with size 21
                           [1,  1,  1,  1,  1,  0,  0,  0],
                           [1,  1,  1,  1,  0,  0,  0,  0],
                           [1,  1,  1,  0,  0,  0,  0,  0],
                           [1,  1,  0,  0,  0,  0,  0,  0],
                           [1,  0,  0,  0,  0,  0,  0,  0],
                           [0,  0,  0,  0,  0,  0,  0,  0],
                           [0,  0,  0,  0,  0,  0,  0,  0]])
        
    if triangular_number == 28:                                                   # Same as above, but with larger triangle        
        matrix = np.array([[1,  1,  1,  1,  1,  1,  1,  0],                       # Swith size 28
                           [1,  1,  1,  1,  1,  1,  0,  0],
                           [1,  1,  1,  1,  1,  0,  0,  0],
                           [1,  1,  1,  1,  0,  0,  0,  0],
                           [1,  1,  1,  0,  0,  0,  0,  0],
                           [1,  1,  0,  0,  0,  0,  0,  0],
                           [1,  0,  0,  0,  0,  0,  0,  0],
                           [0,  0,  0,  0,  0,  0,  0,  0]])
  
    return matrix

#%%

'''# =========================================================================================================================================================='''
'''#    FUNCTION TO ENCODE (COMPRESSED) IMAGES INTO IMAGE FEATURE VECTOR   '''
'''# =========================================================================================================================================================='''

def encoding(compressed_tiles, huff_matrix):
                        
    reducing_value = np.count_nonzero(huff_matrix)                              # Number to which 8x8 = 64 coeffs will be reduced,
    reducing_indices = np.nonzero(huff_matrix)                                  # indexed as positions of 'ones' in huffman matrix.
    
    encoded_tiles = compressed_tiles[:] * huff_matrix                           # KEY DATA ENCODING STEP. 'Information lost' by zeroing
                                                                                # subimage DCT values that do not align with Huffman matrix.
    encoded_subfeatures = np.zeros((compressed_tiles.shape[0], reducing_value))
    
    height = encoded_subfeatures.shape[0]                                       # Create an array to contain subfeatures, i.e. DCT
    width = encoded_subfeatures.shape[1]                                        # coefficients that are retained following quantisation.
    
    ''' LOOPING OVER EACH TILE / SUBIMAGE... '''
    for tile in range(compressed_tiles.shape[0]):
                                                                                # Subfeatures for each tile obtained by mapping 
        encoded_subfeatures[tile] = compressed_tiles[tile][reducing_indices]    # huffman matrix onto DCT coefficients.

    encoded_features = np.reshape(encoded_subfeatures, height*width)            # Flattening the subfeatures for each tile to give a 
                                                                                # list of overall features for the whole image 
    return encoded_features, encoded_tiles

#%%



