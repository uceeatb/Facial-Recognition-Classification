# Facial Recognition Classification

The following personal project includes a set of Python 3 executable facial recognition tasks (A1, A2, B1, B2), using binary or multiclass classification methods on two separate datasets.

Project report: https://online.fliphtml5.com/srowp/vcet/#p=1

# Contents:

## Main Scripts
- main.py
- common_functions.py
- training_examples.py

main.py is the MAIN file for assessment, accessessing task specific modules (see below) to run the data processing and model training for each task.
common_functions.py and training_examples.py have functionality for execution of the task-specific files

## Task scripts
The folder also contains task-specific folders:
- A1   (Binary gender detection: male or female)
- A2   (Binary emotion detection: smiling or not smiling)
- B1   (Multiclass face shape recognition: 5 types of face shapes)
- B2   (Multiclass eye color recognition: 5 types of eye colors)

These folders each contain the following files, where 'XX' denotes the task name):
- XX_code.py
- XX_helper_functions.py

For further information on each module, please open the modules as .txt files or in Spyder.
EACH FILE CONTAINS A SMALL DESCRIPTION AT THE BEGINNING WHICH IMRPOVES READER'S UNDERSTANDING.

## Datasets folder

Due to GitHub storage limits, the Datasets folder is empty.

- celeba: A sub-set of CelebA dataset. This dataset contains 5000 images for task A1 and A2.
- cartoon_set: A subset of Cartoon Set. This dataset contains 10000 images for task B1 and B2.

Datasets can be accessed via the following link:
https://drive.google.com/file/d/1wGrq9r1fECIIEnNgI8RS-_kPCf8DVv0B/view?usp=sharing 


# Installation requirements
 - Sklearn
 - scipy
 - cv2
 - PIL
 - numpy
 - matplotlib
 - seaborn
 - time
 - os
 - sys 
 
# References

CelebFaces Attributes Dataset (CelebA), a celebrity image dataset (S. Yang, P. Luo, C. C. Loy, and X. Tang, "From facial parts responses to face detection: A Deep Learning Approach", in IEEE International Conference on Computer Vision (ICCV), 2015)

Cartoon Set, an image dataset of random cartoons/avatars (source: https://google.github.io/cartoonset/).
