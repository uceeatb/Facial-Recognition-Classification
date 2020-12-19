''' 
    'training.example.py' is an example of the general process used to train models for
    each task. The process for the tasks was largely similar, so it is encapsulted in
    this file in a format that is generalised for all tasks.
    With this generality, teh file contains empty assignments and theoretical values,
    but aims to guide the user via comments.
    
    
    The module is split into 5 sections:
        
        -  Loading dataframe, obtaining training and test sets and scaling features
        
        -  GridsearchCV fitting of model with various model hyperparameters and PCA
        
        -  Analysing output of GridsearchCV using heatmap to determine how to shrink
           the parameter space for the next GridSearch.
           
               (REPEAT BULLETS 2 AND 3 UNTIL OPTIMAL HYPERPARAMETERS ARE FOUND)
           
        -  Fitting, predicting and scoring training/test accuracies of best model 
           using several scoring metrics 
        
        -  Plotting of learning curve for best model to understand bias / variance
           of overall preprocessing and learning model 
'''


#%%

import os

import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, plot_confusion_matrix
#%%





'''# =========================================================================================================================================================='''
'''#            SECTION 1: LOADING DATA FROM SAVED DATAFRAME                '''
'''# =========================================================================================================================================================='''
#%%
# Specifying the directory in which the desired dataframe can be fine
# Dafarames were made using XX_data_saving functions in each task specific module (XX_code.py)
data_directory = r'C:\Users\prave\OneDrive - University College London\SEM 1\Applied ML\CW\AMLS_assignment_kit\AMLS_20-21_SN12345678\dataframes\A2'

# Loading the saved dataframe containing features vectors and labels of preprocessed images
os.chdir(data_directory)

# Converting file into Pandas dataframe. Data was saved as both .pkl and .csv files
# for safety and reliability purposes. Either can be loaded.
file = 'A2__Cell_size_8x8'
#dataset = pd.read_csv(file + '.csv', index_col = 0)
dataset = pd.read_pickle(file + '.pkl')

#%%
# Dataframes that were created using XX_saving_data functions in the task-specific 
# modules (XX_code.py, where XX is the task name) 
# Dataframe contains concatenated feature vectors and labels.
# The following slicing reseparates the feature vectors and labels.

X = dataset[list(dataset.columns)[0:-1]]
Y = dataset[list(dataset.columns)[-1]]
            
#%%
# Dataframes vertically stacked the training and test sets. During preprocessing, 
# the two sets underwent identical procedures, so they can be accessed simply by
# indexing the dataframe in the ratio of their splitting (80% to 20%).

total_examples = X.shape[0]
train_size = int(total_examples * 0.8)
test_size = int(total_examples * 0.2)

X_train = X[0:train_size]
X_test = X[train_size:]
y_train = Y[0:train_size]
y_test = Y[train_size:]

#%%
# Data is scaled for equal contribution of features to classification model

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#%%






'''# =========================================================================================================================================================='''
'''#   SECTION 2: RUNNING GRIDSEARCHCV ALGORITHM TO ASSESS HYPERPARAMETERS  '''
'''# =========================================================================================================================================================='''
#%%
# Building pipeline of PCA then SVM fitting. Gridsearch using 4-fold stratified
# cross-validation such that the current training (80% of total data) is constantly
# further split into a new training set (60%) and test set (20%) so that all examples
# are used for training and testing and a mean test score can be calculated 

pca = PCA()
svc = SVC()

order = Pipeline(steps = [('pca', pca), ('svm', svc)])

parameters = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__kernel':['rbf', 'linear'],
    'svm__gamma': [0.00005, 0.0001, 0.0005, 0.001, 0.005],
    'pca__n_components': [0.9, 0.8, 0.7, 0.6]
    }
# The above hyperparameters have 160 permutations. Some will result in repeated fittings
# i.e. redundant fittings, since the hyperparameters are not compatible, e.g.
# linear kernel does not require gamma

# njobs argument allow spearate fittings to be computed in parallel on different cores
# njobs = -1 uses all the cores on the users computer
# 4-fold CV results in 160 x 4 = 640 total fittings

clf = GridSearchCV(order,
                   param_grid = parameters,
                   scoring = 'accuracy',
                   n_jobs = -1,
                   cv = 4,
                   return_train_score = True,
                   verbose = 10)

clf.fit(X_train_scaled, y_train)

#%%
# Saving results dictionary to a pickle file for later interpretation
results_dictionary = clf.cv_results_

ff = open(str(file) + '_GRIDSEARCH.pkl',"wb")
pickle.dump(results_dictionary ,ff)
ff.close()

best_params = [clf.best_params_, clf.best_score_]
print(best_params)    
'''
EXAMPLE OUTPUT: {pca__n_components': 0.8, 'svm__C': 0.001', 'svm__gamma': 0.001, 'svm__kernel': 'linear'}, 0.8125
'''
#%%





'''# =========================================================================================================================================================='''
'''#  SECTION 3: PLOT HEATMAP TO ASSESS EFFECT OF HYPERPARAMS ON TEST SCORE '''
'''# =========================================================================================================================================================='''
#%%
# Obtaining masked arrays from the results_dictionary. Here, C and 
# gamma are assessed, however PCA n_components could also be analysed.
# Masked arrays are difficult to index, so their data requires recreation

C_values = results_dictionary['param_svm__C']
gamma_values = results_dictionary['param_svm__C']
Mean_scores = results_dictionary['mean_test_score']

#%%
# Reiterating the C and gamma values used for the gridsearch (for clarity)
C_range = [0.1, 1, 10, 100]
gamma_range = [0.00005, 0.0001, 0.0005, 0.001, 0.005]

#%%
#Recreation of C_values and gamma_values as numpy arrays
C_values = np.asarray(C_range * len(gamma_range))

gamma_values = []

for i in range(len(gamma_range)):
    for j in range(len(C_range)):
        gamma_values.append(gamma_range[i])
    
gamma_values = np.asarray(gamma_values)

#%%

# Plotting heatmap of Gridsearch results using Seaborn:
#   x-axis      = C_values
#   y-axis      = gamma_values
#   colour axis = Mean_scores

fig, ax = plt.subplots(1, 1, figsize=(7,6))
plt.title('Gridsearch A1 test scores for Linear SVM varying PCA and C. [Huffman = 15, CLAHE = Yes]')

data = pd.DataFrame(data={'C':C_values,
                          'Gamma':gamma_values,
                          'Mean score':Mean_scores})

data = data.pivot(index='Gamma captured',
                  columns='C',
                  values='Mean score')

fig = sns.heatmap(data, 
                  cmap = 'rocket', 
                  cbar_kws={'label': 'Mean Test Score'}, 
                  fmt='g')

# Changing the degree of decimal places for x and y axes
x_labels = ['%.4f' % float(t.get_text()) for t in fig.get_xticklabels()]
ax.set_xticklabels(x_labels)
y_labels = ['%.3f' % float(t.get_text()) for t in fig.get_yticklabels()]
ax.set_yticklabels(y_labels)

# Inverting y-axis so that highest number in gamma_range is at top of heatmap
ax.invert_yaxis()

## Adding border to heatmap
for _, spine in fig.spines.items():
    spine.set_visible(True)
    
plt.show()
#%%

''' Further Gridsearches to narrow down the parameter space and find optimal parameters
                                            
                                            ...                                   
                                            
                                            ...
                                            
                                            ...
                                            
                                            ...
                                            
'''



'''# =========================================================================================================================================================='''
'''#  SECTION 4: FITTING MODEL WITH OPTIMAL HYPERPARAMS & SCORING PREDICTIONS '''
'''# =========================================================================================================================================================='''
#%%
# Defining the optimal SVM classifier based on optimal hyperparameters
#svm = SVC(C = 10, kernel = 'rbf', gamma = 0.001)
svm  = SVC(C = 0.001, kernel = 'linear')

#%%
# Applying optimal explained variance for PCA
pca = PCA(n_components = 0.7)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

#%%
# Fitting the optimal SVM on the PCA reduced dataset
svm.fit(X_train_pca, y_train)

#%%
#Predicting the accuracy of the model on the training and test set
y_train_pred = svm.predict(X_train_pca)
y_test_pred = svm.predict(X_test_pca)


#%%
#Scoring the accuracy via different metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print('A1 training accuracy score: ', train_accuracy)
print('A1 testing accuracy score: ', test_accuracy)

# Plotting a confusion matrix to display how precison and recall can be calculated
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(svm, X_test_pca, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title) 
    print(title)
    print(disp.confusion_matrix)

# Weighted averages used F1 score for MULTICLASS classification tasks, i.e. 
# B1 and B2. In binary classification problems, this argument is redundant.
F_score = f1_score(y_test, y_test_pred, average='weighted')
print('\nF1 score:', F_score)

#%%







'''# =========================================================================================================================================================='''
'''#  SECTION 5: PLOT LEARNING CURVE TO ASSESS VARAINCE/BIAS OF FINAL MODEL '''
'''# =========================================================================================================================================================='''
#%%
# Locate directory on the 'common_functions' module and importing funciton to plot learning curve
common_functions_directory = r'...'
os.chdir(common_functions_directory)

from common_functions import plot_learning_curve

#%%
# Creating a figure to display learning curve
fig, plot_axes = plt.subplots(1, 1, figsize=(8, 5))

# Sizes of training subsets at which data points for training and validation accuracy 
#will be plot to build a curve
training_subsets = np.linspace(0.1, 1.0, 10)
plt.ylim(0.76, 1.00)
##
plot_title = "Learning Curve for Task XX (Linear SVM, C = 0.001, PCA = 0.8)"

#Defining cross-validation method as Stratified K fold cross validation
cv = StratifiedKFold(n_splits=8)

# Fitting the data to the training set in different sized subsets and displaying the graph
plot_learning_curve(svm, plot_title, X_train_pca, y_train, training_subsets, plot_axes, cv=cv, n_jobs=-1)
plt.show()

#%%
