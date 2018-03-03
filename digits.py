#
#
# digits.py
#
#

import numpy as np
from sklearn import cross_validation
import pandas as pd

# For Pandas's read_csv, use header=0 when you know row 0 is a header row
# df here is a "dataframe":
df = pd.read_csv('digits.csv', header=0)
df.head()
df.info()

# Convert feature columns as needed...
# You may to define a function, to help out:
# the transform function transforms the result column from just number to a string
# Ex. 3 -> 'digit 3'
def transform(s):
    """ from number to string
    """
    return 'digit ' + str(s)
    
df['label'] = df['64'].map(transform)  # apply the function to the result column (64)
print("+++ End of pandas +++\n")

# import sys
# sys.exit(0)

print("+++ Start of numpy/scikit-learn +++")

# We'll stick with numpy - here's the conversion to a numpy array
X_data = df.iloc[:,0:64].values        # iloc == "integer locations" of rows/cols
y_data = df[ 'label' ].values      # also addressable by column name(s)

#
# you can divide up your dataset as you see fit here...
#
X_data_unknown = X_data[10:22,0:64] #the set of X_data that don't have the result column
X_data_half_known = X_data[:10,0:40] #the set of X_data that are half erased 
X_data_full = X_data[22:,0:64] #the rest of complete x_data with results 
y_data_full = y_data[22:] #y_data(result column) for the x_data_full


#
# feature display - use %matplotlib to make this work smoothly
#
from matplotlib import pyplot as plt

def show_digit( Pixels ):
    """ input Pixels should be an np.array of 64 integers (from 0 to 15) 
        there's no return value, but this should show an image of that 
        digit in an 8x8 pixel square
    """
    print(Pixels.shape)
    Patch = Pixels.reshape((8,8))
    plt.figure(1, figsize=(4,4))
    plt.imshow(Patch, cmap=plt.cm.gray_r, interpolation='nearest')  # cm.gray_r   # cm.hot
    plt.show()
    


#
# we can scramble the remaining data if we want - only if we know the test set's labels
# 
indices = np.random.permutation(len(X_data_full))  # this scrambles the data each time
print(indices)
X_data_full = X_data_full[indices]
y_data_full = y_data_full[indices]


#
# feature engineering based on my own intuition (on the X_data_full)
#
X_data_full[:,0] *= 0.1   
X_data_full[:,1] *= 1.7
X_data_full[:,2] *= 1.4
X_data_full[:,3] *= 1.1
X_data_full[:,4] *= 1.1
X_data_full[:,5] *= 1.4   
X_data_full[:,6] *= 1.7
X_data_full[:,7] *= 0.1

#
# splitting X_data_full into X_train and X_test (split value can change)
#
split = 40
X_test = X_data_full[0:split,:]              # the final testing data
X_train = X_data_full[split:,:]              # the training data
# creating another pair of test and train data based for the half erased digit test (keeping column 0 to 40)
X_half_test = X_data_full[0:split,:40]
X_half_train = X_data_full[split:,:40]
#splitting y_data accorind to the x_data split 
y_test = y_data_full[0:split]                  # the final testing outputs/labels (unknown)
y_train = y_data_full[split:]                  # the training outputs/labels (known)



#
# here, you'll implement the kNN model
#

from sklearn.neighbors import KNeighborsClassifier

k_choices = [1,3,5,7,9,11,15,21,32,42,51,71,91]
best_k = 0
best_train_average = 0
best_test_average = 0
best_data_average = 0

for k_value in k_choices:
    knn = KNeighborsClassifier(n_neighbors=k_value)   

    #initializing variables for average scores for training and testing 
    training_data_score_average = 0
    testing_data_score_average = 0

    fold_number = 10
    for i in range(fold_number):

#
# run cross-validation
#
        # cross-validate (use part of the training data for training - and part for testing)
        # here 3/4 train and 1/4 test
        cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
            cross_validation.train_test_split(X_train, y_train, test_size=0.25) 

        # fit the model using the cross-validation data
        knn.fit(cv_data_train, cv_target_train) 
        train_score = knn.score(cv_data_train,cv_target_train)
        training_data_score_average += train_score #data data score aggregating 

        test_score = knn.score(cv_data_test,cv_target_test)
        testing_data_score_average += test_score #test data score aggregating 

    #calculating averages scores for each k value
    training_data_score_average = training_data_score_average / fold_number
    testing_data_score_average = testing_data_score_average / fold_number
    print("\nRunning cv tests for k = ", k_value)
    print("Digits KNN cv average training-data score:",training_data_score_average)
    print("Digits KNN cv average testing-data score:", testing_data_score_average)
    #calculating the total data average score 
    #whole data average is calculated as the following because the training data and testing data 
    #accounts for the total training data 25% and 75% respectively.
    data_score_average = training_data_score_average * 0.25 + testing_data_score_average * 0.75
    print("Digits KNN cv average testing-data score:", data_score_average) 

    #Use the avaerages of each k value to find the best k value accordingly
    if (training_data_score_average >= best_train_average):
        best_train_average = training_data_score_average
    if (testing_data_score_average >= best_test_average):
        best_test_average = testing_data_score_average
    if (data_score_average >= best_data_average):
        best_data_average = data_score_average
        best_k = k_value

    print (best_data_average)

print ("best k value is: ", best_k)
#once k value is found, create knn using the best k value 
knn = KNeighborsClassifier(n_neighbors=best_k)

""""
k_choices = [1,3,5,7,9,11,15,21,32,42,51,71,91]
best_k = 0
best_train_average = 0
best_test_average = 0
best_data_average = 0

for k_value in k_choices:
    knn2 = KNeighborsClassifier(n_neighbors=k_value)   

    #initializing variables for average scores for training and testing 
    training_data_score_average = 0
    testing_data_score_average = 0

    fold_number = 10
    for i in range(fold_number):

#
# run cross-validation
#
        #   first, create cross-validation data (here 3/4 train and 1/4 test)
        cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
            cross_validation.train_test_split(X_half_train, y_train, test_size=0.25) # random_state=0 

        # fit the model using the cross-validation data
        knn2.fit(cv_data_train, cv_target_train) 
        train_score = knn2.score(cv_data_train,cv_target_train)
        #print("KNN cv training-data score:", train_score)
        #print("KNN cv training-data score:", knn.score(cv_data_train,cv_target_train))
        training_data_score_average += train_score

        test_score = knn2.score(cv_data_test,cv_target_test)
        #print("KNN cv testing-data score:", test_score)
        testing_data_score_average += test_score
        #print("KNN cv testing-data score:", knn.score(cv_data_test,cv_target_test))

    #calculating averages scores for each k value
    training_data_score_average = training_data_score_average / fold_number
    testing_data_score_average = testing_data_score_average / fold_number
    print("\nRunning cv tests for k = ", k_value)
    print("Digits KNN cv average training-data score:",training_data_score_average)
    print("Digits KNN cv average testing-data score:", testing_data_score_average)

    #calculating the total data average score 
    #whole data average is calculated as the following because the training data and testing data 
    #accounts for the total training data 25% and 75% respectively.
    data_score_average = training_data_score_average * 0.25 + testing_data_score_average * 0.75
    print("Digits KNN cv average testing-data score:", data_score_average) 

    #Use the avaerages of each k value to find the best k value accordingly
    if (training_data_score_average >= best_train_average):
        best_train_average = training_data_score_average
    if (testing_data_score_average >= best_test_average):
        best_test_average = testing_data_score_average
    if (data_score_average >= best_data_average):
        best_data_average = data_score_average
        best_k = k_value

    print (best_data_average)

print ("best k value is: ", best_k)
knn2 = KNeighborsClassifier(n_neighbors=best_k)
"""
knn2 = KNeighborsClassifier(n_neighbors=3)


# this next line is where the full training data is used for the model
knn.fit(X_train, y_train) 
print("\nCreated and trained a knn classifier ([Predict the full-data unknowns])")  

# here are some examples, printed out:
print("digit_X_test's predicted outputs are")
print(knn.predict(X_test))

# and here are the actual labels (digits)
print("and the actual digits are")
print(y_test)

print("\n figuring out what unknown digits for rows 12-23 lab [Predict the full-data unknowns]")
print(knn.predict(X_data_unknown))
#"""

print("\n")
knn2.fit(X_half_train, y_train) 
print("\nCreated and trained knn2 classifier [Predict the partial-data unknowns]")  
print(knn2.predict(X_half_test))
print("and the actual digits are")
print(y_test)
print("\ndeciphering the partially erased digits [Predict the partial-data unknowns]:")
print(knn2.predict(X_data_half_known))

## showing a digit based on the row in the dataset:

#row = 4
#Pixels = X_data[row:row+1,:]
#show_digit(Pixels)
#print("That image has the label:", y_data[row])
#





"""
Comments and results:

Briefly mention how this went:
  + what value of k did you decide on for your kNN?   1
  + how smoothly were you able to adapt from the iris dataset to here? pretty smooth 
  + how high were you able to get the average cross-validation (testing) score? 
  For the first prediction, the average cross-validation can get as high as 0.98
  For the second predcition, the average cross-validation is around 0.95




Then, include the predicted labels of the 12 digits with full data but no label:
Past those labels (just labels) here:
You'll have 12 lines:
['digit 9' 'digit 9' 'digit 5' 'digit 5' 'digit 6' 'digit 5' 'digit 0'                                                              
 'digit 3' 'digit 8' 'digit 9' 'digit 8' 'digit 4']



And, include the predicted labels of the 10 digits that are "partially erased" and have no label:
Mention briefly how you handled this situation!?

Past those labels (just labels) here:
You'll have 10 lines:

['digit 0' 'digit 0' 'digit 0' 'digit 1' 'digit 7' 'digit 2' 'digit 3'                                                              
 'digit 4' 'digit 0' 'digit 1']    



"""