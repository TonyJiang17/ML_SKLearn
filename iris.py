import numpy as np
from sklearn import cross_validation
import pandas as pd

print("+++ Start of pandas +++\n")
# df here is a "dataframe":
df = pd.read_csv('iris.csv', header=0)    # read the file
df.head()                                 # first five lines
df.info()                                 # column details


def transform(s):
    """ from string to number
          setosa -> 0
          versicolor -> 1
          virginica -> 2
    """
    d = { 'unknown':-1, 'setosa':0, 'versicolor':1, 'virginica':2 }
    return d[s]
    
# 
# this applies the function transform to a whole column
#
df['irisname'] = df['irisname'].map(transform)  # apply the function to the column

print("+++ End of pandas +++\n")

print("+++ Start of numpy/scikit-learn +++")
# Data needs to be in numpy arrays - these next two lines convert to numpy arrays
X_data_full = df.iloc[:,0:4].values        # iloc == "integer locations" of rows/cols
y_data_full = df[ 'irisname' ].values      # individually addressable columns (by name)


#
# we can drop the initial (unknown) rows -- if we want to test with known data
X_data_unknown = X_data_full[:9,:]
X_data_full = X_data_full[9:,:]   # 2d array
y_data_full = y_data_full[9:]     # 1d column


#
# we can scramble the remaining data if we want - only if we know the test set's labels
# 
indices = np.random.permutation(len(X_data_full))  # this scrambles the data each time
#print(indices)
X_data_full = X_data_full[indices]
y_data_full = y_data_full[indices]



#
# The first nine are our test set - the rest are our training
#
X_test = X_data_full[0:9,0:4]              # the final testing data
X_train = X_data_full[9:,0:4]              # the training data

y_test = y_data_full[0:9]                  # the final testing outputs/labels (unknown)
y_train = y_data_full[9:]                  # the training outputs/labels (known)


#
# create a kNN model and tune its parameters (just k!)
#   here's where you'll loop to run 5-fold (or 10-fold cross validation)
#   and loop to see which value of n_neighbors works best (best cv testing-data score)
#
from sklearn.neighbors import KNeighborsClassifier
k_choices = [1,3,5,7,9,11,15,21,32,42,51,71,91]
best_k = 0
best_train_average = 0
best_test_average = 0
best_data_average = 0

for k_value in k_choices:
    knn = KNeighborsClassifier(n_neighbors=k_value)   # 7 is the "k" in kNN

    #initializing variables for average scores for training and testing 
    training_data_score_average = 0
    testing_data_score_average = 0

    fold_number = 10
    for i in range(fold_number):

        # cross-validate (use part of the training data for training - and part for testing)
        #   first, create cross-validation data (here 3/4 train and 1/4 test)
        cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
            cross_validation.train_test_split(X_train, y_train, test_size=0.25) # random_state=0 

        # fit the model using the cross-validation data
        #   typically cross-validation is used to get a sense of how well it works
        #   and tune any parameters, such as the k in kNN (3? 5? 7? 41?, etc.)
        knn.fit(cv_data_train, cv_target_train) 
        train_score = knn.score(cv_data_train,cv_target_train)
        #print("KNN cv training-data score:", train_score)
        #print("KNN cv training-data score:", knn.score(cv_data_train,cv_target_train))
        training_data_score_average += train_score

        test_score = knn.score(cv_data_test,cv_target_test)
        #print("KNN cv testing-data score:", test_score)
        testing_data_score_average += test_score
        #print("KNN cv testing-data score:", knn.score(cv_data_test,cv_target_test))

    #calculating the average scores for traning/testing/total data
    training_data_score_average = training_data_score_average / fold_number
    testing_data_score_average = testing_data_score_average / fold_number
    #printing results
    print("\n")
    print("KNN cv average training-data score:",training_data_score_average)
    print("KNN cv average testing-data score:", testing_data_score_average)
    #whole data average is calculated as the following because the training data and testing data 
    #accounts for the total training data 25% and 75% respectively.
    data_score_average = training_data_score_average * 0.25 + testing_data_score_average * 0.75
    print("KNN cv average testing-data score:", data_score_average) 

    #Use the avaerages of each k value to find the best k value accordingly 
    if (training_data_score_average > best_train_average):
        best_train_average = training_data_score_average
    if (testing_data_score_average > best_test_average):
        best_test_average = testing_data_score_average
    if (data_score_average > best_data_average):
        best_data_average = data_score_average
        best_k = k_value

    print (best_data_average)

#once k value is found, create knn using the best k value 
print ("best k value is: ", best_k)
knn = KNeighborsClassifier(n_neighbors=best_k)



#
# now, train the model with ALL of the training data...  and predict the labels of the test set
#

# this next line is where the full training data is used for the model
knn.fit(X_train, y_train) 
print("\nCreated and trained a knn classifier")  #, knn

# here are some examples, printed out:
print("iris_X_test's predicted outputs are")
print(knn.predict(X_test))

# and here are the actual labels (iris types)
print("and the actual labels are")
print(y_test)

print("The predicted outputs for the 9 initially unknown flowers are:")
print(knn.predict(X_data_unknown))





#
# for testing values typed in
#
def test_by_hand(knn):
    """ allows the user to enter values and predict the
        label using the knn model passed in
    """
    print()
    Arr = np.array([[0,0,0,0]]) # correct-shape array
    T = Arr[0]
    T[0] = float(input("sepal length? "))
    T[1] = float(input("sepal width? "))
    T[2] = float(input("petal length? "))
    T[3] = float(input("petal width? "))
    prediction = knn.predict(Arr)[0]
    print("The prediction is", prediction)
    print()


# import sys   # easy to add break points...
# sys.exit(0)


"""
Comments and results:

Briefly mention how this went:
  + what value of k did you decide on for your kNN?
  + how smoothly did this kNN workflow go...

+The best k value I found and used was : 5

I believe the kNN workflow went fairly smoothly 

Then, include the predicted labels of the first 9 irises (with "unknown" type)
Paste those labels (or both data and labels here)
You'll have 9 lines:

the 9 unknwon flower are (in order):
setosa -> 0
versicolor -> 1
virginica -> 2

[1 2 1 1 0 0 2 1 0]




"""