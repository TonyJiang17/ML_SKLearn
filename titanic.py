import numpy as np
from sklearn import datasets
from sklearn import cross_validation
import pandas as pd

# For Pandas's read_csv, use header=0 when you know row 0 is a header row
# df here is a "dataframe":
df = pd.read_csv('titanic.csv', header=0)
df.head()
df.info()

# let's drop columns with too few values or that won't be meaningful
df = df.drop('body', axis=1)  # axis = 1 means column
df = df.drop('home.dest', axis=1)  
df = df.drop('name', axis=1)
df = df.drop('ticket', axis=1)
df = df.drop('boat', axis=1)
df = df.drop('cabin', axis = 1)


# drop all of the rows with missing data:
df = df.dropna()

#see our dataframe again
df.head()
df.info()



# You'll need conversion to numeric datatypes for all input columns
#
def tr_mf(s):
    """ transforming the sex column from string to int
    """
    d = { 'male':0, 'female':1 }
    return d[s]

df['sex'] = df['sex'].map(tr_mf)  #apply

def tr_sur(s):
    """ transforming the survived column from int to string
    """
    dic = {0:'Died', 1:'Survived', -1:'unsure'}
    return dic[s]
    
df['survived'] = df['survived'].map(tr_sur) #apply

def tr_emb(s):
  """transform the embarked column from string to int 
  """
  dic = {'C':0, 'Q':1, 'S':2}
  return dic[s]
df['embarked'] = df['embarked'].map(tr_emb)

df.head()
df.info()


print("+++ end of pandas +++\n")

# import sys
# sys.exit(0)

print("+++ start of numpy/scikit-learn +++")


# extract the underlying data with the values attribute:
y_data = df[ 'survived' ].values                  # also addressable by column name(s)
X_data = df.drop('survived', axis=1).values        # everything except the 'survival' column

# feature engineering based on personal intuition (changing the X_data)
# kinda arbitrary 
X_data[:,0] *= 100   
X_data[:,1] *= 80   
X_data[:,2] *= 5
X_data[:,3] *= 80
X_data[:,4] *= 80
X_data[:,6] *= 20
print(X_data[42:43,:])
#
# you can take away the top 42 passengers (with unknown survival/perish data) here:
#
X_data_full = X_data[42:,:]
X_data_unknown = X_data[:42,:]
y_data_full = y_data[42:]

"""
indices = np.random.permutation(len(X_data_full))  # this scrambles the data each time
#print(indices)
X_data_full = X_data_full[indices]
y_data_full = y_data_full[indices]
"""

#spliting the full data further into test and train (split value can change based on preference)
split = 10
X_test = X_data_full[0:split,:]              # the final testing data
X_train = X_data_full[split:,:]              # the training data

y_test = y_data_full[0:split]                  # the final testing outputs/labels (unknown)
y_train = y_data_full[split:]



#
# the rest of this model-building, cross-validation, and prediction will come here:
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

#
# run cross-validation
#
        #  create cross-validation data (here 3/4 train and 1/4 test)
        cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
            cross_validation.train_test_split(X_train, y_train, test_size=0.25) # random_state=0 

        # fit the model using the cross-validation data
        knn.fit(cv_data_train, cv_target_train) 
        train_score = knn.score(cv_data_train,cv_target_train)
        training_data_score_average += train_score

        test_score = knn.score(cv_data_test,cv_target_test)
        testing_data_score_average += test_score

    #calculating averages scores for each k value
    training_data_score_average = training_data_score_average / fold_number
    testing_data_score_average = testing_data_score_average / fold_number
    print("\nRunning cv tests for k = ", k_value)
    print("Titanic KNN cv average training-data score:",training_data_score_average)
    print("Titanic KNN cv average testing-data score:", testing_data_score_average)
    #calculating the total data average score 
    #whole data average is calculated as the following because the training data and testing data 
    #accounts for the total training data 25% and 75% respectively.
    data_score_average = training_data_score_average * 0.25 + testing_data_score_average * 0.75
    print("Titanic KNN cv average testing-data score:", data_score_average) 

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

knn.fit(X_train, y_train) 
print("\nCreated and trained a knn classifier")  #, knn

# here are some examples, printed out:
print("digit_X_test's predicted outputs are")
print(knn.predict(X_test))

# and here are the actual labels (digits)
print("and the actual digits are")
print(y_test)

print("\n figuring out what unknown digits for first 42 passengers are:")
print(knn.predict(X_data_unknown))


"""
Comments and results:

Briefly mention how this went:
  the average cross-validation (testing) score:

  0.804328572049

Survived or not for the first 42 passengers:
['Survived' 'Died' 'Died' 'Survived' 'Died' 'Died' 'Died' 'Survived'                                                                
 'Survived' 'Survived' 'Survived' 'Survived' 'Died' 'Survived' 'Died'                                                               
 'Died' 'Died' 'Died' 'Survived' 'Died' 'Survived' 'Survived' 'Died'                                                                
 'Survived' 'Survived' 'Survived' 'Survived' 'Died' 'Survived' 'Died'                                                               
 'Died' 'Died' 'Survived' 'Survived' 'Survived' 'Died' 'Survived'                                                                   
 'Survived' 'Died' 'Survived' 'Died' 'Survived']




"""