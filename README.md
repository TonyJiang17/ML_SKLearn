#Machine Learning Project Part 1 

Overview: 
I have always been interested in machine learning, but I would only be able to take AI/ML
related courses starting from the second half of Junior year in college. Thus, I decided to
learn some basic ML modeling using python during freedom on my own. Here are some mini ML projects
I did using sklearn (cross validation)'s closest neighbor (KNN) model to predict specific values in
various datasets. 

Specific Files:
1. iris.py
In this mini project, I am using a dataset that lists different types of irises based off of specific 
flower traits such as, "sepal width, petal length, etc". Using the KNN Model, I tried to determine the 
type of the irises of the first 10 rows, which has their type column listed as unknown. 
(sub-file: iris.csv)

2. digits.py 
Digits.py uses pixals of handwritten numbers to determine what pixal patterns represent 
a specific number. Similar to iris.py, I used the KNN model to predict the written number based on
its pixals. Some other data includes pixal information of only the top half of the written number. Using the same 
KNN model, I tried to predict the written number from only analying the top half pixals information.
(sub-file: digits.csv)

3. titanic.py 
This mini project uses the famous titanic datasets, which includes information of all passenger aboard the ship.
By training the dataset using KNN model, I was able to predict with 80% accuracy whether a specific passenger had 
survived the shipwreck or not. 
(sub-file:titanic.csv)

Final Thoughts:
KNN is quite useful and it's very good at predicting dataset that have data in the same units. It's not necessary very 
good when different features in the datasets have different units. Feature engineering is quite interesting and hard to apply
well with KNN.

