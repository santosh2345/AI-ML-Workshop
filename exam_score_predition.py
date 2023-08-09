import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib




data =  [
    [6.0, 6.0, 12.0, 75],
    [7.0, 5.0, 12.0, 82],
    [8.0, 4.0, 12.0, 90],
    [9.0, 3.0, 12.0, 88],
    [10.0, 2.0, 12.0, 94],
    [5.0, 7.0, 12.0, 68],
    [7.5, 5.5, 11.0, 79],
    [8.0, 5.0, 11.0, 85],
    [9.5, 4.5, 10.0, 92],
    [10.5, 4.0, 9.5, 96],
    [6.0, 6.0, 12.0, 72],
    [7.0, 5.0, 12.0, 84],
    [8.0, 4.0, 12.0, 87],
    [9.0, 3.0, 12.0, 91],
    [10.0, 2.0, 12.0, 95],
    [5.0, 7.0, 12.0, 70],
    [7.0, 5.0, 12.0, 81],
    [8.0, 4.0, 12.0, 89],
    [9.0, 3.0, 12.0, 93],
    [10.0, 2.0, 12.0, 97],
    [6.0, 6.0, 12.0, 76],
    [7.0, 5.0, 12.0, 83],
    [8.0, 4.0, 12.0, 86],
    [9.0, 3.0, 12.0, 88],
    [10.0, 2.0, 12.0, 98],
    [5.0, 7.0, 12.0, 78],
    [7.0, 5.0, 12.0, 80],
    [8.0, 4.0, 12.0, 83],
    [9.0, 3.0, 12.0, 91],
    [10.0, 2.0, 12.0, 99],
    [6.0, 6.0, 12.0, 74]
]

columns = ['study_hours', 'practice_hours', 'sleep_hours','exam_Score']
dataset = pd.DataFrame(data,columns = columns)
file = 'study.csv'
dataset.to_csv(file, index=False)
# print(dataset)

# input

x = dataset.drop(columns= ["exam_Score"])
y = dataset["exam_Score"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

# print(x_train)
# print(x_test)

# creating the train model
model = DecisionTreeClassifier()

# importing our input and output data in the model 
model.fit(x_train,y_train)

# model.fit(x,y)

# joblib.dump('model', 'weight-prediction.joblib')

# prediction input
# prediction_input = [[8,8,8]]

# making prediction
prediction = model.predict(x_test)  

# 10 , 3 is our input 
print(prediction)
score = accuracy_score(y_test, prediction)
print(score)