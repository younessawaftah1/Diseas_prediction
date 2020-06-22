import sklearn
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
import pickle
import numpy as np
import pandas as pd
from sklearn import model_selection
dataset=pd.read_csv('dicease_data.csv',sep=",")
array=dataset.values
dataset.drop(['foul_smell_of urine','lack_of_concentration',
'blackheads','continuous_sneezing','mucoid_sputum'],
inplace=True , axis=1)

X = dataset.drop(['prognosis'],axis =1)
y = dataset['prognosis']
print(dataset.shape)


new_input = np.array([[0,0,0,1,1,1,0,1,0,0,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]])


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y
                                    , test_size=0.2,random_state=7)
model = BernoulliNB()
model = model.fit(x_train, y_train)

#
Accuracy= model.score(x_test, y_test) * 100
print("the Accuracy of this test :",Accuracy)
    
with open("Disease.pkl", "wb") as f:
    pickle.dump(model, f)
    

pickle_in = open("Disease.pkl", "rb")
model = pickle.load(pickle_in)

predictions = model.predict(new_input)
print("the predicted output is:",predictions)

