import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from matplotlib import style


data = pd.read_csv("student-mat.csv",sep = ";")

data = data[["G1","G2","G3","studytime","failures","absences",]] #pick data that you want to see
print(data)
predict = "G3"
X = np.array(data.drop([predict],1))
Y = np.array(data[predict])

X_train, X_test,Y_train, Y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.1)

linear = linear_model.LinearRegression()#Linear agression algorithms(직선회귀 알고리즘)
linear.fit(X_train,Y_train)  #finding best fit line using by x_fit line(x_train),y_fitline(y_train)
acc = linear.score(X_test,Y_test) #직선회귀적 정확도 (기울기값)(acc = accuracy)

print("acc : ",acc)
with open("studentmodel.pickle","wb") as f:
    pickle.dump(linear, f)

pickle_in = open("studentmodel.pickle","rb")
linear = pickle.load(pickle_in)

print("Coefficient : ",linear.coef_)
print("Intercept :",linear.intercept_)

predictions = linear.predict(X_test)

for x in range(len(predictions)):
    print("predictions : ",predictions[x],X_test[x],Y_test[x])

p = 'G1'
style.use("ggplot")
plt.scatter(data[p],data["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()