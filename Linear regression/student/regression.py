import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
data = pd.read_csv("student-mat.csv", sep=";")

print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# print("--------------")
# print(data.head())

predict = "G3"

x = np.array(data.drop([predict], 1))  # Drop G3 column
y = np.array(data[predict]) # Only the G3 column
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

# This part is only use for training models, and save the best one.
# best = 0
# for times in range(30):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
#
#     linear = linear_model.LinearRegression()
#
#     linear.fit(x_train, y_train)
#     accuracy = linear.score(x_test, y_test)
#     print(f"{times+1} time: {accuracy}")
#
#     if accuracy > best:
#         best = accuracy
#         with open("studentmodel.pickle", "wb") as f:
#             pickle.dump(linear, f) # Use pickle to save our model
# print(f"\tbest is {best}")

# Open the best model that we trained before.
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print(f"Co: {linear.coef_}")
# list 5 coef since we have 5 columns in x_train (5 dimension), meaning the line(model) we trained
print(f"Intercept: {linear.intercept_}")

predictions = linear.predict(x_test) # will take a array
for x in range(len(predictions)):
    print(f"predeictions: {predictions[x]}  \t x_test: {x_test[x]}    \t y_test: {y_test[x]}")

p = "absences" # change feature(column) to check the correlaion by visualization
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
# print(style.available)

