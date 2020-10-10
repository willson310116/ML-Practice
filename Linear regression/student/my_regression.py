import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")
# print(data.head())
data = data[["G1", "G2", "G3", "studytime", "failures", "internet",
             "freetime", "absences"]]



# print(data.head())
# print("--------")

# for i in range(len(data)):
#     data["internet"][i] = 0 if data["internet"][i] == "no" else 1
# print(data.head())

"""
This method will bump into warnings:
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
So I use a more straightforward method to replace the value in dataframe.
"""

data["internet"] = data["internet"].replace(["no"], 0)
data["internet"] = data["internet"].replace(["yes"], 1)
# print(data.head())

predict = "G3"
# x = data.drop([predict], 1)
# y = data[predict]
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# print(x)
# print("-----------")
# print(y)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, train_size=0.1)

"""
best = 0
for times in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, train_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    print(f"{times} time: {accuracy} ")
    if accuracy > best:
        best = accuracy
        with open("my_studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
print(f"Best is {best}")
"""


pickle_in = open("my_studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print(f"Co: {linear.coef_}")
print(f"Intercept: {linear.intercept_}")
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(f"predeictions: {predictions[x]}  \t x_test: {x_test[x]}    \t y_test: {y_test[x]}")
print(predictions)
print(type(predictions))
print(y_test)

acc = linear.score(x_test, y_test)
print(f"accuracy is {acc}")

# print(style.available)
style.use("ggplot")
# feature = "freetime"
# plt.scatter(data[feature], data["G3"])
# plt.xlabel(feature)
# plt.ylabel("Final Grade")
# plt.xkcd()
x_index = np.arange(len(x_test))

plt.plot(x_index, predictions, label="predictions")
plt.plot(x_index, y_test, label="test_data", linestyle="--")
plt.title("testing_data v.s. predictions")
plt.xlabel("student_index")
plt.ylabel("Final Grade")
plt.legend()
plt.tight_layout()
plt.savefig("predictions.png")
plt.show()

