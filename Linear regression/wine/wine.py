import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import sklearn
from sklearn import linear_model
import seaborn as sns

data = pd.read_csv("winequality-red.csv", sep=";")  # Always remember to seperate the data first.

corr = data.corr()
# print(corr)
# # f, ax = plt.subplots(figsize=(15, 8))
style.use("ggplot")
sns.heatmap(corr, annot=True, linewidths=.5, fmt='.1f')
plt.savefig("correlation.png")
plt.show()



# print(data.head())
# data_to_np = []

# for i in range(len(data.columns)):
#     data_to_np.append(data.columns[i])

# predict = "quality"
#
# data = data[data_to_np]

# x = np.array(data.drop([predict], 1))
# print(x)
# y = np.array(data[predict])
# print(y)
# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
# print(y_test)
# best = 0
# for times in range(50):
# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#     linear = linear_model.LinearRegression()
#     linear.fit(x_train, y_train)
#     accuracy = linear.score(x_test, y_test)
#     # print(f"The {times} time, accuracy is {accuracy}")
#     if accuracy>best:
#         best = accuracy
#         best_model = linear
#
# print(f"The best accuracy is {best}")
#
# print(f"coef: {best_model.coef_}")
# print(f"intercept: {best_model.intercept_}")
#
# predictions = best_model.predict(x_test)
# # the LinearRegression model returns a float value while the quality pof wine is an integer
# # So we use round() to get close to the result, and recalculate the accuracy.
# print(predictions)
# count_error = 0
# for i in range(len(predictions)):
#     if round(predictions[i]) != y_test[i]:
#         print(f"predictions: {round(predictions[i])}\ty_test: {y_test[i]}, wrong prediction")
#         count_error += 1
#     else:
#         print(f"predictions: {round(predictions[i])}\ty_test: {y_test[i]}")
# print(f"There are {count_error} errors, giving a accuracy of {1-(count_error/len(predictions))}")
#
#
#
# corr = data.corr()
# print(corr)
# # f, ax = plt.subplots(figsize=(15, 8))
# style.use("ggplot")
# sns.heatmap(corr, annot=True, linewidths=.5, fmt='.1f')
# plt.show()

# First we use all the columns, but linear regression model only gives accuracy no over 30%
# And even we round the prediction of the result, it still only gives accuracy no over 70%
# By the heatmap we can find that only sulphates, volatile acidity, alcohol give correlation over 0.3
# Therefore we try to reduce the data into only these 3 columns to train the model

# print(data.head())
predict = "quality"
data_to_np = ["volatile acidity", "sulphates", "alcohol", predict]
data = data[data_to_np]

x = np.array(data.drop([predict], 1))
print(x)
y = np.array(data[predict])
print(y)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
# print(y_test)
best = 0

for times in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.15)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    print(f"The {times} time, accuracy is {accuracy}")
    if accuracy > best:
        best = accuracy
        best_model = linear

print(f"The best accuracy is {best}")
# The best accuracy is still low around 40%

print(f"coef: {best_model.coef_}")
print(f"intercept: {best_model.intercept_}")

predictions = best_model.predict(x_test)
# the LinearRegression model returns a float value while the quality pof wine is an integer
# So we use round() to get close to the result, and recalculate the accuracy.
# print(predictions)
count_error = 0
for i in range(len(predictions)):
    if round(predictions[i]) != y_test[i]:
        print(f"predictions: {round(predictions[i])}\ty_test: {y_test[i]}, wrong prediction")
        count_error += 1
    else:
        print(f"predictions: {round(predictions[i])}\ty_test: {y_test[i]}")
print(f"There are {count_error} errors, giving a accuracy of {1-(count_error/len(predictions))}")



