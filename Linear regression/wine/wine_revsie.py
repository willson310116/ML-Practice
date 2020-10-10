import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import sklearn
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("winequality-red.csv", sep=";")  # Always remember to seperate the data first.
print(data.head())

data_to_np = []

for i in range(len(data.columns)):
    data_to_np.append(data.columns[i])

predict = "quality"

# data_to_np = ["volatile acidity", "sulphates", "alcohol", predict]
print(pd.DataFrame(data[predict]))

# print(len(data[predict]))
for i in range(len(data[predict])):
    data[predict][i] = data[predict][i] * 10
print(pd.DataFrame(data[predict]))


data = data[data_to_np]

x = np.array(data.drop([predict], 1))
# print(x)
y = np.array(data[predict])
# print(y)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Standardization z-score
x_train_std = StandardScaler().fit_transform(x_train)
x_test_std = StandardScaler().fit_transform(x_test)
# y_train_std = StandardScaler().fit_transform(y_train)  # StandardScaler只轉換2D資料，所以y不能轉換，也不需要轉換
# y_test_std = StandardScaler().fit_transform(y_test)
print(pd.DataFrame(x_train_std).head())  # 用dataFrame的形式表示標準化的x_train資料

# Scaling 0~1
x_train_scale = MinMaxScaler().fit_transform(x_train)
x_test_scale = MinMaxScaler().fit_transform(x_test)
print(pd.DataFrame(x_train_scale).head())

# training for std data
best_std = 0

for times in range(200):
    linear_std = linear_model.LinearRegression()
    linear_std.fit(x_train_std, y_train)
    accuracy = linear_std.score(x_test_std, y_test)
    # print(f"{times} time, accuracy = {accuracy}")
    if accuracy > best_std:
        best_std = accuracy
        best_model_std = linear_std
print(f"the best accuracy for std is {best_std}")
# The accuracy after data preprocessing is still low around 30~45%

print("---------------------------------------------")
best_scale = 0
for times in range(200):
    linear_scale = linear_model.LinearRegression()
    linear_scale.fit(x_train_scale, y_train)
    accuracy = linear_scale.score(x_test_scale, y_test)
    # print(f"{times} time, accuracy = {accuracy}")
    if accuracy > best_scale:
        best_scale = accuracy
        best_model_scale = linear_scale
print(f"the best accuracy for scale is {best_scale}")

prediction_std = best_model_std.predict(x_test_std)
prediction_scale = best_model_scale.predict(x_test_scale)

count_error_std = 0
for i in range(len(prediction_std)):
    if round(prediction_std[i]/10)*10 != y_test[i]:
        print(f"predictions: {round(prediction_std[i]/10)*10}\ty_test: {y_test[i]}, wrong prediction")
        count_error_std += 1
    else:
        print(f"predictions: {round(prediction_std[i]/10)*10}\ty_test: {y_test[i]}")

print("--------------------------")
count_error_scale = 0
for i in range(len(prediction_scale)):
    if round(prediction_scale[i]/10)*10 != y_test[i]:
        # print(f"predictions: {round((prediction_scale[i]))}\ty_test: {y_test[i]}, wrong prediction")
        print(f"predictions: {round(prediction_scale[i]/10)*10}\ty_test: {y_test[i]}, wrong prediction")
        count_error_scale += 1
    else:
        print(f"predictions: {round(prediction_scale[i]/10)*10}\ty_test: {y_test[i]}")

print(f"Std: There are {count_error_std} errors, giving a accuracy of {1-(count_error_std/len(prediction_std))}")
print(f"Scale: There are {count_error_scale} errors, giving a accuracy of {1-(count_error_scale/len(prediction_scale))}")

# The accuracy after preprocessing and using round is still not high around 50~60%
# Reducing low correlation with quality columns does not help improve the performance.


