import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

data = pd.read_csv("car.data")
print(data.head())

le = preprocessing.LabelEncoder()
#  turn data into numerical data
buying = le.fit_transform(list(data["buying"]))  # numpy array
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))
# print(buying)
#  len(buying) = 1728, type(buying) = numpy.ndarray, value = 0, 1, 2, 3

predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot, safety))
#  put every array together list(zip([1,2,3],[4,5,6])) = [(1, 4), (2, 5), (3, 6)]
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# KNN classifier needs to set k a odd number, and can't choose too big number.
# Time-consuming since the alg will check the distance of every data to find the nearest data.

# train and save model with a best accuracy when 9 neighbors

best = 0
count = 0
acc_list = []
for n in range(1,50):
    model = KNeighborsClassifier(n_neighbors=n)
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    # print(f"for {n} neighbors acc is {acc}")
    acc_list.append(acc)
    if acc > best:
        best = acc
        count = n
        best_model = model
        with open("car.pickle", "wb") as f:
            pickle.dump(best_model, f)

print(f"{count} neighbors gives a best acc {best}")

pickle_in = open("car.pickle", "rb")
model_trained = pickle.load(pickle_in)

predicted = model_trained.predict(x_test)
print(predicted)
# print(best_model)  # KNeighborsClassifier(n_neighbors=7)
# print(type(best_model))  # <class 'sklearn.neighbors._classification.KNeighborsClassifier'>

names = ["unacc", "acc", "good", "vgood"]

wrong_count = 0
for x in range(len(x_test)):
    n = model_trained.kneighbors([x_test[x]], 9, True)  # n have two arrays of the distance of each neighbors and neighbors
    if predicted[x] != y_test[x]:
        print(f"Predicted: {names[predicted[x]]} Data: {x_test[x]}, Actual: {names[y_test[x]]}  wrong prediction")
        wrong_count += 1
    else:
        print(f"Predicted: {names[predicted[x]]} Data: {x_test[x]}, Actual: {names[y_test[x]]}")
    print(f"N: {n}")

print(f"There are {wrong_count} wrong predictions within {len(x_test)} data giving a {1-(wrong_count/len(x_test))} correct ratio")


# plot to see what number of neighbors gives a better accuracy.
# print(style.available)
# plt.xkcd()
# style.use("fivethirtyeight")
# plt.plot(list(range(1,50)), acc_list)
# plt.title("KNN")
# plt.xlabel("Neighbors")
# plt.ylabel("Accuracy")
# plt.grid(True)
# plt.show()

