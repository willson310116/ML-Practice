import pandas as pd
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

# using data from sklearn
cancer = datasets.load_breast_cancer()

# gives all the features/target in a list
# print(cancer.feature_names)
# print(cancer.target_names)

x = cancer.data
y = cancer.target

# print(pd.DataFrame(x))  # the data have [569 rows x 30 columns] with float values
# print(pd.DataFrame(y))  # the data have [569 rows x 1 columns] with values 0/1

best = 0
count = 0
for k in range(1,30):
    for times in range(30):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, train_size=0.2)
        # print(x_train)
        # print(y_train)
        #
        classes = ["malignant", "benign"]

        # clf = svm.SVC()
        # clf = svm.SVC(kernel="poly", degree=2)
        # clf = svm.SVC(kernel="linear", C=2)  # Bring up the data dimension at one, remap the data to generate a hyperplane easier
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)

        accuracy = metrics.accuracy_score(y_test, y_pred) # 前後順序不重要，metrics只是比較裡面兩個而已
        print(f"{times} time: accuracy {accuracy}")
        if accuracy > best:
            count = k
            best = accuracy
print(f"the best accuracy is {best}, with {count}")

#  Compare with KNN, SVM still have a better performance
