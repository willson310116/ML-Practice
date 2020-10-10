import tensorflow as tf
from tensorflow import keras  # a API for tensorflow
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

# load data
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# rescale data
train_images = train_images/255.0
test_images = test_images/255.0

# print(train_images[7])
# plt.imshow(train_images[7], cmap=plt.cm.binary)
# plt.show()

# conduct a neural network with 3 layer fully-connected layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# complie and train our model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# for times in range(1,11):
#     model.fit(train_images, train_labels, epochs=times)
#
#     # show result by using evaluate
#     test_loss, test_acc = model.evaluate(test_images, test_labels)
#     print(f"{times} epochs\ttest Acc: {test_acc}")

model.fit(train_images, train_labels, epochs=5)
prediction = model.predict(test_images[7])

# print(prediction[0]) give the prediction array
# print(np.argmax(prediction[0])) give the prediction of the most likely answer

# show images and what the model predict they are
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(f"Actual: {class_names[test_labels[i]]}")
    plt.title(f"Prediction: {class_names[np.argmax(prediction[i])]}")
    plt.show()
