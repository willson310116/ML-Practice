import tensorflow as tf
from tensorflow import keras
import numpy as np

# IMDB movie review sentiment classification, labels (1,0) giving the review sentiment (positive/negative)
data = keras.datasets.imdb

# only take words that are 10000 most frequent
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

# print(train_data[0])  # give a list of integer which each of them stands for a words

word_index = data.get_word_index()

word_index = {k: (v + 3) for k, v in word_index.items()}  # v+3 for adding special tags below
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)


# print(len(train_data), len(test_data))

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


# print(decode_review(test_data[0]))
# print(len(test_data[0]), len(test_data[1]))


# # create model
# vocab_size = 88000  # should set properly
# model = keras.Sequential()
# model.add(keras.layers.Embedding(vocab_size, 16))  # create 100000 word vectors with 16 dimensions for each word pass in
# model.add(keras.layers.GlobalAveragePooling1D())  # scale down the dimensions
# model.add(keras.layers.Dense(16, activation="relu"))
# model.add(keras.layers.Dense(1, activation="sigmoid"))  # the output is 0 or 1 and based on probability

# model.summary()

# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# # validation data
# x_val = train_data[:10000]  # take parts of training data as validation data
# x_train = train_data[10000:]

# y_val = train_labels[:10000]
# y_train = train_labels[10000:]

# fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

# results = model.evaluate(test_data, test_labels)

# print(results)  # gives [loss, accuracy]

# # h5 stands for the extension for saving models of tensorflow or keras in binary data
# model.save("model.h5")


def review_encode(s):
    encoded = [1]
    for word in s:
        if word.lower() in word_index:  # word_index gives numbers
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)  # 2 stands for UNK
    return encoded

# def review_encode(s):
#     encoded = [1]
#     for word in s:
#         if word.lower() in word_index:
#             encoded.append(word_index[word.lower()])
#         else:
#             encoded.append(2)
#     return encoded


model = keras.models.load_model("model.h5")

with open("test.txt", encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"",
                                                                                                                  "").strip().split(
            " ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post",
                                                            maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])

# with open("test.txt", encoding="utf-8") as f:
#     for line in f.readlines():
#         nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"",
#                                                                                                                   "").strip().split(
#             " ")
#         encode = review_encode(nline)
#         encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post",
#                                                             maxlen=250)  # make the data 250 words long
#         predict = model.predict(encode)
#         print(line)
#         print(encode)
#         print(predict[0])  # give a result of the text is a positive sentiment

# Example of the input review and the prediction result
'''
test_review = test_data[0]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print(f"Prediction: {str(predict[0])}")
print(f"Actual: {str(test_labels[0])}")
print(results)
'''
