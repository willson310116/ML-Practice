import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import sys

# IMDB movie review sentiment classification, labels (1,0) giving the review sentiment (positive/negative)
data = keras.datasets.imdb

# only take words that are 100000 most frequent

#####################################################################
# If the there are many data, then the variety of the words         #
# might exceed the range of num_words, so we should tune num_words  #
#####################################################################


vocab_size = 100000
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=vocab_size)

# print(train_data[0])  # give a list of integer which each of them stands for a words

word_index = data.get_word_index()

word_index = {k: (v + 3) for k, v in word_index.items()}  # v+3 for adding special tags below
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=1000)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=1000)


# print(len(train_data), len(test_data))

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


# print(decode_review(test_data[0]))
# print(len(test_data[0]), len(test_data[1]))


# # create model
# vocab_size = 100000  # should set properly
# model = keras.Sequential()
# model.add(keras.layers.Embedding(vocab_size, 16))  # create 100000 word vectors with 16 dimensions for each word pass in
# model.add(keras.layers.GlobalAveragePooling1D())  # scale down the dimensions
# model.add(keras.layers.Dense(16, activation="relu"))
# model.add(keras.layers.Dense(1, activation="sigmoid"))  # the output is 0 or 1 and based on probability

# model.summary()


# # validation data
# x_val = train_data[:10000]  # take parts of training data as validation data
# x_train = train_data[10000:]

# y_val = train_labels[:10000]
# y_train = train_labels[10000:]

# # best = 0
# # for epok in range(30, 41):
# #     for batch in range(300,501,50):
# #         model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# #         fitModel = model.fit(x_train, y_train, epochs=epok, batch_size=batch, validation_data=(x_val, y_val), verbose=1)

# #         results = model.evaluate(test_data, test_labels)

# #         if results[1] > best:
# #             best = results[1]
# #             model_best = model
# #             epoch_best = epok
# #             batch_best = batch

# #         print(results)  # gives [loss, accuracy]
# # print(f"Best accuracy {best}\tepochs={epoch_best} \tbatch_size={batch_best} ")

# best = 0
# for times in range(5):
#     model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#     fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)
#     results = model.evaluate(test_data, test_labels)
#     if results[1] > best:
#         best = results[1]
#         model_best = model
#     print(results)  # gives [loss, accuracy]
# print(f"Best accuracy {best}\t")

# # h5 stands for the extension for saving models of tensorflow or keras in binary data
# model_best.save("model.h5")
# # model.save("model.h5")


def clean_string(s):
    """
    Modifying multiple lines string into one string with a proper form
    """
    result = list(s.strip())
    # print(result,"\n")
    for i in range(len(result)):
        if result[i] == '\n':
            result[i] = " "
    result = "".join(result)
    result = result.replace(".", "").replace(",", "").replace("(", "").replace(")", "").replace(":", "").replace("\n", "").replace("\"", "").replace("<", "").replace(">", "").split(" ")
    result = list(filter(("").__ne__, result))
    return result


def review_encode(s):
    encoded = [1]
    for word in s:
        if word.lower() in word_index:  # word_index gives numbers
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)  # 2 stands for UNK
    return encoded

# use a trained model from the same file
model = keras.models.load_model("model.h5")


with open("test.txt", encoding="utf-8") as f:
    nline = f.read()
    print(nline)
    nline_clean = clean_string(nline)
    print("\n")
    print(nline_clean)
    encode = review_encode(nline_clean)
    encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post",maxlen=850)
    predict = model.predict(encode)
    print(predict[0])

    # for line in f.readlines():
    #     nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\n", "").replace("\"",
    #                                                                                                               "").strip().split(
    #         " ")
    #     nline = list(filter(("").__ne__, nline))
    #     encode = review_encode(nline)
    #     encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post",
    #                                                         maxlen=850)  # make the data 250 words long
    #     predict = model.predict(encode)
    #     # print(line)
    #     # print(encode)
    #     print(predict[0])  # give a result of the text is a positive sentiment

print("------------------------------------------")

filename = "load.txt"

with open(filename, encoding="utf-8") as f:
    nline = f.read()
    print(nline)
    nline_clean = clean_string(nline)
    print("\n")
    print(nline_clean)
    encode = review_encode(nline_clean)
    encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post",maxlen=850)
    predict = model.predict(encode)
    print(predict[0])

    # for line in f.readlines():
    #     nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\n", "").replace("\"",
    #                                                                                                               "").strip().split(
    #         " ")
    #     nline = list(filter(("").__ne__, nline))
    #     encode = review_encode(nline)
    #     encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post",
    #                                                         maxlen=850)
    #     predict = model.predict(encode)
    #     # print(line)
    #     # print(encode)
    #     print(predict[0])


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





data_out = pd.read_csv("IMDB Dataset.csv")
data_out["sentiment"].replace({"positive": 1, "negative": 0}, inplace=True)

# num = int(input("How many data do you want to test on this model\n> "))
num = int(sys.argv[1])  # no need to input after the program runs
data_test = data_out[:num]
# print(data_test.head())
# print(len(data_test))
x = data_test["review"]
y = data_test["sentiment"]
# verify whether loading data splitting data work well
# for i in range(5):
#     # print(data_test["review"][i])
#     print(x[i])
#     # print(data_test["sentiment"][i])
#     print(y[i])

count = 0
wrong_prediction = []
for i in range(len(data_test)):
    data_preprocess = clean_string(x[i])
    # data_preprocess = x[i].replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"", "").replace("<", "").replace(">", "").replace("/", "").replace("\n", "").strip().split(" ")
    # data_preprocess = list(filter(("").__ne__, data_preprocess))
    encode = review_encode(data_preprocess)
    encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=850)
    predict = model.predict(encode)
    
    # f"{str:<(num)}" allows alignment
    if round(float(predict[0])) == y[i]:
        count += 1
        print(f"{i} data\t  Predict: {str(predict[0]):<15}\t Actual: {y[i]}")
    else:
        wrong_prediction.append(i)
        print(f"{i} data\t  Predict: {str(predict[0]):<15}\t Actual: {y[i]}\twrong answer")
    
print(f"The accuracy of the external data is {count/len(data_test)}")
print(wrong_prediction)

