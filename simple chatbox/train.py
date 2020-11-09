import json 
from chat import tokenize, stem, bag_of_words
with open("intent.json", "r") as f:
	intents = json.load(f)

# print(intents)

all_words = []
tags = []
xy = []
for intent in intents["intents"]:
	tag = intent["tag"]
	tags.append(tag)
	for pattern in intent["patterns"]:
		w = tokenize(pattern)
		all_words.extend(w)  # since w is an array so use extend instead of append to avoid list in a list
		xy.append((w, tag))

ignore_words = ["?", "!", ".", ","]
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))  # remove duplicate elements, and sorted will return list
tags = sorted(set(tags))
# print(all_words)
print(tags)

x_train = []
y_train = []
print(xy)
for (pattern_sentence, tag) in xy:
	bag = bag_of_words(pattern_sentence, all_words)
	x_train.append(bag)

	label = tags.index(tag)
	y_train.append(label) # CrossEntrophyLoss
# print(y_train)