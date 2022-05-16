import pandas as pd
import pandas as pd
from random import randint
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import random
import spacy

data = pd.read_csv("data/olid-train.csv")
test_data = pd.read_csv("data/olid-test.csv")

print(data.head())
print(test_data.head())


# Number of Instances
label_zero = data[(data["labels"] == 0)]
number_of_instances_0 = len(label_zero)

label_one = data[(data["labels"] == 1)]
number_of_instances_1 = len(label_one)

print("Class 0:", number_of_instances_0)
print("Class 1:", number_of_instances_1)

# Relative Label Frequency (%)
total = len(data)
freq_zero = (number_of_instances_0 / total)
freq_one = (number_of_instances_1 / total)

print("0 Label Frequency:", freq_zero)
print("1 Label Frequency:", freq_one)

# Example Tweet with This Label
example_tweet_0 = data["text"][(data["labels"] == 0)][4]
example_tweet_1 = data["text"][(data["labels"] == 1)][0]
print("Example tweet for label 0 : ", example_tweet_0)
print("Example tweet for label 1 : ", example_tweet_1)


def random_baseline(train_data, test_data_text, test_data_labels):
    possible_labels = [0, 1]

    predictions = []
    for instance in test_data_text:
        instance_predictions = [random.choice(possible_labels)]
        predictions.append(instance_predictions)

    predictions = [item for sublist in predictions for item in sublist]

    test_labels_list = []
    for element in test_data_labels:
        test_labels_list.append(element)

    prediction_df = pd.DataFrame({"predictions": predictions})

    accuracy = accuracy_score(test_labels_list, predictions)
    print("Accuracy: ", accuracy)

    print(classification_report(test_labels_list, predictions, zero_division=1))


def majority_baseline(train_data_labels, test_data_text, test_data_labels):
    train_labels_list = []

    for element in train_data_labels:
        train_labels_list.append(element)

    label_one_occurences = train_labels_list.count(1)
    label_zero_occurences = train_labels_list.count(0)

    if label_one_occurences > label_zero_occurences:
        majority_baseline_class = 1
    else:
        majority_baseline_class = 0

    predictions = []
    test_labels_list = []
    for text in test_data_text:
        predictions.append(majority_baseline_class)

    test_labels_list = []
    for element in test_data_labels:
        test_labels_list.append(element)

    final_df = pd.DataFrame({"predictions": predictions})

    accuracy = accuracy_score(test_labels_list, predictions)
    print("Accuracy: ", accuracy)
    print(classification_report(test_labels_list, predictions, zero_division=1))


random_baseline(train_data=data['text'], test_data_text = test_data['text'],test_data_labels =test_data['labels'])
majority_baseline(train_data_labels=data['text'], test_data_text = test_data['text'],test_data_labels =test_data['labels'])