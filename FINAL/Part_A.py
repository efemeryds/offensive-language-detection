import pandas as pd
from sklearn.metrics import classification_report
import random

# TASK 1

train_data = pd.read_csv("../data/olid-train.csv", sep=',')
test_data = pd.read_csv("../data/olid-test.csv", sep=',')

print(train_data['labels'].unique())

# label: 1 -> 4400 samples
# label: 0 -> 8840 samples

# relative freq 1 -> 0.668
# relative freq 0 -> 0.332

# example 1 -> '@USER She should ask a few native Americans what their take on this is.'
# example 0 -> 'Amazon is investigating Chinese employees who are selling internal data to third-party sellers looking for an edge in the competitive marketplace. URL #Amazon #MAGA #KAG #CHINA #TCOT'


# TASK 2

def random_baseline(train_input, test_input):
    subjects = [0, 1]

    predictions = []
    for i in range(len(test_input)):
        predictions.append(random.choice(subjects))

    y_pred = list(predictions)
    y_true = list(test_data['labels'])

    accuracy_report = pd.DataFrame(classification_report(y_true, y_pred, target_names=['0', '1'], output_dict=True)).T

    tmp_dict = {"class_0_precision": accuracy_report.iloc[0, 0],
                "class_0_recall": accuracy_report.iloc[0, 1],
                "class_0_f1": accuracy_report.iloc[0, 2],
                "class_1_precision": accuracy_report.iloc[1, 0],
                "class_1_recall": accuracy_report.iloc[0, 0],
                "class_1_f1": accuracy_report.iloc[1, 2],
                "weighted_average_precision": accuracy_report.iloc[4, 0],
                "weighted_average_recall": accuracy_report.iloc[4, 1],
                "weighted_average_f1": accuracy_report.iloc[4, 2],
                "macro_average_precision": accuracy_report.iloc[3, 0],
                "macro_average_recall": accuracy_report.iloc[3, 1],
                "macro_average_f1": accuracy_report.iloc[3, 2]}

    return tmp_dict


final_eval = []
for i in range(50):
    tmp_results = random_baseline(train_data, test_data)
    final_eval.append(tmp_results)

final_df = pd.DataFrame(final_eval)
print("Random baseline")
print("class_0_precision", final_df['class_0_precision'].mean())
print("class_0_recall", final_df['class_0_recall'].mean())
print("class_0_f1", final_df['class_0_f1'].mean())
print("class_1_precision", final_df['class_1_precision'].mean())
print("class_1_recall", final_df['class_1_recall'].mean())
print("class_1_f1", final_df['class_1_f1'].mean())
print("weighted_average_precision", final_df['weighted_average_precision'].mean())
print("weighted_average_recall", final_df['weighted_average_recall'].mean())
print("weighted_average_f1", final_df['weighted_average_f1'].mean())
print("macro_average_precision", final_df['macro_average_precision'].mean())
print("macro_average_recall", final_df['macro_average_recall'].mean())
print("macro_average_f1", final_df['macro_average_f1'].mean())


def majority_baseline(train_input, test_input):
    zero_label = len(train_input[train_input['labels'] == 0])
    one_label = len(train_input[train_input['labels'] == 1])

    if zero_label > one_label:
        majority_class = 0
    else:
        majority_class = 1

    predictions = []
    tweet = []
    for i in range(len(test_input)):
        predictions.append(majority_class)
        tweet.append(test_input['text'].iloc[i])

    final_df = pd.DataFrame({"text": tweet, "predicted": predictions})
    return final_df


final_calculations = majority_baseline(train_data, test_data)

# calculate the evaluation
y_pred = list(final_calculations['predicted'])
y_true = list(test_data['labels'])

print("Majority Baseline")
print(classification_report(y_true, y_pred, labels=[0, 1]))

print("DONE")


# TASK 3

# Answers in the collab notebook, add the model file


# TASK 4

# subwords



















