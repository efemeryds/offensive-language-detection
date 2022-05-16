import pandas as pd
from sklearn.metrics import classification_report
import random

# 1.  Class distributions (1 point)
# Load the training set (olid-train.csv) and analyze the number of instances for each of the two
# classification labels.


# Class label, Number of instances, Relative label, frequency (%), Example tweet with this label

train_data = pd.read_csv("../data/olid-train.csv", sep=',')
test_data = pd.read_csv("../data/olid-test.csv", sep=',')

print(train_data['labels'].unique())

# label: 1 -> 4400 samples
# label: 0 -> 8840 samples

# relative freq 1 -> 0.668
# relative freq 0 -> 0.332

# example 1 -> '@USER She should ask a few native Americans what their take on this is.'
# example 0 -> 'Amazon is investigating Chinese employees who are selling internal data to third-party sellers looking for an edge in the competitive marketplace. URL #Amazon #MAGA #KAG #CHINA #TCOT'

print("DONE")


# 2. Baselines (1 point)
# Calculate two baselines and evaluate their performance on the test set (olid-test.csv):


# ● The first baseline is a random baseline that randomly assigns one of the 2 classification
# labels.


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


# evaluate -> run 50 times and take average report result

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


# ● The second baseline is a majority baseline that always assigns the majority class.
# Calculate the results on the test set and fill them into the two tables below. Round the results to
# two decimals.


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

# 3. Classification by fine-tuning BERT (2.5 points)
# Run your notebook on colab, which has (limited) free access to GPUs.
# You need to enable GPUs for the notebook:

# ● navigate to Edit → Notebook Settings
# ● select GPU from the Hardware Accelerator drop-down

# ➢ Install the simpletransformers library: !pip install simpletransformers
# (you will have to restart your runtime after the installation)

# ➢ Follow the documentation to load a pre-trained BERT model: ClassificationModel('bert',
# 'bert-base-cased')

# ➢ Fine-tune the model on the OLIDv1 training set and make predictions on the OLIDv1 test
# set (you can use the default hyperparameters). Do not forget to save your model, so that
# you do not need to fine-tune the model each time you make predictions.
# If you cannot fine-tune your own model, contact us to receive a checkpoint.

# a. Provide the results in terms of precision, recall and F1-score on the test set and provide
# a confusion matrix (2 points).


# b. Compare your results to the baselines and to the results described in the paper in 2–4
# sentences (0.5 points).


# 4. Inspect the tokenization of the OLIDv1 training set using the BERT’s tokenizer (2.5
# points) The tokenizer works with subwords. If a token is split into multiple subwords, this is
# indicated with a special symbol.

# a. Calculate how many times a token is split into subwords (hint: use
# model.tokenizer.tokenize()). (0.5 points)

# Number of tokens:

# Number of tokens that have been split into subwords:

# Example: if ‘URL’ is tokenized by BERT as ‘U’, ‘##RL’, consider it as one token

# split into two subwords.


# b. What is the average number of subwords per token? (0.5 points)
# Average number of subwords per token:


# c. Provide 3 examples of a subword split that is not meaningful from a linguistic
# perspective. (1 point)
# Which split would you expect based on a morphological analysis?

# 1. Example 1:
# 2. BERT tokenization:
# 3. Morphologically expected split:
# ....


# d. BERT’s tokenizer uses a fixed vocabulary for tokenizing any input
# (model.tokenizer.vocab). How long (in characters) is the longest subword in the
# BERT’s vocabulary? (0.5 points)
# Length of the longest subword:
# Example of a subword with max. length
