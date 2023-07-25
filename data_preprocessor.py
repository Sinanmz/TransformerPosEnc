from datasets import load_dataset
from transformers import AutoTokenizer


dataset = load_dataset("glue", "sst2")

x_train = dataset['train']['sentence']
y_train = dataset['train']['label']
x_test = dataset['validation']['sentence']
y_test = dataset['validation']['label']

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

attention_mask_train = tokenizer(x_train, padding=True, truncation=True)["attention_mask"]
attention_mask_test = tokenizer(x_test, padding=True, truncation=True)["attention_mask"]

x_train = tokenizer(x_train, padding=True, truncation=True)["input_ids"]
x_test = tokenizer(x_test, padding=True, truncation=True)["input_ids"]


for i in range(len(x_train)):
    attention_mask_train[i] = attention_mask_train[i][1:]
    x_train[i] = x_train[i][1:]
    x_train[i][x_train[i].index(102)] = 101

for i in range(len(x_test)):
    attention_mask_test[i] = attention_mask_test[i][1:]
    x_test[i] = x_test[i][1:]
    x_test[i][x_test[i].index(102)] = 101


# Get all tokens present in training samples
unique_tokens = set([num for sample in x_train for num in sample])

# Building a dictionary which maps bert tokenizer token_ids to our token_ids
my_token_id_dict = {101:101, 100:100, 0:0}  # [UNK] : 100
                                            # [CLS] : 101
v = 0                                       # [PAD] : 0
for i in unique_tokens:
    if v == 0 or v == 100 or v == 101:
        v += 1
    if i == 0 or i == 100 or i == 101:
        continue
    my_token_id_dict[i] = v
    v += 1


# Replacing the tokens_ids in the train samples with our token_ids
for sample in x_train:   
    for i in range(len(sample)):
        sample[i] = my_token_id_dict[sample[i]]

# Replacing the tokens_ids in the test samples with our token_ids and those that our model hasn't seen in training phase with [UNK] token
for sample in x_test:   
    for i in range(len(sample)):
        if sample[i] in my_token_id_dict.keys():
            sample[i] = my_token_id_dict[sample[i]]
        else:
            sample[i] = 100

vocab_size = [len(my_token_id_dict)]


import csv

# Storing the training data in a CSV file
with open(r'.\preprocessed_data\x_train.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(x_train)

with open(r'.\preprocessed_data\attention_mask_train.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(attention_mask_train)

with open(r'.\preprocessed_data\y_train.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for number in y_train:
        writer.writerow([number])

# Storing the test data in a CSV file
with open(r'.\preprocessed_data\x_test.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(x_test)

with open(r'.\preprocessed_data\attention_mask_test.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(attention_mask_test)

with open(r'.\preprocessed_data\y_test.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for number in y_test:
        writer.writerow([number])

# Storing the vocab size in a CSV file
with open(r'.\preprocessed_data\vocab_size.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for number in vocab_size:
        writer.writerow([number])
