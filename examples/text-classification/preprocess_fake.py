import os
import csv
from tqdm import tqdm
from sklearn.model_selection import train_test_split

data_dir = "/data/firewall_pt/data/adversary_training_corpora/fake"
train_file = os.path.join(data_dir, "train_tok.csv")
test_file = os.path.join(data_dir, "test_tok.csv")

def process(fin, fout_train, fout_test):
    lines = []
    with open(fin, "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            line = line.strip().lower()
            text = line[:-1]
            label = int(line[-1])
            lines.append([text, label])
    train_data, test_data = train_test_split(lines, test_size=2000)
    with open(fout_train, "w+", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence1", "label"])
        for line in tqdm(train_data):
            writer.writerow(line)

    with open(fout_test, "w+", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence1", "label"])
        for line in tqdm(test_data):
            writer.writerow(line)
    return 

process(train_file, os.path.join(data_dir, "train_clean.csv"), os.path.join(data_dir, "test_clean.csv"))


