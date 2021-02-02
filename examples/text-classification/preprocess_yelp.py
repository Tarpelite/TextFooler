import os
import csv
from tqdm import tqdm

data_dir = "/data/firewall_pt/data/adversary_training_corpora/yelp"
train_file = os.path.join(data_dir, "train_tok.csv")
test_file = os.path.join(data_dir, "test_tok.csv")

def process(fin, fout):
    lines = []
    with open(fin, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for line in tqdm(f.readlines()):
            line = line.strip().lower()
            text = line[:-1]
            label = int(line[-1])
            lines.append([text, label])
    with open(fout, "w+", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence1", "label"])
        for line in tqdm(lines):
            writer.writerow(line)
    return 

process(train_file, os.path.join(data_dir, "train_clean.csv"))
process(test_file, os.path.join(data_dir, "test_clean.csv"))

