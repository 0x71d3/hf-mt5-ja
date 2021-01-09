import csv
import random
import re

random.seed(42)


def clean(text):
    mention = re.compile(r'^@\w+\s+')
    while mention.search(text):
        text = mention.sub('', text)
    return text


pairs = []
with open('pairs.tsv') as f:
    reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        pairs.append((clean(row[0]), clean(row[1])))

num_samples = len(pairs)

num_val = num_samples // 20
num_train = num_samples - 2 * num_val

random.shuffle(pairs)

split_pairs = {
    'train': pairs[:num_train],
    'val': pairs[num_train:num_train+num_val],
    'test': pairs[num_train+num_val:]
}

for split, pairs in split_pairs.items():
    with open(split + '.tsv', 'w') as f:
        for pair in pairs:
            f.write('\t'.join(pair) + '\n')
