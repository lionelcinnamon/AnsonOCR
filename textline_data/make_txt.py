import json
from sklearn.model_selection import train_test_split


with open('train.json', 'r') as f:
    data = json.load(f)

n = len(data.keys()) 

new_data = {}

for key, value in data.items():
    if 'daiichi' in key or 'sompo' in key:
        new_data[key] = value

data = new_data

X_train, X_val, y_train, y_val = train_test_split(list(data.keys()), list(data.values()), test_size=0.1)

with open('train.txt', 'w', encoding='utf-8') as f:
    for key, value in zip(X_train, y_train):
        if key == '' or value == '':
            continue
        try:
            f.write(key + '|' + value)
            f.write('\n')
        except:
            print(key)

with open('val.txt', 'w', encoding='utf-8') as f:
    for key, value in zip(X_val, y_val):
        if key == '' or value == '':
            continue
        try:
            f.write(key + '|' + value)
            f.write('\n')
        except:
            print(key)