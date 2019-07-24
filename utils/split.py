import numpy as np
import os
from data_utils import fetch_path
import shutil
import csv

def main():
    source = os.path.join(
        'datasets',
        'kuset',
    )

    train_path = os.path.join(
        source,
        'train'
    )
    os.mkdir(train_path)

    val_path = os.path.join(
        source,
        'val'
    )
    os.mkdir(val_path)

    train_csv = os.path.join(
        source,
        'train.csv'
    )

    val_csv = os.path.join(
        source,
        'val.csv'
    )

    # Get data
    img_paths, labels = fetch_path('kuset')

    train_data = []
    val_data = []
    train_labels = []
    val_labels = []

    # Distribute data
    split = int(len(labels) * 0.8)

    indexes = np.arange(len(labels))
    np.random.shuffle(indexes)

    train_indexes = indexes[:split]
    val_indexes = indexes[split:]

    for i in train_indexes:
        img_name = img_paths[i].split('/')[-1]

        train_data.append(img_name)
        train_labels.append(labels[i])

        shutil.move(
            img_paths[i],
            os.path.join(
                train_path,
                img_name
            )
        )
    
    # Write on csv
    with open(train_csv, 'w') as csvFile:
        writer = csv.writer(
            csvFile, 
            dialect='excel' , 
            delimiter=','
        )
        
        writer.writerow(['name', 'GT', 'length'])
	    
        for name, label in zip(train_data, train_labels):
            writer.writerow([name, label, len(label)])

    csvFile.close()

    for i in val_indexes:
        img_name = img_paths[i].split('/')[-1]

        val_data.append(img_name)
        val_labels.append(labels[i])

        shutil.move(
            img_paths[i],
            os.path.join(
                val_path,
                img_name
            )
        ) 
    
    # Write on csv
    with open(val_csv, 'w') as csvFile:
        writer = csv.writer(
            csvFile, 
            dialect='excel' , 
            delimiter=','
        )
        
        writer.writerow(['name', 'GT', 'length'])
	    
        for name, label in zip(val_data, val_labels):
            writer.writerow([name, label, len(label)])

    csvFile.close()


if __name__ == '__main__':
    main()
