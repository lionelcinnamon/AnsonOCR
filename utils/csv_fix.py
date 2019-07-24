import os
import csv

# Delimiter
DEL = chr(255)

def ONMT_to_keras_label(csv_path):
    # Read source csv
    with open(csv_path, 'r') as csvFile:
        reader = csv.reader(
            csvFile, 
            delimiter=DEL
        )
        next(reader)

        images = []
        labels = []

        for row in reader:
            images.append(row[0])

            processed_label = row[1].replace("\\;", " ").strip()
            labels.append(processed_label)

    csvFile.close()

    # Overwrite source csv
    with open(csv_path, 'w') as csvfile:
        writer = csv.writer(
            csvfile, 
            dialect='excel' , 
            delimiter=DEL
        )
        
        writer.writerow(['name', 'GT', 'length'])
        
        for name, label in zip(images,labels):
            writer.writerow([name, label, len(label)])
    
    csvFile.close()

def main():
    source_csv = os.path.join(
        'datasets',
        'kankuset',
        'val',
        'label.csv'
    )

    ONMT_to_keras_label(source_csv)

if __name__ == '__main__':
    main()
    
