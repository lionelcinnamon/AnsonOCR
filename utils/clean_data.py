import os
import csv

def main():
    csv_path = os.path.join(
        'label.csv'
    )

    # Iterate across filenames
    with open(csv_path, 'r') as csvFile:
        reader = csv.reader(csvFile)
        next(reader)

        data = []
        labels = []

        for row in reader:
            img_path = os.path.join(
                'raw', 
                row[0]
            )

            if row[1] in ['aaa', 'aaaa', 'đâsdasdasd']:
                os.remove(img_path)
            else:
                data.append(row[0])
                labels.append(row[1])

    csvFile.close()

    # Write on csv
    with open(csv_path, 'w') as csvfile:
        writer = csv.writer(
            csvfile, 
            dialect='excel' , 
            delimiter=','
        )
        
        writer.writerow(['name', 'GT', 'length'])
	    
        for name, label in zip(data,labels):
            writer.writerow([name, label, len(label)])

if __name__ == '__main__':
    main()