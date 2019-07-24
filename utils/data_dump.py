# Configurate paths
import os, sys
sys.path.insert(
    0,
    os.path.dirname(sys.path[0])
)

from datasets.Mizuho import Mizuho
from data_utils import _cut_out, _required_length

from PIL import Image, ImageDraw
import numpy as np
import csv

# Dump textlines to images
def export_dataset(dataset_name):
    # Setup dataset paths
    dataset_path = os.path.join(
        "datasets",
        dataset_name
    )
    
    raw_path = os.path.join(
        dataset_path,
        "raw"
    )

    # Create dump path
    out_path = os.path.join(
        "datasets",
        dataset_name + "_imgs",
    )
    
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    
    out_raw_path = os.path.join(
        out_path,
        "raw"
    )

    if not os.path.exists(out_raw_path):
        os.mkdir(out_raw_path)

    Mizuho.RAW_DIR = raw_path
    Mizuho.DATASETS_DIR = dataset_path
    Mizuho.dataset_name = '000_mizuho'

    # Fetch data samples
    explore_df = Mizuho.explore()

    data = []
    labels = []

    # Iterate across images
    for index, file in explore_df.iterrows():
        # Get textlines
        try:
            elements = Mizuho.get_element_img(
                index, 
                'textline'
            )
        except:
            continue
        img_path = explore_df.filename[index]
        img = Image.open(img_path).convert('L')
        print(img_path)
        print(index)

        counter = -1
        # Iterate across textlines
        for element in elements:
            counter = counter + 1
            # Filter out unwanted fields
            if dataset_name != 'mizuho':
                valid = True
            else:
                valid = element['textclass'] != 'None' and "Not Valid" not in element['label']
            
            if valid:
                ocr_label = element['label'].split('\n')[0].split('TextLine:')[-1]
                
                # Avoid empty string
                if ocr_label.strip():
                    polygon = [tuple(p) for p in element['points']]
                    cut_out_img = _cut_out(img, polygon)
                    img_np = np.array(cut_out_img)
                    
                    # Avoid CTC running out of time error
                    if (img_np.shape[1] / 4 - 1 >= _required_length(ocr_label)): 
                        name = dataset_name + '_' + str(index) + '_' + str(counter) +'.png'
                        name_path = os.path.join(
                            out_raw_path,
                            name
                        )

                        assert not os.path.exists(name_path)
                        cut_out_img.save(name_path)

                        data.append(name)
                        labels.append(ocr_label.strip())
    
    csv_path = os.path.join(
        out_path,
        "label.csv"
    )

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

def main():
    export_dataset('mizuho')

if __name__ == '__main__':
    main()
