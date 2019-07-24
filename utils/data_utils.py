import os, sys
sys.path.insert(0, os.path.dirname(sys.path[0]))
#from datasets.Mizuho import Mizuho
from PIL import Image, ImageDraw
import numpy as np
import csv

# Cut out polygon from pil_img
def _cut_out(pil_img, polygon):

    temp = np.asarray(polygon)
    xmax = np.max(temp[:,0])
    xmin = np.min(temp[:,0])
    ymax = np.max(temp[:,1])
    ymin = np.min(temp[:,1])

    # Mask for cutting
    alpha_mask = Image.new('L', pil_img.size, 0)
    draw = ImageDraw.Draw(alpha_mask)
    draw.polygon(polygon, outline=255, fill=255)

    img_crop = pil_img.crop((xmin,ymin,xmax,ymax))
    mask_crop = alpha_mask.crop((xmin,ymin,xmax,ymax))
    
    cut_out = Image.new('RGB', (xmax-xmin, ymax-ymin), (0,0,0))
    cut_out.paste(img_crop, (0,0), mask_crop)

    # Keep-aspect resize to fix height
    cut_out = cut_out.resize(
        (int(cut_out.size[0] * 48 / cut_out.size[1]), 48), 
        Image.ANTIALIAS
    )

    cut_out = cut_out.convert('L')

    return cut_out

def _preprocess(img):
    '''
    uint8 to float
    '''
    img = img.astype(float) # Switch to float
    img = img / 255         # Normalize to [0,1]

    return img

def _required_length(label):
    length = len(label)
    
    if length == 1:
        return length
    else:
        require = length
        for i in range(1, length):
            if label[i] == label[i-1]:
                require = require + 1

    return require


# Fetch data from json files
def fetch_data(dataset_name):
    # Setup dataset paths
    dataset_path = os.path.join(
        "datasets",
        dataset_name
    )

    raw_path = os.path.join(
        dataset_path,
        "raw"
    )
    Mizuho.RAW_DIR = raw_path
    Mizuho.DATASETS_DIR = dataset_path

    # Fetch data samples
    explore_df = Mizuho.explore()

    data = []
    labels = []

    # Iterate across images
    for index, file in explore_df.iterrows():

        #print(index)
        # Get textlines
        try:
            elements = Mizuho.get_element_img(
                index, 
                "textline"
            )
        except:
            continue

        img_path = explore_df.filename[index]
        img = Image.open(img_path).convert('L')

        # Iterate across textlines
        for element in elements:

            # Filter out unwanted fields
            if element['textclass'] != 'None' and "Not Valid" not in element['label']:
                ocr_label = element['label'].split('\n')[0].split('TextLine:')[-1]
                
                # Avoid empty string
                if ocr_label.strip():
                    polygon = [tuple(p) for p in element['points']]

                    cut_out_img = _cut_out(img, polygon)
                    cut_out_img = _preprocess(np.array(cut_out_img))
                    
                    # Avoid CTC running out of time error
                    if (cut_out_img.shape[1] / 4 - 1 >= _required_length(ocr_label)): 
                        data.append(cut_out_img)
                        labels.append(ocr_label)

    return (data, labels)

def fetch_path(dataset_name, delimiter=','):
    # Setup dataset paths
    csv_path = os.path.join(
        "datasets",
        dataset_name,
        "label.csv"
    )

    raw_path = os.path.join(
        "datasets",
        dataset_name,
        "raw"
    )

    # Iterate across filenames
    with open(csv_path, 'r') as csvFile:
        reader = csv.reader(
            csvFile,
            delimiter=delimiter
        )

        next(reader)

        data_paths = []
        labels = []

        for row in reader:
            img_path = os.path.join(raw_path, row[0])

            data_paths.append(img_path)
            labels.append(row[1])

    csvFile.close()

    return (data_paths, labels)

def fetch_raw_data(dataset_name, delimiter=','):
    # Setup dataset paths
    csv_path = os.path.join(
        "datasets",
        dataset_name,
        "label.csv"
    )

    raw_path = os.path.join(
        "datasets",
        dataset_name,
        "raw"
    )

    # Iterate across filenames
    with open(csv_path, 'r') as csvFile:
        reader = csv.reader(
            csvFile,
            delimiter=delimiter
        )

        next(reader)

        data_paths = []
        labels = []

        for row in reader:
            img_path = os.path.join(raw_path, row[0])
            img = Image.open(img_path).convert('L')
            img = _preprocess(np.array(img))

            data_paths.append(img)
            labels.append(row[1])

    csvFile.close()

    return (data_paths, labels)
    
