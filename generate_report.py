from glob import glob
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2
import json
import json
from utils.ctc_bestpath import ctcBestPath
import numpy as np
from nltk.metrics.distance import edit_distance
from tqdm import tqdm
import xlsxwriter
import time
import tensorflow as tf
import argparse
import pandas
from text_normalizer.utils.normalize import normalize_text



parser = argparse.ArgumentParser()
parser.add_argument('--input_dir')
parser.add_argument('--output_dir', default='report')
parser.add_argument('--weights_path', default = './logdir/14_mar/keras/basemodel_retrain_250000.h5')
parser.add_argument('--config_path', default = './logdir/14_mar/keras/basemodel.json')
parser.add_argument('--label_text_path', default = './datasets/label_text_4855.json')
args = parser.parse_args()


def get_img_path(ann_path, input_dir, ext='png'):
    d = read_json(ann_path)
#     img_name = os.path.basename(ann_path).split('.')[0]+'.'+ext
#     img_path = os.path.join(input_dir, 'imgs', img_name)
    import ipdb; ipdb.set_trace()
    if os.path.exists(img_path):
        return img_path
    
def read_json(json_path):
    d = json.load(open(ann_path, 'r', encoding='utf-8'))
    return list(d['_via_img_metadata'].values())[0]['regions']

def get_shape(box, origin_image):
    """
        Arguments:
            box: QA label dictionary of {'shape_attributes':something, 'region_attributes':something}
            origin_image: full page image document
        return:
            ndarray: small image contain the text 
    """
    shape_attribute = box['shape_attributes']
    shape_type = shape_attribute['name']
    if shape_type == 'polygon' or shape_type=='polyline':
        all_points_x = shape_attribute['all_points_x']
        all_points_y = shape_attribute['all_points_y']
        x = np.min(all_points_x)
        max_x = np.max(all_points_x)
        y = np.min(all_points_y)
        max_y = np.max(all_points_y)
        w = max_x - x
        h = max_y - y
    elif shape_type == 'rect':
        x, y = shape_attribute['x'], shape_attribute['y']
        w, h = shape_attribute['width'], shape_attribute['height']
        
    else:
        raise Exception('shape_type : {} not yet process'.format(shape_type))
    
    return (x,y,w,h), origin_image[y:y+h, x:x+w]




class Eval(object):
    def __init__(self,model):
        self.model = model
        
    def error_rate(self,label,pred,batch_size):
        label = np.array(label).reshape(-1,)
        pred = np.array(pred).reshape(-1,)
        mean_ed = 0.0
        for i in range(batch_size):
            if(len(label[i])) != 0:
                mean_ed += float(edit_distance(pred[i].replace(" ", ""), label[i].replace(" ", "")))/len(label[i])
        mean_ed /= batch_size
        return min(mean_ed, 1)

    def export_report_jp(self,data,label,tesseract=False,google=False, name='', output_dir_image='eval_img', output_dir_report='report'):
        timestr = time.strftime("%Y%m%d")

        os.makedirs(output_dir_report, exist_ok=True)
        workbook = xlsxwriter.Workbook(os.path.join(output_dir_report,'ocr_report_{}_time_{}.xlsx'.format(name,timestr)))
        worksheet = workbook.add_worksheet()
        worksheet.set_default_row(50)
        worksheet.write('A1', 'Image')
        worksheet.write('B1', 'RnD OCR')
        worksheet.write('D1', 'LABEL')
        worksheet.write('E1', 'RnD OCR ERROR RATE')
        worksheet.write('G1', 'TOTAL RnD OCR ACCURACY BY CHAR')
        worksheet.write('H1', 'TOTAL Google API ACCURACY BY CHAR')
        worksheet.write('I1', 'TOTAL RnD OCR ACCURACY BY FIELD')
        worksheet.write('J1', 'TOTAL Google API ACCURACY BY FIELD')
        total_rnd_acc = 0
        total_google_acc = 0
        total_length = 0
        total_rnd_field_acc = 0
        total_google_field_acc = 0
        total_field = 0
        total_rnd_fscore = 0
        total_google_fscore = 0
        if google:
            worksheet.write('C1', 'Google Vision API')
            worksheet.write('F1', 'Google Vision API ERROR RATE')
        for i in range(len(data)):
            pred = self.model.process(data[i])[0]
            rnd_model_err =  self.error_rate(label[i],pred,1)
            total_rnd_acc += (1.0 - rnd_model_err)*len(label[i])
            if (rnd_model_err - 0) == 0.0:
                total_rnd_field_acc += 1
            total_field += 1
            total_length += len(label[i])
            worksheet.write('B{}'.format(i + 2), pred)
            worksheet.write('E{0:.4f}'.format(i + 2), rnd_model_err)
            os.makedirs(output_dir_image, exist_ok=True)
            plt.imsave(os.path.join(output_dir_image,"{}.png").format(i),data[i][:,:],cmap='gray')
            if google:
                google_pred = google_vision(os.path.join("eval_img","{}.png").format(i))
                google_err = self.error_rate(label[i], google_pred, 1)
                worksheet.write('C{}'.format(i + 2), google_pred)
                worksheet.write('F{0:.4f}'.format(i + 2), google_err)
                if(google_err - 0) == 0.0:
                    total_google_field_acc += 1
                total_google_acc += (1.0 - google_err)*len(label[i])
            worksheet.write('D{}'.format(i + 2), label[i])
            worksheet.insert_image('A{}'.format(i+2),os.path.join("eval_img","{}.png".format(i)),{'x_scale': 0.4, 'y_scale': 0.4})
        
        total_rnd_acc /= total_length
        total_google_acc /= total_length
        total_rnd_field_acc /= total_field
        total_google_field_acc /= total_field
        worksheet.write('G2', total_rnd_acc)
        worksheet.write('H2', total_google_acc)
        worksheet.write('I2', total_rnd_field_acc)
        worksheet.write('J2', total_google_field_acc)
        workbook.close()
        return  {'accuracy':total_rnd_acc, 'field_accuracy': total_rnd_field_acc, 'total_length': total_length}
    
    
class OCR():
    def __init__(self, weights_path, label_text_path, config=None):
        self.weights = weights_path
        self.label_text = ""
        self._load_label_text(label_text_path)
        self.nclass = len(self.label_text) + 1
        if config is not None:
            self.model = tf.keras.models.Model.from_config(config=config)
            self.model.load_weights(self.weights)
            print("OCR: Model Loaded from config!!!")
        else:
            self._build_graph(self.nclass)

    def _load_label_text(self, label_text):
        with open(label_text) as f:
            self.label_text = json.load(f)
            self.label_text = {int(k): v for k, v in self.label_text.items()}

    def _build_graph(self, n_class):
        _, self.model, _ = libmodel.cnn_lstm_ctc_model(height=48, nclass=n_class)
        self.model.load_weights(self.weights)
        print("OCR: Model Loaded from code!!!")
        # self.model.summary()

    def _preprocess(self, img, hsize=48):
        if type(img) is np.ndarray:
            img = Image.fromarray(img)
        img = img.convert('L')
        w, h = img.size
        img = img.resize((int(hsize / h * w), hsize), Image.ANTIALIAS)
        w, h = img.size
        img_4d = np.array(img).reshape(-1, h, w, 1)
        return img_4d / 255

    def process(self, X):
        result = []

        out = self.model.predict(self._preprocess(X))
        
        for i in range(out.shape[0]):
            decoded_output = ctcBestPath(out[i], self.label_text)
            decoded_output = normalize_text(decoded_output)
            result.append(decoded_output)

        return result



if __name__ == '__main__':
    config = json.load(open(args.config_path, 'r'))
    ocr = OCR(args.weights_path, args.label_text_path, config=config)

    label_text_dict = json.load(open(args.label_text_path, 'r'))
    availabel_chars = list(label_text_dict.values())
    valid_char_set = set(label_text_dict.values())




    eval_ocr = Eval(ocr)
    ann_paths = glob(os.path.join(args.input_dir, 'ann','*.json'))
    print(len(ann_paths))

    results = {}
    for ann_path in ann_paths:
        name = os.path.basename(ann_path).split('.')[0]
        img_path = get_img_path(name, os.path.join(args.input_dir, 'imgs'))
        if img_path:
            print('Exporting excel file for:', name)
            data = []
            labels = []
            img = cv2.imread(img_path)
            lst_boxes = read_json(ann_path)
            for box in lst_boxes:
                (x,y,w,h), small_image = get_shape(box, img)
                label = box['region_attributes']['label']
                data.append(small_image)
                norm_label = normalize_text(label)
                labels.append(norm_label)
            output = eval_ocr.export_report_jp(data, labels, name=name,
                                              output_dir_report=args.output_dir)
            results[name] =output
        else:
            print('no img path: ', name)



    summary_path = os.path.join(args.output_dir, 'summary.xlsx')

    df = pandas.DataFrame.from_dict(results)
    df = df.transpose()
    print(np.mean(df['accuracy']), np.mean(df['field_accuracy']))
    df.to_excel(summary_path)