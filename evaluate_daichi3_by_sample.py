
from pyson.vision import resize_by_factor
import ipdb
import pandas
st = ipdb.set_trace
from PIL import Image
from glob import glob
import os
import matplotlib.pyplot as plt
import cv2
import json
import numpy as np
from nltk.metrics.distance import edit_distance
from tqdm import tqdm
import xlsxwriter
import time
from text_normalizer.utils import normalize
import tensorflow as tf
import argparse
from pyson.utils import *


config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras



def get_img_path(ann_path, input_dir, ext='png'):
    d = read_json(ann_path)
    img_name = os.path.basename(ann_path).split('.')[0]+'.'+ext
    img_path = os.path.join(input_dir, 'imgs', img_name)
    # import ipdb; ipdb.set_trace()
    if path_exists(img_path):
        return img_path
    
def read_qa_json(json_path):
    d = json.load(open(json_path, 'r', encoding='utf-8'))
    d = list(d['_via_img_metadata'].values())
    if len(d) > 0:
        return d[0]['regions']
    else:
        return None

def get_shape(box):
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
    return (x,y,w,h)


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

    def export_report_jp(self,images,labels,tesseract=False,google=False, name='', output_dir_image='eval_img', output_dir_report='report', max_sample_per_file=50):
        timestr = time.strftime("%Y%m%d")

        os.makedirs(output_dir_report, exist_ok=True)
        results = {}
        num_data = len(labels)
        os.makedirs(output_dir_image, exist_ok=True)
        total_rnd_field_acc = 0
        total_field = 0
        total_length = 0
        total_rnd_acc = 0
        for i, (image, label) in tqdm(enumerate(zip(images, labels))):
            label = normalize_text(label)
            pred = self.model.process(image)[0]
            
            rnd_model_err =  self.error_rate(label,pred,1)
            total_rnd_acc += (1.0 - rnd_model_err)*len(label)
            if (rnd_model_err - 0) == 0.0:
                total_rnd_field_acc += 1
            total_field += 1
            total_length += len(label)
            img_path = os.path.join(output_dir_image, '{}.png'.format(i))
            plt.imsave(img_path,image[:,:],cmap='gray')
            results[img_path] = {'pred':pred, 'rnd_model_err':rnd_model_err, 'label':label}
    
        results = sorted(results.items(), key=lambda x: x[1]['rnd_model_err'])
        json.dump(results, open(os.path.join(output_dir_report,'results.json'), 'w'))
        total_rnd_field_acc /= total_field
        total_rnd_acc /= total_length
        def mean_accuracy(results):
            accs = []
            for path, result in results:
                acc = result['rnd_model_err']
                accs.append(acc)
            return np.mean(accs)
        
        for k in range(0, len(results), max_sample_per_file):
            results_k = results[k:k+max_sample_per_file]
            if mean_accuracy(results_k) == 0: continue
            current_name = os.path.join(output_dir_report,'ocr_report_time_{}.xlsx'.format(k+500))
            print('Writing: ', current_name)
            
            workbook = xlsxwriter.Workbook(current_name)
            worksheet = workbook.add_worksheet()
            worksheet.set_default_row(50)
            worksheet.write('A1', 'Image')
            worksheet.write('B1', 'RnD OCR')
            worksheet.write('D1', 'LABEL')
            worksheet.write('E1', 'RnD OCR ERROR RATE')
            worksheet.write('G1', 'TOTAL RnD OCR ACCURACY BY CHAR')
            worksheet.write('H1', 'TOTAL Google API ACCURACY BY CHAR')
            worksheet.write('I1', 'TOTAL RnD OCR ACCURACY BY FIELD')

            for i, (img_path, v) in enumerate(results_k):
                worksheet.write('B{}'.format(i + 2), v['pred'])
                worksheet.write('E{0:.4f}'.format(i + 2), v['rnd_model_err'])
                worksheet.write('D{}'.format(i + 2), v['label'])
                worksheet.insert_image('A{}'.format(i+2), img_path,{'x_scale': 1, 'y_scale': 1})

            worksheet.write('G2', total_rnd_acc)
            worksheet.write('I2', total_rnd_field_acc)
            workbook.close()
        return  {'accuracy':total_rnd_acc ,'field_accuracy': total_rnd_field_acc, 'total_length': total_length}


import tensorflow as tf
import json
from utils.ctc_bestpath import ctcBestPath


class OCR():
    def __init__(self, weights_path, label_text_path, config=None, model=None):
        self.weights = weights_path
        self.label_text = ""
        self._load_label_text(label_text_path)
        self.nclass = len(self.label_text) + 1
        if config is not None:
            self.model = tf.keras.models.Model.from_config(config=config)
            self.model.load_weights(self.weights)
            print("OCR: Model Loaded from config!!!")
        elif model is not None:
            self.model = model
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





# config_path = './logdir/28_feb/keras/basemodel.json'
# label_text_path = './logdir/box/trained_model/label_text.json'
label_text_path = './datasets/label_text_4855.json'


#config = json.load(open(config_path, 'r'))

label_text_dict = read_json(label_text_path)
availabel_chars = list(label_text_dict.values())
valid_char_set = set(label_text_dict.values())
def is_valid_text(text):
    char_set = set(text)
    in_valid_char = char_set - valid_char_set
    if len(in_valid_char) == 0:
        return True
    return False




K = tf.keras.backend
#saver = tf.train.Saver()
#sess = K.get_session()




def pad_begin(img):
    assert img.dtype == np.uint8, img.dtype
    h, w = img.shape[:2]
    zero_pad = np.zeros([h, 8])
    if np.mean(img)>128:
        zero_pad+=255
    img = np.concatenate([zero_pad, img], axis=1)
    return img





from text_normalizer.utils.normalize import normalize_text


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--weights_path')

    args = parser.parse_args()
    ann_paths = glob(os.path.join(args.input_dir, 'ann/*.json'))
    print(len(ann_paths))
    from data_utils import extract_qa_label
    results = {}
    all_datas = []
    all_labels = []
    for ann_path in ann_paths:
        name = os.path.basename(ann_path).split('.')[0]
        img_path = get_img_path(ann_path, args.input_dir)
        if img_path:
            small_images, labels = extract_qa_label(img_path, ann_path, padding_style='small')
            all_datas.extend(small_images)
            all_labels.extend(labels)
        else:
            print('no img path: ', name)


    sorted_datas = sorted(zip(all_datas, all_labels), key=lambda x: x[0].shape[1])
    all_datas = []
    all_labels = []
    for _ in sorted_datas:
        all_datas.append(_[0])
        all_labels.append(_[1])
    mkdirs(args.output_dir)
    #ocr = OCR(args.weights_path, label_text_path, config=config)
    
    
    from wavenet import build_model
    # config = json.load(open(config_path, 'r'))
    old_config_path = './logdir/15_mar/sompo/keras/basemodel.json'
    label_text = json.load(open('./datasets/label_text_4855.json', 'r'))
    basemodel = build_model(len(label_text)+1, old_config_path)




    basemodel.load_weights('./logdir/dc3/test/keras/basemodel_retrain_485000.h5')
    model = tf.keras.models.Model(basemodel.input, basemodel.outputs[-1])

    ocr = OCR(None, label_text_path, config=None, model=model)



    
    eval_ocr = Eval(ocr)
    output = eval_ocr.export_report_jp(all_datas, all_labels, name=name, max_sample_per_file=1000,  output_dir_report=args.output_dir, output_dir_image='{}/eval_imgs'.format(args.output_dir))
    print(output)

    
    
#     import shutil
    
#     shutil.copy(args.weights_path, path_join(args.output_dir, os.path.basename(args.weights_path)))
#     shutil.copy(config_path, path_join(args.output_dir, os.path.basename(config_path)))
#     with open(path_join(args.output_dir, 'summary.json'), 'w') as f:
#         json.dump(output, f)
#     os.system('zip -r {}.zip {}'.format(args.output_dir, args.output_dir))
    
