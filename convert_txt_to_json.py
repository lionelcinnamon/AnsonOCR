# from pyson.losses import suppervise_ctc_loss, ctc_loss
from tensorflow.python.keras.models import Sequential,Model
from text_normalizer.utils.normalize import normalize_text
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import *
from pyson.utils import memoize, show_df
from pyson.vision import plot_images
from data_utils import *
#import data_utils
import tensorflow as tf
from tqdm import tqdm
from time import time
import os, sys
import numpy as np
import pandas
from glob import glob
from pyson.utils import multi_thread
from fuzzywuzzy import fuzz
import cv2
# from data_utils import run_data_init
print(tf.__version__)

## UTILS
def printr(string):
    sys.stdout.write("\r\033[2K")
    sys.stdout.write(string)
    sys.stdout.flush()
    
def batch_ratio(preds, targets):
    rt = []
    for p, t in zip(preds, targets):
        r = fuzz.ratio(p, t)
        rt.append(r)
    return np.mean(rt)
def vocab_txt_to_json(txt_path, json_path):
    vocab_json = pandas.DataFrame(columns=['label'])
    with open(txt_path, 'r', encoding='utf-8') as rf:
            for line in rf.read():
                vocab_json = vocab_json.append({'label': line}, ignore_index=True)
    vocab_json = vocab_json.to_dict()['label']
    with open(json_path, 'w') as f:
        json.dump(vocab_json, f)

def label_txt_to_json(txt_path, json_path, num_sample=None):
    label_json = pandas.DataFrame(columns=['text', 'path', 'name'])
    i_sample = 0
    print('Create {}...'.format(json_path))
    with open(txt_path, 'r', encoding='utf-8') as rf:  
        for line in rf.readlines():
            try:
                index = line.strip().index("|")
                impath = line[0:index].strip()
                imlabel = line[index + 1:].strip()
                ## add to json
                # add weight and height
#                 shape = cv2.imread(os.path.join(data_root, impath)).shape
# , 'height': shape[0], 'width': shape[1]
                
                label_json = label_json.append({'text': imlabel, 'name': impath, 'path': os.path.join(data_root, impath)}, ignore_index=True)
                ## display
                i_sample += 1
                if i_sample % 100 == 0:
                    printr('[{}]'.format(i_sample))
                ## stop at num_sample
                if num_sample is not None: ## there is limited number of lines
                    if i_sample > num_sample:
                        break
            except:
                print(line)
            
    print('--> done.\n')
    label_json = label_json.to_dict()
    with open(json_path, 'w') as f:
        json.dump(label_json, f)
		
# -------- ANSON driver 		
# make json vocab file
# CHANGE FILE PATH HERE
vocab_text_path = 'configs/invoice_charset_v2.txt'
vocab_txt_to_json(vocab_text_path, 'data/charset_v2.json')
# make json label file
data_root = 'textline_data/Japanese'
train_label_text_path = 'textline_data/train.txt'
val_label_text_path = 'textline_data/val.txt'
label_txt_to_json(train_label_text_path, 'data/train.json')
label_txt_to_json(val_label_text_path, 'data/val.json')

# ## -----------CONFIG
# ## make vocab
# label_text_path = 'charset_v2.json'
# label_text = {int(label): text for label, text in read_json(label_text_path).items()}
# text_label = {text:int(label) for label, text in label_text.items()}

# ## To correct
# weights_path = './logdir/23_may/sompo/keras/basemodel_retrain_160000.h5'

# LOGDIR = 'logdir/dc3/train-ngf32/'
# summary_dir = os.path.join(LOGDIR, 'tb')
# test_project = 'dc3'

# batch_size = 16
# max_train_samples  = None # take all
# max_test_samples = None# take all
# max_eval_steps = None

# display_freq = 10
# save_freq = 5000
# eval_freq = 1000

# ## -------------- SESSION
# #K.clear_session()
# #sess = K.get_session()
# ## Load data
# ## load ctc dataset
# label_path = 'train.json'
# df_train = load_dataset_ctc([label_path], factor=4, text_label=text_label)
# label_path = 'val.json'
# df_test = load_dataset_ctc([label_path], factor=4, text_label=text_label)
# print(len(df_train), len(df_test))