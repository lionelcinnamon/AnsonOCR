from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks

import os
import editdistance
import itertools
import pylab , re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from nltk.metrics.distance import edit_distance
from PIL import Image
matplotlib.rc('font', family='Yu Mincho')
OUTPUT_DIR = 'checkpoint'
def labels_to_text(labels,alphabet):
    ret = []
    for c in labels:
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)

def decode_batch(test_func, word_batch,labels_text):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j,2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best,labels_text)
        ret.append(outstr)
    return ret

class EvalCallback(keras.callbacks.Callback):

    def __init__(self,run_name, test_func, text_img_gen, labels_text={}, num_display_words=8):
        self.test_func = test_func
        self.output_dir = os.path.join(
            OUTPUT_DIR, run_name)
        self.text_img_gen = text_img_gen
        self.num_display_words = num_display_words
        self.labels_text = labels_text
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    # def show_edit_distance(self, num):
        # num_left = num
        # mean_norm_ed = 0.0
        # mean_ed = 0.0
        # while num_left > 0:
        #     word_batch = next(self.text_img_gen)[0]
        #     num_proc = min(word_batch['the_input'].shape[0], num_left)
        #     decoded_res = decode_batch(self.test_func, word_batch['the_input'][0:num_proc],self.labels_text)
        #     for j in range(num_proc):
        #         edit_dist = editdistance.eval(decoded_res[j], word_batch['source_str'][j])
        #         mean_ed += float(edit_dist)
        #         mean_norm_ed += float(edit_dist) / len(word_batch['source_str'][j])
        #     num_left -= num_proc
        # mean_norm_ed = mean_norm_ed / num
        # mean_ed = mean_ed / num
        # print('\nOut of %d samples:  Mean edit distance: %.3f Mean normalized edit distance: %0.3f'
        #       % (num, mean_ed, mean_norm_ed))

    def show_edit_distance(self,word_batch):
        eval_data = word_batch
        num_sample = len(eval_data['label_length'])
        mean_ed = 0.0
        w,h,c = eval_data['the_input'][0].shape
        for i in range(num_sample):
            p_label = decode_batch(self.test_func, eval_data['the_input'][i].reshape(1,w,h,c), self.labels_text)[0]
            t_label = eval_data['source_str'][i]
            p_label = re.sub(r'\s', '', p_label)
            t_label = re.sub(r'\s', '', t_label)
            #print("Tr= %s \nPr= %s" %(t_label,p_label))
            mean_ed += float(edit_distance(p_label,t_label))
        mean_ed /= num_sample
        print('Out of %d samples:  Gale&Andy Model Mean edit distance: %.5f'% (num_sample, mean_ed))

    def on_epoch_end(self, epoch, logs={}):
        period = 3
        if (epoch % period == 0):
            loss = logs['val_loss']
            print("EVAL: save checkpoint weights.%05d_v:%.3f.h5" % (epoch,loss))
            self.model.save_weights(os.path.join(OUTPUT_DIR, 'weights.%05d_v:%.3f.h5' % (epoch,loss)))
        word_batch = next(self.text_img_gen)[0]
        self.show_edit_distance(word_batch)
        res = decode_batch(self.test_func, word_batch['the_input'][0:self.num_display_words],self.labels_text)
        if word_batch['the_input'][0].shape[0] < 256:
            cols = 2
        else:
            cols = 1
        plt.figure(figsize=(10, 10))
        for i in range(self.num_display_words):
            plt.subplot(self.num_display_words // cols, cols, i + 1)
            the_input = Image.fromarray(word_batch['the_input'][i][:, :, 0]*255)
            plt.imshow(the_input, cmap='gray')
            plt.xlabel('Truth = \'%s\'\nDecoded = \'%s\'' % (word_batch['source_str'][i], res[i]))
        plt.savefig(os.path.join(self.output_dir, 'e%03d.png' % (epoch)))
        plt.close()
