from wavenet import build_model
import json
import tensorflow as tf


old_config_path = './logdir/15_mar/sompo/keras/basemodel.json'
label_text = json.load(open('./datasets/label_text_4855.json', 'r'))
label_text = {int(l):t for l, t in label_text.items()}

weights_path = './logdir/dc3/train/keras/basemodel_retrain_215000.h5'
basemodel = build_model(len(label_text)+1, old_config_path)
basemodel.load_weights(weights_path)
basemodel = tf.keras.models.Model(basemodel.input, basemodel.outputs[1])
