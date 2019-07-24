from tensorflow.python.keras import backend as K
from pyson.vision import plot_images
from data_utils import *
import tensorflow as tf
from time import time
import os
import numpy as np
from pyson.utils import multi_thread, read_json
from fuzzywuzzy import fuzz
from data_utils import create_tf_dataset, run_data_init


def batch_ratio(preds, targets):
    rt = []
    for p, t in zip(preds, targets):
        r = fuzz.ratio(p, t)
        rt.append(r)
    return np.mean(rt)


def clear_output():
    os.system( 'cls' )


if __name__ == '__main__':
    label_text_path = 'data/charset_v2.json'
    label_text = {int(label): text for label, text in read_json(label_text_path).items()}
    text_label = {text:int(label) for label, text in label_text.items()}

    weights_path = './weights/AnsonOCR-v2.8.0.h5'
    LOGDIR = 'weights'
    summary_dir = os.path.join(LOGDIR, 'tb')
    test_project = 'dc3'

    batch_size = 16
    max_train_samples = None # take all
    max_test_samples = None# take all
    max_eval_steps = None

    display_freq = 10
    save_freq = 5000
    eval_freq = 1000

    K.clear_session()
    sess = K.get_session()

    phase = 'train'

    # CHANGE FILE PATH HERE
    from data_utils import load_dataset_ctc
    if phase == 'train':
        print('Training Phase')
        df_train = load_dataset_ctc(['data/train.json'],
                                   factor=4, text_label=text_label)
        print(len(df_train))
    #     df_test = df_train[df_train['path']] # .map(lambda x: '/fuse/raw/' in x)]
        df_test = load_dataset_ctc(['data/val.json'], factor=4, text_label=text_label)
    else:
        df_train = load_dataset_ctc(['data/val.json'], factor=4, text_label=text_label)
        df_test = df_train

    print(len(df_train), len(df_test))


    # ### 1.2.1 CTC dataset
    # ##### After this we have:
    # inputs-label pairs: `[inputs_train, target_train], [inputs_test, target_test]`
    #
    # initializers:  `[dataset_train_initializer, dataset_test_initializer]`
    #
    # placeholders: `[dataset_train_x, dataset_test_x]`
    #
    # feeding data: `[dataset_train_index, dataset_test_index]`

    train_tensors, train_initer = create_tf_dataset(df_train, batch_size, 'train', learn_space=True)
    test_tensors, test_initer = create_tf_dataset(df_test, batch_size, 'test', learn_space=True)
    run_data_init([train_initer], shuffle=True, sess = sess, batch_size=batch_size)

    imgs, labels, train_paths, train_losses, train_preds = sess.run(train_tensors)
    print(imgs.shape)
    imgs = np.clip(imgs, 0, 1)
    # plot_images(imgs[...,0], mxn=[3,1], dpi=200)

    # 2. Build model
    from wavenet import build_model
    basemodel = build_model(len(label_text)+1, ngf=64)
    # basemodel = Model.from_config(read_json('../lib-ocr/ocr/anson/model_config_files'))
    basemodel.summary()

    # ## 2.2 Build predict tensors
    # For ctc loss only we take the 2nd output
    #
    # For entropy loss we take the 1st output
    logits_train, preds_train = basemodel(train_tensors[0])
    preds_test = basemodel(test_tensors[0])[1]

    from pyson.losses import ctc_loss_v2

    global_step = tf.Variable(0, dtype=tf.int64, name='global_step')

    train_maxout = tf.argmax(preds_train, axis=-1)

    loss_ctc_with_argmax = ctc_loss_v2(tf.cast(train_tensors[1], tf.float32),
                                    tf.one_hot(train_maxout, len(text_label)+1))

    loss_ctc = ctc_loss_v2(train_tensors[1], preds_train)
    ops_ctc = tf.train.AdamOptimizer(0.0001).minimize(loss_ctc, global_step=global_step)

    summary = tf.Summary()

    train_writer = tf.summary.FileWriter(summary_dir,sess.graph)

    display_freq = 20
    plot_freq = 150
    batch_losses = {'loss_ctc': [], 'loss_ent':[], 'accuracy_train':[], 'accuracy_loc':[]}

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
    os.makedirs(LOGDIR, exist_ok=True)
    print('Create logdir:', LOGDIR)

    checkpoint = tf.train.latest_checkpoint(LOGDIR)
    try:
        saver.restore(sess, checkpoint)
    except Exception as e:
        print('Cannot restore')
        sess.run(tf.global_variables_initializer())
        os.system('rm -r {}/*'.format(summary_dir))

    run_data_init([train_initer, test_initer], shuffle=False, sess=sess, batch_size=batch_size)

    basemodel.load_weights(weights_path)

    init_step = sess.run(global_step)

    start = time()
    init_step = sess.run(global_step)
    print('INFO: Init_step:{}'.format(init_step))
    while True:
        try:
            g_step = sess.run(global_step)
            train_dict = {
                'ops_ctc': ops_ctc,
                'loss_ctc':loss_ctc,
                'global_step':global_step,
            }

            if g_step % display_freq == 0:
                train_dict['preds_train'] = preds_test
                train_dict['target_train'] = test_tensors[1]

            if g_step % plot_freq == 0:
                train_dict['inputs_train'] = test_tensors[0]
                train_dict['preds_train'] = preds_test
                train_dict['target_train'] = test_tensors[1]

            results = sess.run(train_dict)
            mean_ctc = np.mean(results['loss_ctc'])
            batch_losses['loss_ctc'].append(mean_ctc)

            if g_step % display_freq == 0:
                pred_decoded_train = convert_batch_int_to_text(np.argmax(results['preds_train'], -1), label_text)
                label_decoded_train = convert_batch_int_to_text(results['target_train'], label_text, repeat_colapse=False)
                batch_losses['accuracy_train'].append(batch_ratio(pred_decoded_train, label_decoded_train))

                speed = ((g_step-init_step)*batch_size) / (time()-start)
                summary.value.add(tag='Loss Ctc', simple_value=batch_losses['loss_ctc'][-1])

                summary.value.add(tag='train Accuracy', simple_value=batch_losses['accuracy_train'][-1])

                sum_str = '\rGlobal_step:{}\tLoss-ctc:{:0.5f}\tLoss-ent:{:0.5f}\tAccuracy-train:{:0.2f}\tSpeed:{:0.2f} '.format(
                                                results['global_step'],
                                                np.mean(batch_losses['loss_ctc'][-1000:]),
                                                -1,#np.mean(batch_losses['loss_ent'][-1000:]),
                                                np.mean(batch_losses['accuracy_train'][-1000:]),
                                                speed,

                )

                print(sum_str, end='')
                train_writer.add_summary(summary, g_step)

            if g_step % plot_freq == 0:
                clear_output()
                # plot_images(np.clip(results['inputs_train'][...,0], 0, 1), dpi=100)

                pred_decoded = convert_batch_int_to_text(np.argmax(results['preds_train'], -1), label_text)
                label_decoded = convert_batch_int_to_text(results['target_train'], label_text, False)

                accuracy = batch_ratio(pred_decoded, label_decoded)
                print('Batch accuracy:', accuracy)
                for a, b in zip(pred_decoded[:16], label_decoded[:16]):
                    print('\n',a,'\n',b,'-----------------')

            if g_step % save_freq == 0 and g_step > 100:
                save_path = os.path.join(LOGDIR, 'keras','basemodel_retrain_{}.h5'.format(g_step))
                basemodel.save_weights(save_path)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                saver.save(sess, os.path.join(LOGDIR, 'ckpt'), global_step=global_step)
                print('Save keras model at:', save_path)

        except KeyboardInterrupt:
            save_path = os.path.join(LOGDIR, 'keras','basemodel_retrain_{}.h5'.format(g_step))
            saver.save(sess, os.path.join(LOGDIR, 'ckpt'), global_step=global_step)
            break

        except Exception as e:
            print('Exeption: {}'.format(str(e)))
            run_data_init([train_initer, test_initer], shuffle=True, sess=sess, batch_size=batch_size)
