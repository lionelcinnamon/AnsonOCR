import json
from pyson.vision import resize_by_factor
import os
from pyson.utils import memoize
from tqdm import tqdm
import numpy as np
import cv2
from pyson.utils import read_json
import pandas
import tensorflow as tf
from pyson.utils import multi_thread


def shuffle_by_batch(lst, batch_size):
    """
        inputs: a list of data to be shuffle
        return: shuffled batch
    """
    num_batch = len(lst)//batch_size
    idxs = list(range(num_batch*batch_size))
    idxs = np.reshape(idxs, [num_batch, batch_size])
    np.random.shuffle(idxs)
    idxs = idxs.reshape([-1])
    print(idxs[:100])
    return [lst[i] for i in idxs]

def read_image(image_path, adaptive_padding=True):
    """
        Read image and normalize to [0, 1]
        
        inputs:
            image_path
        returns:
            nr.array of shape [h,w,1]
    """
    img = cv2.imread(image_path, 0) / 255.0
    img = resize_by_factor(img, 48/img.shape[0])
    img_w = 64*(1+img.shape[1]//64)
    
    pad = np.zeros([48, img_w, 1], dtype=np.float32)
    if adaptive_padding:
        pad += -1#np.float(np.mean(img) > .5)
        
    pad[:,:img.shape[1], 0] = img
    return pad

def concat_full_page(ann_path):
    import json
    """
        Inputs:
            ann_path: anotation path
            image_path: image path
        Returns:
            np.array of shape: 48xNonex1
            location for each small images
            string for each
    """
    def load_qa_json(json_path):
        d = json.load(open(json_path, 'r', encoding='utf-8'))
        d = list(d['_via_img_metadata'].values())
        if len(d) == 0:
            print(json_path)
            return None
        else:
            d = d[0]
        return (d['regions'], os.path.basename(d['filename']))

    def get_shape(box, origin_image, img_h, img_w ):
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


        # loc is the relative position of the text-line on the page
        loc = (x+w//2,y+h//2,img_w, img_h)
        
        return loc, origin_image[y:y+h, x:x+w]
        

    lst_boxes, file_name = load_qa_json(ann_path)
    if type(ann_path) is bytes:
        ann_path = ann_path.decode('utf-8')
    image_path = os.path.join(os.path.dirname(ann_path).replace('ann', 'imgs'), file_name)
    origin_img = read_image(image_path)

    
    images = []
    labels = []
    locs = []
    img_h, img_w = origin_img.shape[:2]
    for box in lst_boxes:
        label = box['region_attributes']['label']
        labels.append(label)
        location, img = get_shape(box, origin_img, img_h, img_w)
        
        images.append(img)
        locs.append(location)
    
    return images, labels, locs
    
    
    
def extract_qa_label(image_path, ann_path, min_size=5, ignore_label='<?>', padding_style='small'):
    """
        inputs:
            image_path
            ann_path
        return imgs, labels
    """
    
    def normalize_shape(shape, median_height, image_shape, offset_x = 8):
        x,y,w,h=shape
        cx = x+w//2
        cy = y+h//2
        
        new_h = max(median_height, h)
        y = cy - new_h // 2
        
        if padding_style=='small':
            if min(h, w) < median_height:
                new_w = w + offset_x*2
                x = cx-new_w//2
            else:
                offset_x /= 2
                new_w = w + offset_x*2
                x = cx-new_w//2
        else:# padding_style == 'all':
            new_w = w + offset_x*2
            x = cx-new_w//2

        return (x,y,new_w, new_h)
    
    def get_shape(box):
        """
            Arguments:
                box: QA label dictionary of \
                    {'shape_attributes':something, 'region_attributes':something}
                origin_image: full page image document
                min_size: the minimal size of image, if smaller then ignore,\
                            eg. image with shape[48,3] would be ignored
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

    import json
    """
        Arguments:
            image_path: path to raw image
            ann_path: path to qa annotation json file
        Returns:
            images: list of images with padded 10 at the begining
            labels: list of labels
    """
    
    def read_qa_json(json_path):
        d = json.load(open(json_path, 'r', encoding='utf-8'))
        d = list(d['_via_img_metadata'].values())
        if len(d) > 0:
            return d[0]['regions']
        else:
            return None
    
    images = []
    labels = []
    img = cv2.imread(image_path, 0)
    assert os.path.exists(image_path), image_path
    lst_boxes = read_qa_json(ann_path)
    if lst_boxes is None: return
    shapes = [get_shape(box) for box in lst_boxes]
    median_height = np.median(np.array(shapes)[:,3])
    shapes = [normalize_shape(shape, median_height, img.shape) for shape in shapes]
    for box, shape in zip(lst_boxes, shapes):
        x,y,w,h = [int(_) for _ in shape]
        small_image = img[y:y+h, x:x+w]
        label = box['region_attributes']['label']
        img_min_size = min(small_image.shape[:2])
        if img_min_size > min_size and not ignore_label in label:
            images.append(small_image)
            
            labels.append(label)
            
    return images, labels



#def check_valid(text, valid_text):
#    if text is None or len(text) == 0:
#        return False
#    text = text.strip()
#    for ch in text:
#        if not ch in valid_text:
#            return False
#    return True


def build_qa_dataset(mode, output_dir, padding_style, project_name='dc3'):
    """
        Arguments:
            mode: [train, test]
            output_dir: directory to dump small images
            padding_style: [small, all]
        
    """
    def get_img_path(name):
        for img_path in img_paths:
            img_name = os.path.basename(img_path).split('.')[0]
            if img_name == name:
                return img_path
        return None
    
    import json
    from glob import glob
    
    img_paths = glob('{}/{}/imgs/*.*'.format(project_path, mode))
    ann_paths = glob('{}/{}/ann/*.json'.format(project_path, mode))
    
    
    label_csv_path = os.path.join(output_dir, 'label.json')
    os.makedirs(os.path.join(output_dir, 'raw'), exist_ok=True)
    
    output_dict = {}
    from tqdm import tqdm
    for ann_path in tqdm(ann_paths):
        name = os.path.basename(ann_path).split('.')[0]
        img_path = get_img_path(name)
        results = extract_qa_label(img_path, ann_path, padding_style=padding_style)
        if results:
            images, labels = results
            for i, (small_image, label) in enumerate(zip(images, labels)):
                small_name = '{}_{}.png'.format(name, i)
                small_path = os.path.join(output_dir, 'raw', small_name)
                cv2.imwrite(small_path, small_image)
                output_dict[small_name] = {'text':label, 'width':small_image.shape[1]}
                
    json.dump(output_dict, open(label_csv_path, 'w'))

def convert_string_to_int(input_string):
    """
        inputs: string
        returns: something like [1,2,3,4]
    """
    input_string = input_string
    rt = []
    for ch in input_string:
        if ch in text_label.keys():
            lb = text_label[ch]
        else:
            raise Exception('{} does not exist, textline: {}'.format(ch, input_string))
        rt.append(lb)
    return rt

    
    
def create_tf_dataset_with_precise_location(df, batch_size, label_text, dataset_name=None):
    """
        inputs: data frame
        return: 1.dataset of (image, aligment_ctc, target_int)
                2.placeholder for feeding data
                3.data to be feed placeholder
                
    """
    assert 'img_path' in df.keys()
    def map_func(index):
        if type(index) is bytes:
            index = index.decode('utf-8')
        location_dict = eval(df['location'][index])
        start_location = 0
        text = ''.join([_[-1] for _ in location_dict])
        strip_text = text.strip()
        s_strip = text.index(strip_text)
        e_strip = s_strip+len(strip_text) 
        
        img_path = df['img_path'][index]
        img = read_image(img_path)
        img = img.astype(np.float32)


        max_x2 = max([_[2] for _ in location_dict])
        label_mask = np.zeros([ max_x2 // 4], dtype=np.int32)
        label_mask += len(label_text)
        label_int = []
        for x1,y1,x2,y2, ch in location_dict:
            start_location = x1//4
            end_location = x2//4
            norm_ch = text_label[ch]
            label_int.append(norm_ch)
            label_mask[(start_location+end_location)//2] = norm_ch

        return img, label_mask,  np.array(label_int, dtype=np.int32)


    index = [str(_) for _ in df.index]
    data_placeholder = tf.placeholder(tf.string, [None], name=dataset_name)
    dataset = tf.data.Dataset.from_tensor_slices(data_placeholder)
    dataset = dataset.map(lambda item1: tf.py_func(
              map_func, [item1], [tf.float32,  tf.int32, tf.int32]), num_parallel_calls=8)

    dataset = dataset.padded_batch(batch_size, padded_shapes=([48, None, 1], [None], [None]), 
                                        padding_values=(-1.,  len(label_text),  -1))

    dataset = dataset.prefetch(100)
    
    prefetch_op = tf.data.experimental.prefetch_to_device(device="/gpu:0")

    dataset = dataset.apply(prefetch_op)
    iterator = dataset.make_initializable_iterator()
    inputs_loc, targets_loc_1, targets_loc_2 = iterator.get_next()
    input_tensors = (inputs_loc, targets_loc_1, targets_loc_2)
    return input_tensors, {'initer':iterator.initializer, 'x':data_placeholder, 'index':index}
    
    
    
def check_load_able(df, map_fn):
    mapables = []
    for index in tqdm(df.index):
        try:
            map_fn(index)
            mapables.append(True)
        except Exception as ex:
            mapables.append(str(ex))
            
    df['mapable'] = mapables
    return df

class DataIniter(object):
    def __init__(self, initializer, index, data_placeholder):
        self.initializer = initializer
        self.index = index
        self.data_placeholder = data_placeholder
        
    
def create_tf_dataset(df, batch_size, dataset_name=None, learn_space=False, mode='train'):
    """
        inputs: data frame
        return: 1.dataset of (image, aligment_ctc, target_int)
                2.placeholder for feeding data
                3.data to be feed placeholder
    """
    if mode == 'train':
        needed_cols = ['path', 'text_int']
    elif mode == 'test':
        needed_cols = ['path']
    
    for needed_col in needed_cols:
        assert needed_col in df.keys(), needed_col
    
    def map_fn_ctc(index):
        if type(index) is bytes:
            index = index.decode('utf-8')
        img_path = df['path'][index]
        img = read_image(img_path)
        img = img.astype(np.float32)
        if mode == 'train':
            label_int  = df['text_int'][index]
            latest_loss = 32e5
            latest_pred = [-1]
            latest_pred_path = img_path+'.json'

            if os.path.exists(latest_pred_path):
                try:
                    with open(latest_pred_path, 'r') as f:
                        d = json.load(f)
                        latest_loss = d['loss']
                        latest_pred = d['latest_pred']
                except:
                    print('Error loading json:', latest_pred_path)

            return img, np.array(label_int, dtype=np.int32),latest_pred_path, np.float32(latest_loss), np.array(latest_pred, dtype=np.int32)
        elif mode == 'test':
            return img, index

    if mode == 'train':
        output_dtypes = [tf.float32,   tf.int32, tf.string, tf.float32, tf.int32]
        padded_shapes = ([48, None, 1], [None], [], [], [None])
        padded_values = (-1., -1, '', -1., -1)
    elif mode == 'test':
        output_dtypes = [tf.float32,  tf.string]
        padded_shapes = ([48, None, 1], [])
        padded_values = (-1., '')
    else:
        raise Exception('mode must be train|test')
        
    index = list(df.index)
    df = df.sort_values(by=['width'])
    df.index = df.index.map(lambda x: str(x))
    
    index = [str(_) for _ in index]
    
    data_placeholder = tf.placeholder(tf.string, [None], name=dataset_name)
    
    dataset = tf.data.Dataset.from_tensor_slices(data_placeholder)
    
    dataset = dataset.map(lambda item1: tf.py_func(
              map_fn_ctc, [item1], output_dtypes), num_parallel_calls=8)
    
    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes, 
                                        padding_values=padded_values)
    
    dataset = dataset.prefetch(100)
    
    prefetch_op = tf.data.experimental.prefetch_to_device(device="/gpu:0")

    dataset = dataset.apply(prefetch_op)

    iterator = dataset.make_initializable_iterator()
    input_tensors = iterator.get_next()
    return input_tensors, DataIniter(iterator.initializer, index=index ,data_placeholder= data_placeholder) 

def run_data_init(initers, shuffle, sess, batch_size):
    """
        
    """
    for initer in initers:
        data = initer.index
        if shuffle:
            data = shuffle_by_batch(data, batch_size)
        sess.run(initer.initializer, {initer.data_placeholder: data})




def convert_batch_int_to_text(int_lines, label_text, repeat_colapse=True):
    """
        Inputs  : batch of decoded text (int)
        Returns : text lines(chars)
    """
    texts = []
    last = len(label_text)
    for line in int_lines:
        text = ''
        line = np.reshape(line, [-1]).astype(int)
        for ichar in line:
            if ichar != -1 and ichar != len(label_text):
                if repeat_colapse==True:
                    if ichar != last:
                        text += label_text[ichar] 
                else:
                    text += label_text[ichar]
            last = ichar
        texts.append(text)
    return texts

# @memoize
def load_dataset_ctc(paths_to_label, factor=4, text_label=None):
    # @memoize
    def load(path):
        df = pandas.read_json(path)
        df['text'] = df['text'].map(lambda x: str(x))
        if 'mapable' in df.keys():
            df = df[df['mapable']==True]
        if not 'path' in df.keys():
            raw_dir = os.path.join(os.path.dirname(path), 'raw')
            df['path'] = df['name'].map(lambda x: os.path.join(raw_dir, x))
            
        # if not 'width' in df.keys():
        # calculate width
        print('not width')
        def f_width(index):
            try:
                shape = cv2.imread(df.loc[index, 'path']).shape
            except:
                shape = [-1, -1]
            return {index: {'height': shape[0], 'width': shape[1]}}


        print('read width:, ', len(df))


        results = multi_thread(f_width, df.index, 8).values()
        df['width'] = [_['width'] for _ in results]
        df['height'] = [_['height'] for _ in results]

        df.to_json(path)

        if not 'text_int' in df.keys() and text_label is not None:
            def convert_to_textint(text):
                try:
                    text_int =  [text_label[ch] for ch in text]
                    return text_int
                except Exception as e:
                    return None

            df['text_int'] = df['text'].map(convert_to_textint)
            
            df = df[df['text_int'].values != None]
            
        df.index = df['path']
        print("before return", len(df))
        return df
    
    
    
    rt = None
    for path in paths_to_label:
        print('loading pandas:', path)
        df = load(path)


        if rt is None:
            rt = df
        else:
            rt = pandas.concat([rt, df], sort=False)


    # remove invalid text
    valid_idx = [(_ is not None and len(_.replace(' ', '')) > 0) for _ in rt['text'].values]
    rt = rt[valid_idx]
    rt['len'] = rt['text'].map(lambda x: len(x)).values
    pred_len = (rt['width']/(rt['height']/48)//factor).values
    rt['pred_len'] = pred_len
    
    rt['is_long_enough'] = np.logical_and(rt['pred_len'].values > rt['len'].values, rt['width'] > 4)
    rt = rt[rt['pred_len'] < 2000/factor]
    rt = rt[rt['is_long_enough']==True]
    
    rt = rt[['path', 'text_int', 'width']]

    rt = rt.sort_values(by=['width'])
    return rt



if __name__ == '__main__':
    build_qa_dataset('train', os.path.join('./datasets/dc3/train_pad_all'), padding_style='all')
    build_qa_dataset('train', os.path.join('./datasets/dc3/train_pad_small'), padding_style='only_small')
    build_qa_dataset('test', os.path.join('./datasets/dc3/test_pad_all'), padding_style='all')
    build_qa_dataset('test', os.path.join('./datasets/dc3/test_pad_small'), padding_style='only_small')
    
    
    build_qa_dataset('train', os.path.join('./datasets/dc3/train_pad_all'), padding_style='all')
    build_qa_dataset('train', os.path.join('./datasets/dc3/train_pad_small'), padding_style='only_small')
    build_qa_dataset('test', os.path.join('./datasets/dc3/test_pad_all'), padding_style='all')
    build_qa_dataset('test', os.path.join('./datasets/dc3/test_pad_small'), padding_style='only_small')
