# Auto generate image files
# =============================================================================
import abc
import glob
import os
import json
import math
import multiprocessing
import pdb
import pickle
import re
import time
import unicodedata

import cv2
import numpy as np
import numpy.random as random
from fontTools.ttLib import TTFont
from fontTools.ttLib.sfnt import readTTCHeader
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import rotate

import dataloader.utils.constants as constants
from dataloader.utils.misc import (get_truncated_normal,
    check_allowed_char_version, dump_json)
from dataloader.utils.normalize import load_conversion_table, normalize_text
from dataloader.utils.preprocessing import (
    crop_image, adjust_stroke_width,
    normalize_grayscale_color, unsharp_masking, skeletonize_image,
    trim_image_horizontally)
from dataloader.utils.augment import (
    HandwritingAugment, PrintAugment, HandwritingMildAugment,
    PrintMildAugment, HandwritingCharacterAugment)



WHITE_RE = re.compile('\s')


class TrainingHelper(object):

    def initialize_helper(self, **kwargs):
        pass

    def postprocess_label(self, label, **kwargs):
        return label

    def postprocess_image(self, image, **kwargs):
        return image

    def postprocess_outputs(self, images, widths, labels, **kwargs):
        return images, widths, labels


class HandwritingHelper(TrainingHelper):
    """
    Help with making out data clean for generation
    """
    def __init__(self):
        """Initialize the helper"""

        self.label_2_char = {}
        self.char_2_label = {}

    def initialize_helper(self, allowed_chars, **kwargs):
        """Update the label_2_char and char_2_label dictionaries"""

        letter_list = list(allowed_chars)
        letter_list.sort()
        for _idx, each_character in enumerate(letter_list):
            self.label_2_char[_idx+1] = each_character
            self.char_2_label[each_character] = _idx+1

        self.label_2_char[0] = '_pad_'
        self.char_2_label['_pad_'] = 0

    def is_char_exists(self, char):
        """Check if the character exists in database

        # Arugments
            char [str]: the character to check

        # Returns
            [bool]: True if the character exists, False otherwise
        """
        return char in self.char_2_label

    def postprocess_label(self, label, **kwargs):
        """Perform post-processing on the label

        # Arguments
            text [str]: the text to post-process

        # Returns
            [str]: the post-processed text
        """
        if kwargs.get('label_converted_to_list', False):
            label_ = []
            for each_char in label:
                label_.append(self.char_2_label[each_char])
            label = label_

        return label

    def postprocess_image(self, image, **kwargs):
        """Perform post-processing on the image

        # Arguments
            image [np array]: the image to postprocess

        # Returns
            image [np array]: the result image
        """
        if kwargs.get('get_6_channels', False):
            channel_first = kwargs.get('channel_first', False)
            kwargs['append_channel'] = False
            image = self._get_6_features(1 - image, channel_first)

        if kwargs.get('append_channel', False):
            axis = 0 if kwargs.get('channel_first', False) else -1
            image = np.expand_dims(image, axis=axis)

        return image

    def postprocess_outputs(self, images, widths, labels, **kwargs):
        return np.asarray(images, dtype=np.uint8), widths, labels

    def get_number_of_classes(self):
        """Get the number of classes

        # Returns
            [int]: number of classes
        """
        return len(self.char_2_label)

    def _get_4_features(self, image, channel_first):
        """Create a 4-channel feature image

        The features include: binary image, canny edge image, gradient
        in y and gradient in x.

        # Arguments
            image [np array]: the image, should be binary
            channel_first [bool]: whether the image is CxHxW or HxWxC

        # Returns
            [np array]: the 4-channel image
        """
        edge = (cv2.Canny(image * 255, 50, 150) / 255).astype(np.uint8)
        dx = cv2.Scharr(image, ddepth=-1, dx=1, dy=0) / 16
        dy = cv2.Scharr(image, ddepth=-1, dx=0, dy=1) / 16

        axis = 0 if channel_first else -1
        image_4_features = np.stack([image, edge, dx, dy], axis=axis)

        return image_4_features

    def _get_6_features(self, image, channel_first):
        """Create a 6-channel feature image

        The features include: binary image, canny edge image, gradient
        in y, gradient in x, crop 4 and crop 8.

        # Arguments
            image [np array]: the image, should be binary
            channel_first [bool]: whether the image is CxHxW or HxWxC

        # Returns
            [np array]: the 6 channel image
        """
        height, width = image.shape

        image_4_features = self._get_4_features(image, channel_first)
        img1 = cv2.resize(image[4:-4,:], (width, height),
                          interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(image[8:-8,:], (width, height),
                          interpolation=cv2.INTER_LINEAR)

        axis = 0 if channel_first else -1
        img1 = np.expand_dims(img1, axis=-1)
        img2 = np.expand_dims(img2, axis=-1)

        return np.concatenate([image_4_features, img1, img2], axis=axis)


class ToniHelper(TrainingHelper):
    """
    Help with making out data clean for generation for Toni
    """
    def __init__(self):
        """Initialize the helper"""

        self.label_2_char = {}
        self.char_2_label = {}

    def initialize_helper(self, allowed_chars, **kwargs):
        """Update the label_2_char and char_2_label dictionaries"""

        letter_list = list(allowed_chars)
        letter_list.sort()
        for _idx, each_character in enumerate(letter_list):
            self.label_2_char[_idx+1] = each_character
            self.char_2_label[each_character] = _idx+1

        self.label_2_char[0] = '_pad_'
        self.char_2_label['_pad_'] = 0

    def is_char_exists(self, char):
        """Check if the character exists in database

        # Arugments
            char [str]: the character to check

        # Returns
            [bool]: True if the character exists, False otherwise
        """
        return char in self.char_2_label

    def postprocess_label(self, label, **kwargs):
        """Perform post-processing on the label

        # Arguments
            text [str]: the text to post-process

        # Returns
            [str]: the post-processed text
        """
        label = WHITE_RE.sub('', label)

        if kwargs.get('label_converted_to_list', False):
            label_ = []
            for each_char in label:
                label_.append(self.char_2_label[each_char])
            label = label_

        return label

    def postprocess_image(self, image, **kwargs):
        """Perform post-processing on the image

        # Arguments
            image [np array]: the image to postprocess

        # Returns
            image [np array]: the result image
        """
        if kwargs.get('append_channel', False):
            axis = 0 if kwargs.get('channel_first', False) else -1
            image = np.expand_dims(image, axis=axis)

        return image

    def postprocess_outputs(self, images, widths, labels, **kwargs):
        return np.asarray(images, dtype=np.uint8), widths, labels

    def get_number_of_classes(self):
        """Get the number of classes

        # Returns
            [int]: number of classes
        """
        return len(self.char_2_label)


class BaseOCRGenerator(metaclass=abc.ABCMeta):

    def __init__(self, height=64, allowed_chars=None, helper=None,
        is_binary=False, verbose=2):
        """Initialize the object

        # Arguments
            height [int]: the height of text line
            helper [TrainingHelper object]: use as hook when postprocess result
            allowed_chars [str]: the path to a text file of allowed character.
                Each character a line, and each character should be in
                ordinal form
            is_binary [bool]: whether this is a binary or grayscale image
                generator
            verbose [int]: the verbosity level
        """
        if not isinstance(height, int):
            raise ValueError('`height` should be an integer')

        if not isinstance(verbose, int):
            raise ValueError('`verbose` should be an integer')

        self.height = height        # the height of generated image
        self.verbose = verbose
        self.is_binary = is_binary
        self.folder_path = None
        self.folder_list_files = None

        # text configuration
        self.conversion_table = load_conversion_table()
        self.corpus_lines = []      # list of text strings
        self.corpus_size = 0

        # utility
        self.helper = TrainingHelper() if helper is None else helper
        self.iterations = 0
        self.allowed_chars = None
        if allowed_chars is not None:
            if check_allowed_char_version(allowed_chars):
                with open(allowed_chars, 'r') as f_in:
                    self.allowed_chars = [
                        each_line for each_line in f_in.read().splitlines()]
                    self.allowed_chars = set(self.allowed_chars)
            else:
                print(':WARNING: number allowed_char text file is deprecated')
                self.allowed_chars = np.loadtxt(allowed_chars, dtype=int)
                self.allowed_chars = set(
                    [chr(each_ord) for each_ord in self.allowed_chars])


    ###################
    # Utility methods #
    ###################
    def _is_number(self, char):
        """Whether the character is a number

        # Arguments
            char [str]: the character to check

        # Returns
            [bool]: True if the character is a number, False otherwise
        """
        if 48 <= ord(char) <= 57:
            return True

        return False


    ######################
    # Generation methods #
    ######################
    @abc.abstractmethod
    def _remove_unknown_characters(self, text):
        pass

    @abc.abstractmethod
    def _get_config(self):
        pass

    @abc.abstractmethod
    def _generate_single_image(self):
        pass

    @abc.abstractmethod
    def _generate_sequence_image(self, text, debug=True):
        pass


    #####################
    # Interface methods #
    #####################
    def get_image(self, image_path, label_file=None, is_binarized=False):
        """Get the image and label from image_path

        The input image should have dark text on white background. The filename
        for for the image should also contains label, and have the form of
        <idx_number>_<label>.[png/jpg]

        # Arguments
            image_path [str]: the path to stored image
            label_file [str or dictionary]: the file containing label. If it is
                a function, then it will take `image_path` and returns the
                label, if it is a dictionary, then the key should be
                image_path's filename, if it is None, then label is the
                filename
            is_binarized [bool]: whether the loaded image is already in
                binary form

        # Returns
            [np array]: the image in binary form (0 - 1) or grayscale form
                (0 - 255)
            [str]: the corresponding label
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.is_binary:
            if not is_binarized:
                image = cv2.GaussianBlur(image, (3, 3), 0)
                image = cv2.threshold(image, 0, 255,
                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            image = (image / 255).astype(np.uint8)

        if label_file is None:
            filename = os.path.basename(image_path)
            filename, _ = os.path.splitext(filename)
            label = '_'.join(filename.split('_')[1:])
        elif isinstance(label_file, dict):
            filename = os.path.basename(image_path)
            # filename, _ = os.path.splitext(filename)
            label = label_file.get(filename, None)
        elif callable(label_file):
            label = label_file(image_path)
        if label:
            return image, normalize_text(label, self.conversion_table)
        else:
            return image, None

    def get_batch(self, batch_size, max_width=-1, debug=False, **kwargs):
        """Get the next data batch.

        # Arguments
            batch_size [int]: the size of the batch
            max_width [int]: the maximum width that an image can have. If this
                value is smaller than 1, then `max_width` is automatically
                calculated
            debug [bool]: whether to retrieve debug value
            **append_channel [bool]: if True, then pad the image
            **label_converted_to_list [bool]: whether label should be in list
                form (converted from char_2_label). Otherwise label is just a
                text string

        # Returns
            [list 4D np.ndarray]: list of binary image (0: background)
            [list of strs]: the label of each image
            [list of ints]: the width of each image
        """
        if max_width <= 0:
            max_width = float('inf')

        # get the images and determine its eligibility
        if self.iterations >= len(self.corpus_lines):
            # shuffle the corpus every iterations
            random.shuffle(self.corpus_lines)
            self.corpus_lines.sort(key=lambda each_line: len(each_line))
            self.iterations = 0
        self.iterations += batch_size

        # get the images and determine their eligibility
        _temp_store = []
        debug_info = []
        index = random.choice(self.corpus_size)
        while len(_temp_store) < batch_size:
            index = (index + 1) % self.corpus_size
            each_string = self.corpus_lines[index]
            image, chars, each_debug_info = self._generate_sequence_image(
                each_string, debug=True)

            if image is None or len(chars) == 0 or len(np.unique(image)) < 2:
                continue

            original_height, original_width = image.shape
            new_width = original_width * self.height // original_height
            if new_width > max_width:
                continue

            _temp_store.append((image, chars, new_width))
            debug_info.append(each_debug_info)

        batch_images = []
        batch_labels = []
        batch_widths = []
        if max_width == float('inf'):
            if len(_temp_store) == 0:
                raise ValueError("_temp_store is empty, no image is appended")
            max_width = max(_temp_store, key=lambda obj: obj[2])[2] + 1

        # Process the image
        for each_image, each_label, new_width in _temp_store:
            image = self._batching_image(each_image, new_width, max_width)
            image = self.helper.postprocess_image(image, **kwargs)
            label = self.helper.postprocess_label(each_label, **kwargs)

            batch_images.append(image)
            batch_labels.append(label)
            batch_widths.append(new_width)

        if debug:
            return self.helper.postprocess_outputs(
                batch_images, batch_widths, batch_labels
            ), debug_info

        return self.helper.postprocess_outputs(
            batch_images, batch_widths, batch_labels
        )

    def get_batch_from_files(self, batch_size=1, max_width=-1, path=None,
        label_file=None, mode=constants.TRAIN_MODE, skip_invalid_image=True,
        is_binarized=False, **kwargs):
        """Get the next data batch.

        # Arguments
            batch_size [int]: the size of the batch, if batch size is smaller
                than 1, then the batch size is the whole image files
            path [str]: the folder containing image, otherwise generate
                image. If
            max_width [int]: the maximum width that an image can have. If this
                value is smaller than 1, then `max_width` is automatically
                calculated
            label_file [str]: the path to the json file containing mapping
                between image filename with the true label. If `path`
                is not None and `label_file` is None, then the label is the
                filename
            mode [int]: whether TRAIN_MODE (1), TEST_MODE (2) or INFER_MODE
                (3). If that is the case, then in TRAIN_MODE, invalid
                label characters will either be removed, or the whole image
                is removed (depending on `skip_invalid_image`); in TEST_MODE,
                label_converted_to_list will be ignored; in INFER_MODE, label
                will be ignored
            skip_invalid_image [bool]: whether to skip invalid image or skip
                the whole image (only applicable in TRAIN_MODE)
            is_binarized [bool]: whether the image is already in binary form
            **append_channel [bool]: if True, then pad the image
            **label_converted_to_list [bool]: whether label should be in list
                form (converted from char_2_label). Otherwise label is just a
                text string

        # Returns
            [list 4D np.ndarray]: list of binary image (0: background)
            [list of strs]: the label of each image
            [list of ints]: the width of each image
        """
        if max_width <= 0:
            max_width = float('inf')

        if batch_size < 1:
            batch_size = float('inf')

        # Get the images and determine its eligibility
        _temp_store = []
        if os.path.isfile(path):
            working_images = [path]
            batch_size = 1
        else:
            if path != self.folder_path:
                self.initialize_folder(path)
            if mode == constants.TRAIN_MODE:
                random.shuffle(self.folder_list_files)
            working_images = self.folder_list_files

        # Access the label
        if label_file is not None:
            with open(label_file, 'r',encoding="utf8") as f:
                label_obj = json.load(f)
        else:
            label_obj = None

        # Load each image
        for _idx, each_file in enumerate(working_images):
            if len(_temp_store) >= batch_size:
                break

            image, chars = self.get_image(each_file, label_obj, is_binarized)
            if image is None:
                print(':WARNING: skip {} (image is None)'.format(each_file))
                continue

            if (chars is None) and (mode != constants.INFER_MODE):
                print(':WARNING: skip {} (no label)'.format(each_file))
                continue

            # this means true label contains more other characters than
            # the allowed characters
            if self.allowed_chars is not None and mode == constants.TRAIN_MODE:
                invalid_chars = set(chars).difference(self.allowed_chars)
                if skip_invalid_image:
                    if len(invalid_chars) > 0:
                        print(":WARNING: skip {} (contains invalid char: {})"
                            .format(each_file, invalid_chars))
                        continue
                else:
                    text = [each_char for each_char in chars
                        if each_char in self.allowed_chars]
                    _temp_chars = ''.join(text)
                    if _temp_chars == '':
                        print(':WARNING: skip {} (all valid chars removed'
                            .format(each_file))
                        continue
                    if len(invalid_chars) > 0:
                        print(':WARNING: miss {}. Original text {} - New text'
                              ': {}'.format(invalid_chars, chars, _temp_chars))
                    chars = _temp_chars

            original_height, original_width = image.shape
            new_width = original_width * self.height // original_height
            if new_width > max_width:
                continue

            _temp_store.append((image, chars, new_width))

        batch_images = []
        batch_labels = []
        batch_widths = []
        if max_width == float('inf'):
            if len(_temp_store) == 0:
                raise ValueError("_temp_store is empty, no image is appended")
            max_width = max(_temp_store, key=lambda obj: obj[2])[2] + 1

        # Process the image
        for each_image, each_label, new_width in _temp_store:
            image = self._batching_image(each_image, new_width, max_width)
            image = self.helper.postprocess_image(image, **kwargs)
            label = self.helper.postprocess_label(each_label, **kwargs)

            batch_images.append(image)
            batch_labels.append(label)
            batch_widths.append(new_width)

        return self.helper.postprocess_outputs(
            batch_images, batch_widths, batch_labels
        )

    def generate_images(self, start=0, end=None, save_dir=None,
        label_json=False, debug=True, text_lines=None):
        """Generate images into png files

        # Arguments
            start [int]: the first text used to generate image
            end [int]: the last text used to generate image. If None, then it
                will generate to the last text corpus_lines
            save_dir [str]: path to the folder
            label_json [bool]: whether to generate a json label file
            debug [bool]: whether to retrieve debug information
            text_lines [str or list of str]: the text line to specifically
                generate. If None, then generate from self.corpus_lines
        """
        max_width = 0
        missing_chars = set([])
        if end is None:
            end = len(self.corpus_lines)

        labels = {}
        now = int(time.time())
        if save_dir == None:
            save_dir = str(now)
        os.makedirs(save_dir, exist_ok=True)

        if isinstance(text_lines, str):
            text_lines = [text_lines]

        debug_info = {}
        corpus = (
            self.corpus_lines[start:end]
            if text_lines is None
            else text_lines)
        cl = len(str(len(corpus)))
        for _idx, each_text in enumerate(corpus):
            image, chars, each_debug_info = self._generate_sequence_image(
                text=each_text, debug=True)

            if image is None:
                print(':WARNING: image is None for text {}'.format(each_text))
                continue

            if len(np.unique(image)) < 2:
                print(':WARNING: image has just a single color for text {}'
                    .format(each_text))
                continue

            if chars == '':
                print(':WARNING: there isn\'t any character for text {}'
                    .format(each_text))
                continue

            if max_width < image.shape[1]:
                max_width = image.shape[1]

            if len(chars) < len(each_text):
                print(' Label: {} --> Move to: {} --> Debug: {}'
                    .format(each_text, chars, each_debug_info))

            if self.is_binary:
                image = (image * 255).astype(np.uint8)

            if label_json:
                filename = '{:0>{al}}_{}.png'.format(_idx, now, al=cl)
            else:
                filename = '{:0>{al}}_{}.png'.format(_idx, chars, al=cl)
            
            cv2.imwrite(os.path.join(save_dir, filename), image)
            labels[filename] = chars
            debug_info[filename] = each_debug_info
            missing_chars = missing_chars.union(
                set(each_debug_info['missing_chars']))

        if label_json:
            dump_json(
                labels,
                os.path.join(save_dir, 'labels.json'),
                sort_keys=True)
        
        if debug:
            dump_json(debug_info, os.path.join(save_dir, 'debug.json'))

        if self.verbose >= 2:
            print('Max-Width', max_width)
            print('Missing:', missing_chars)

    def get_helper(self):
        """Return the helper"""
        return self.helper

    def initialize(self):
        """Intialize the generator, augmentator, helper"""

        self.augment.build_augmentators()
        self.helper.initialize_helper(allowed_chars=self.allowed_chars)

        # remove unkwown characters and sort by length (would make more
        # efficient training if sequence lengths in a batch are similar)
        random.shuffle(self.corpus_lines)
        temp_corpus_ = []
        missing_ = set([])
        for each_line in self.corpus_lines:
            clean_line, missing = self._remove_unknown_characters(each_line)
            missing_.update(missing)
            if len(clean_line) > 0:
                temp_corpus_.append(clean_line)
        temp_corpus_.sort(key=lambda each_line: len(each_line))
        self.corpus_lines = temp_corpus_
        self.corpus_size = len(self.corpus_lines)

        if self.verbose > 2:
            print('Generator contains {} classes and {} lines'
                .format(len(self.allowed_chars), self.corpus_size))

    def initialize_folder(self, folder_path):
        """Initialize folder to generate images from folders

        # Arguments
            folder_path [str]: the path to folder containing images
        """
        self.folder_path = folder_path
        self.folder_list_files = glob.glob(os.path.join(folder_path, '*.png'))
        self.folder_list_files += glob.glob(os.path.join(folder_path, '*.jpg'))


class HandwrittenLineGenerator(BaseOCRGenerator):
    """
    This generator assumes there are:
        (1) a collection of text files, from which the content of images will
            be generated;
        (2) a collection of pickle files, each of which contains
            [list of np array images] and [list of labels]. Each image should
            have white background, black foreground, and not having outer
            background

    Usage example:
    ```
    import os
    from dataloader.generate.image import HandwrittenLineGenerator

    lineOCR = HandwrittenLineGenerator()
    lineOCR.load_character_database('images.pkl')
    lineOCR.load_text_database('text.txt')

    # to get random characters
    X, widths, y = lineOCR.get_batch(4, 1800)

    # or to generate 100 images
    lineOCR.generate_images(start=0, end=100, save_dir='/tmp/Samples')
    ```
    """

    def __init__(self, height=64, helper=None, limit_per_char=1000, verbose=2,
                 allowed_chars=None, is_binary=False, augmentor=None,
                 deterministic=True):
        """Initialize the generator.

        # Arguments
            limit_per_char [int]: the maximum number of images that each
                character will contain (to save on RAM)
            augmentor [Augment object]: the augmentator to use
        """
        super(HandwrittenLineGenerator, self).__init__(
            height=height, helper=helper, allowed_chars=allowed_chars,
            is_binary=is_binary, verbose=verbose)

        # image configuration
        self.limit_per_char = limit_per_char
        self.character_height = height - 10     # to help resize to self.height
        self.char_2_imgs = {}
        self.char_2_imgs_train = {}    # separate set of characters for training
        self.char_2_imgs_val = {}    # separate set of characters for validation
        self.char_2_imgs_test = {}      # separate set of characters for testing
        self.background_value = 1 if is_binary else 255
        self.interpolation = (
            cv2.INTER_NEAREST if self.is_binary else
            cv2.INTER_LINEAR)

        # text configuration
        self.chemical_formulas = []

        # utility
        self.augment = (
            augmentor(is_binary=is_binary) if augmentor is not None
            else HandwritingMildAugment(is_binary=is_binary))
        self.helper = HandwritingHelper() if helper is None else helper
        self.deterministic= deterministic

    def _add_chemical_formula(self, text):
        """Add chemical formula to the current text

        # Arguments
            text [str]: the text to add chemical formula

        # Returns
            [str]: the text with chemical formula added
            [set of ints]: the index of characters associated with the
                chemical formula
        """
        formula = random.choice(self.chemical_formulas)
        insert_idx = random.choice(len(text))

        if insert_idx == 0:
            insert_formula = '{} '.format(formula)
            start_idx = 0
        elif insert_idx == len(text) - 1:
            insert_formula = ' {}'.format(formula)
            start_idx = insert_idx + 1
        else:
            insert_formula = ' {} '.format(formula)
            start_idx = insert_idx + 1

        result = text[:insert_idx] + insert_formula + text[insert_idx:]

        return result, set(range(start_idx, start_idx+len(formula)))

    def _remove_unknown_characters(self, text):
        """Check for characters in the text that are not in database.

        # Arguments
            text [str]: the string to check

        # Returns
            [str]: the text that have missing characters removed
            [set of str]: missing characters
        """
        exist = []
        missing_chars = set([])
        for each_char in text:

            if each_char not in self.char_2_imgs:
                missing_chars.add(each_char)
                continue

            if not (each_char in self.allowed_chars or
                    self.helper.is_char_exists(each_char)):
                missing_chars.add(each_char)
            else:
                exist.append(each_char)

        return ''.join(exist), missing_chars

    def _get_config(self, text, default_config={}):
        """Returns a configuration object for the text

        The idea is that each line of text will have a specific configuration,
        which then will be used during image generation. The configuration file
        has the following format: {
            config_key: config_value
            'text': [list of {}s with length == len(text), with each {} is a
                     config for that specific word]
        }

        The returning object contains these following keys:
            'text': a list of each character configuration

        Each character configuration object contains these following keys:
            'skewness': the skew angle of character
            'character_normalization_mode': an integer specifcy how to
                normalize resized characters
            'space': the space distance to the next character
            'bottom': the bottom padding
            'height': height of character (in pixel)
            'width_noise': the amount to multiply with character width

        # Arguments
            text [str]: the text to generate string
            default_config [dict]: the default config value

        # Returns
            [obj]: the configuration object
        """
        # config = {'text': []}
        # text_config = {'height': 0, 'width': 0, 'skewness': 0, 'space': 0}

        ### Configuration for each char
        is_space_close = random.random() > 0.8
        is_skewness = random.random() > 0.8
        is_curve_line = ((len(text) > 20) and (random.random() > 0.9)
            and default_config.get('is_curve_line', True))
        is_last_numbers_up = (len(text) > 10 and self._is_number(text[-1]) and
            self._is_number(text[-2]) and self._is_number(text[-3]) and
            random.random() > 0.9)
        if is_skewness:
            skew_value = random.randint(-10, 10)

        if is_curve_line:
            curve_start = random.randint(5, len(text) - 10)
            curve_middle = random.randint(curve_start+3, len(text)-1)
            curve_end = random.randint(curve_middle, len(text))

            curve_max_angle = random.randint(10, 45)
            curve_first_half = set(range(curve_start, curve_middle))
            curve_second_half = set(range(curve_middle, curve_end+1))
            curve_delta_first_half = curve_max_angle / len(curve_first_half)
            curve_delta_second_half = curve_max_angle / len(curve_second_half)

            # convex vs concave curve
            curve_type = -1 if random.random() <= 0.5 else 1

        # how to normalize character width
        is_character_normalize = 1.0 if self.is_binary else random.random()
        if is_character_normalize <= 0.7:
            character_normalization_mode = 3
        elif is_character_normalize <= 0.8:
            character_normalization_mode = 2
        elif is_character_normalize <= 0.9:
            character_normalization_mode = 1
        else:
            character_normalization_mode = 0
        ### End char configuration

        # each text configuration
        text_config = []
        last_bottom = 0
        chem_chars = default_config.get('chem_chars', {})
        for _idx, each_char in enumerate(text):

            # space
            if _idx == len(text) - 1:
                space = 2
            elif (_idx in chem_chars) and (_idx != max(chem_chars)):
                space = random.randint(0, 2)
            elif (self._is_number(each_char)
              and not self._is_number(text[_idx+1])):
                space = random.randint(15, 25)
            elif (not self._is_number(each_char)
              and self._is_number(text[_idx+1])):
                space = random.randint(15, 25)
            elif ord(each_char) < 12800:
                space = random.randint(5, 15)
            else:
                if is_space_close:
                    # @TODO: currently no overlapping
                    # space = random.randint(-4, 2)
                    space = random.randint(0, 2)
                else:
                    # space = random.randint(-2, 5)
                    space = random.randint(2, 10)

            # skew value
            if not self._is_number(each_char) and is_skewness:
                skew = 2 * random.random() - 1 + skew_value
            else:
                skew = 0

            # character height and width
            if _idx in chem_chars:
                if each_char in constants.NUMBERS:
                    height_ratio = random.uniform(low=0.45, high=0.55)
                elif each_char in constants.SMALL_LATIN:
                    height_ratio = random.uniform(low=0.55, high=0.65)
                else:
                    height_ratio = random.uniform(low=0.8, high=0.9)
            elif each_char in constants.SMALL_CHARS:
                height_ratio = random.uniform(low=0.15, high=0.3)
            elif each_char in constants.MEDIUM_CHARS:
                height_ratio = random.uniform(low=0.3, high=0.5)
            elif each_char in constants.SMALL_KATA_HIRA:
                height_ratio = random.uniform(low=0.5, high=0.7)
            elif each_char in constants.SMALL_LATIN:
                height_ratio = random.uniform(low=0.6, high=0.75)
            elif each_char in constants.NORMAL_FORCE_SMALLER:
                height_ratio = random.uniform(low=0.7, high=0.75)
            elif each_char in constants.KATA_HIRA:
                height_ratio = random.uniform(low=0.75, high=0.95)
            elif each_char in constants.NUMBERS:
                height_ratio = random.uniform(low=0.8, high=1.0)
            elif each_char in constants.LARGER_THAN_NORMAL:
                height_ratio = random.uniform(low=1.0, high=1.3)
            else:
                height_ratio = get_truncated_normal(1, 0.02, 0.95, 1.05)
            height = int(self.character_height * height_ratio)
            width_noise = get_truncated_normal(1, 0.02, 0.95, 1.05)

            # character bottom padding value
            base_bottom = last_bottom
            if random.random() > 0.8:
                base_bottom = last_bottom + random.randint(-3, 3)
                last_bottom = base_bottom

            if each_char in constants.MIDDLE_CHARS:
                bottom = base_bottom + random.randint(10, 25)
            elif each_char in constants.TOP_CHARS:
                bottom = base_bottom + random.randint(30, 45)
            elif ((each_char in constants.BOTTOM_CHARS) or
                  (each_char in constants.NUMBERS and _idx in chem_chars)):
                bottom = base_bottom + random.randint(
                    -int(height * 2 / 3), -int(height / 3))
            else:
                bottom = base_bottom

            text_config.append({'skewness': skew, 'space': space,
                'character_normalization_mode': character_normalization_mode,
                'bottom': bottom, 'height': height,
                'width_noise': width_noise})

        # perform curve line configuration
        if is_curve_line:
            total_delta_bottom = 0
            total_delta_angle = 0
            for _idx, each_config in enumerate(text_config):
                if _idx in curve_first_half:
                    total_delta_bottom += curve_type * random.randint(2,5)
                    each_config['bottom'] += total_delta_bottom
                    total_delta_angle += curve_type * curve_delta_first_half
                    each_config['skewness'] += total_delta_angle
                elif _idx in curve_second_half:
                    total_delta_bottom += curve_type * random.randint(2,5)
                    each_config['bottom'] += total_delta_bottom
                    total_delta_angle -= curve_type * curve_delta_second_half
                    each_config['skewness'] += total_delta_angle
                elif _idx == curve_middle:
                    total_delta_bottom += curve_type * random.randint(2,5)
                    each_config['bottom'] += total_delta_bottom
                    each_config['skewness'] += curve_type * curve_max_angle
                elif _idx >= curve_end:
                    each_config['bottom'] += total_delta_bottom

        # normalize the bottom value (such that the lowest value should be 3)
        min_bottom = min(text_config, key=lambda obj: obj['bottom'])['bottom']
        for each_config in text_config:
            each_config['bottom'] = each_config['bottom'] - min_bottom + 3

        return {
            'text': text_config,
        }

    def _generate_single_image(self, char, config, char_idx=None):
        """Generate a character image

        # Arguments
            char [str]: a character to generate image
            config [obj]: a configuration object for this image
            char_idx [int]: to determine the character beforehand

        # Returns
            [np array]: the generated image for that specific string
        """
        if char not in self.char_2_imgs.keys():
            return None

        choice = (random.choice(len(self.char_2_imgs[char]))
                  if char_idx is None
                  else char_idx)
        image = self.char_2_imgs[char][choice]

        # rotate image
        image = rotate(image, config['skewness'], order=1,
                       cval=self.background_value)

        # resize image
        if char not in constants.NOT_RESIZE:
            height, width = image.shape
            desired_width = int(
                width * (config['height'] / height) * config['width_noise'])
            image = self._resize_character(
                image, config['height'],
                desired_width, config['character_normalization_mode'])

        # add horizontal space and bottom space
        image = np.pad(image, ((0, config['bottom']), (0, config['space'])),
                       'constant', constant_values=self.background_value)

        return image

    def _generate_sequence_image(self, text, debug=True):
        """Generate string image of a given text

        # Arguments
            text [str]: the text that will be used to generate image

        # Returns
            [np array]: the image generated
            [str]: the text label
            [list of str]: list of missing characters
        """
        char_images = []
        default_config = {}
        missing_chars = {}

        if self.chemical_formulas and random.random() < 0.3:
            text, chem_chars = self._add_chemical_formula(text)
            default_config['chem_chars'] = chem_chars
            default_config['is_curve_line'] = False

        config = self._get_config(text, default_config)

        # Calculate the average height of a character
        if self.deterministic:
            indices = {each_char:random.choice(len(self.char_2_imgs[each_char]))
                       for each_char in list(set(text))}
        else:
            indices = {}

        for _idx, each_char in enumerate(text):
            char_images.append(self._generate_single_image(
                each_char, config['text'][_idx], indices.get(each_char, None))
            )

        # Normalize character image height to have the same height by padding
        # the top into desired_height
        max_height = max(char_images, key=lambda obj: obj.shape[0]).shape[0]
        desired_height = max_height + 6
        norm_img_seq = []
        for each_img in char_images:
            top_pad = desired_height - each_img.shape[0] - 3
            norm_img_seq.append(np.pad(each_img, ((top_pad,3), (0,0)),
                mode='constant', constant_values=self.background_value))

        image = np.concatenate(norm_img_seq, axis=1)
        image = self.augment.augment_line(image)
        
        if debug:
            return image, text, {
                'missing_chars': missing_chars
            }

        return image, text

    def _batching_image(self, image, new_width, max_width):
        """Prepare and normalize size image for batching

        # Arguments
            image [np array]: the image to process
            new_width [int]: the new width of the image (when image is resized
                to have `self.height` height)
            max_width [int]: the maximum width that an image can have

        # Returns
            [np array]: the image with fixed width (black stroke on white
                background)
        """
        resized_image = cv2.resize(image, (new_width, self.height),
                interpolation=self.interpolation)
        image = (np.ones([self.height, max_width], dtype=np.uint8)
            * self.background_value).astype(np.uint8)
        image[:, :new_width] = resized_image

        return image

    def _resize_character(self, image, desired_height, desired_width,
        character_normalization_mode=4):
        """Resize and normalize the character

        This method optionally normalizes the characters, so that the affect
        of resizing characters do not have a bias affects on the model.
        Sometimes we can skip normalization to provide more noise effects.

        # Arguments
            image [np array]: the character image to resize
            desired_height [int]: the desired height to resize the image into
            desired_width [int]: the desired width to resize the image into
            character_normalization_mode [int]: to have value of 0-3, variate
                the normalization scheme

        # Returns
            [np array]: the resized character image
        """
        original_height, original_width = image.shape
        ratio = desired_height / original_height

        # Resize the character
        image = cv2.resize(image, (desired_width, desired_height),
            interpolation=self.interpolation)

        # Adjust the stroke width, color, and deblur the result with some
        # randomness to enhance model robustness
        if character_normalization_mode > 0:
            image = adjust_stroke_width(image, ratio, is_binary=self.is_binary)

        if character_normalization_mode > 1:
            image = normalize_grayscale_color(image)

        if character_normalization_mode > 2:
            image = unsharp_masking(image)

        return image

    def load_character_database(self, file_path, shuffle=True):
        """Load image database into the dataset

        # Arguments
            file_path [str]: the path to pickle file to load X, y
            shuffle [bool]: whether to shuffle the wholething
        """
        with open(file_path, 'rb') as f:
            X, y = pickle.load(f)

            if shuffle:
                idx_permutation = random.permutation(len(y))
                X = [X[each_idx] for each_idx in idx_permutation]
                y = [y[each_idx] for each_idx in idx_permutation]

            # Sanity check a random image
            unique_pixels = len(np.unique(X[random.choice(idx_permutation)]))
            if self.is_binary:
                if unique_pixels != 2:
                    print(':WARNING: binary image should have 2 pixel values '
                          'but have {} values'.format(unique_pixels))
            else:
                if unique_pixels == 2:
                    print(':WARNING: the loaded dataset might be binary data')

            for _idx, each_X in enumerate(X):
                key = y[_idx]

                if key in self.char_2_imgs:
                    if len(self.char_2_imgs[key]) > self.limit_per_char:
                        continue

                if (self.allowed_chars is not None
                    and key not in self.allowed_chars):
                    continue

                if key in self.char_2_imgs:
                    self.char_2_imgs[key].append(each_X)
                else:
                    self.char_2_imgs[key] = [each_X]

        if self.verbose >= 2:
            print('{} loaded'.format(file_path))

    def load_text_database(self, file_path):
        """Load the text database from which to generate line strings

        # Arguments
            file_path [str]: the path to text file

        # Returns
            [int]: the number of text lines
        """
        with open(file_path, 'r', encoding="utf-8") as f_text:
            for each_row in f_text:
                if len(each_row) < 2:
                    continue

                line_string = normalize_text(
                    each_row.strip(), self.conversion_table)
                self.corpus_lines.append(line_string)

        return len(self.corpus_lines)

    def load_text_database_avd(self, file_path, is_random=False, top_freq=True,
        max_len=-1, ignore_strange=True, combine_text=1, max_char_per_line=20,
        max_char_rand_range=5, random_space_limit=5):
        def split_bylen(item, maxlen):
            '''
            Requires item to be sliceable (with __getitem__ defined)
            '''
            return [item[ind:ind + maxlen] for ind in range(0, len(item), maxlen)]
        nfname = file_path[:-4] + '_{}.txt'.format(max_len)
        all_stri = []
        if top_freq and not is_random and os.path.isfile(nfname):
            txt_file=nfname
            max_len=-1
        cline=0
        try:
            with open(file_path) as f:
                for l in f:
                    if max_len > 0 and cline > max_len * random_space_limit:
                        break
                    cline+=1
                    l = unicodedata.normalize('NFKC', l)
                    if ignore_strange and len(l)>0 and l[0] in self.char_2_label:
                        all_stri.append(normalize_text(
                    l.strip(), self.conversion_table))
        except:
            with open(file_path, encoding='utf-8') as f:
                for l in f:
                    if max_len > 0 and cline > max_len * random_space_limit:
                        break
                    cline+=1
                    l = unicodedata.normalize('NFKC', l)
                    if ignore_strange and len(l) > 0 and l[0] in self.char_2_label:
                        all_stri.append(normalize_text(
                    l.strip(), self.conversion_table))
        if max_len>0:
            if not top_freq:
                if not is_random:
                    all_stri=all_stri[:max_len]
                else:
                    all_stri=np.random.choice(all_stri, max_len)
                    #print(self.all_stri)
            else:
                cw={}
                for si, st in enumerate(all_stri):
                    if si%100==0:
                        print('\rcounted {}/{}'.format(si, len(all_stri)))
                    for c in st:
                        if c not in cw:
                            cw[c]=0
                        cw[c]+=1
                list_score=[]
                print('\n-----')
                for si, st in enumerate(all_stri):
                    if si%100==0:
                        print('\rscored {}/{}'.format(si, len(all_stri)))
                    cur_s=0
                    for c in st:
                        cur_s+=cw[c]
                    list_score.append(cur_s)
                print('\n-----')
                sort_strs=[x for _, x in sorted(zip(list_score, all_stri), reverse=True)]
                all_stri = sort_strs[:max_len]
                with open(nfname, 'w', encoding='utf-8') as f:
                    for si, st in enumerate(all_stri):
                        if si % 100 == 0:
                            print('\rwrited {}/{}'.format(si, len(all_stri)))
                        f.write(st)
                        f.write('\n')
        # raise False
        # np.random.shuffle(all_stri)
        print('num name {}'.format(len(all_stri)))
        if combine_text>1:
            c=0
            new_all_stri=[]
            while c<len(all_stri)-combine_text:
                news=''
                for j in range(combine_text):
                    news+=all_stri[c+j]
                new_all_stri.append(news)
                c+=combine_text
            all_stri=new_all_stri

        for stri in all_stri:
            if len(stri)<max_char_per_line:
                self.corpus_lines.append(stri)
            else:
                rdl=np.random.randint(max_char_per_line-max_char_rand_range,
                                           max_char_per_line+max_char_rand_range, 1)[0]
                stris=split_bylen(stri, rdl)
                for st in stris:
                    self.corpus_lines.append(st)
        print('num string {}'.format(len(self.corpus_lines)))
        return len(self.corpus_lines)

    def load_chemical_formulas(self, file_path):
        """Load the chemical formula text

        # Arguments
            file_path [str]: the path to text file containing formulas
        """
        with open(file_path, 'r') as f:
            for each_line in f:
                self.chemical_formulas.append(each_line.strip())

    def load_background_image_files(self, folder_path):
        """Load background image files

        # Arguments
            folder_path [str]: the path to folder that contains background
        """
        if self.is_binary:
            print(':WARNING: background image files are not loaded for binary '
                  'generation mode.')
        else:
            self.augment.add_background_image_noises(folder_path)

    def initialize(self, train_percent=0.8, val_percent=0.1):
        """Initialize the generator

        This method will be called when all text and image files are loaded. It
        will:
            1. check for missing characters
            2. construct validation and test characters
            3. randomize corpus_lines

        # Arguments
            train_percent [float]: percentage of training characters
            val_percent [float]: percentage of validation characters
        """
        if self.allowed_chars is None:
            self.allowed_chars = set(self.char_2_imgs.keys())
        super(HandwrittenLineGenerator, self).initialize()

        missing_chars = self.allowed_chars.difference(self.char_2_imgs.keys())
        if missing_chars:
            print(":WARNING: missing the current characters to generate: {}"
                .format(missing_chars))

        # create validation and test set
        if (train_percent + val_percent > 1 or
            train_percent + val_percent < 0):
            raise ValueError('invalid train_percent and/or val_percent')
        for char, image_list in self.char_2_imgs.items():
            num_image = len(image_list)
            if num_image < 10:
                self.char_2_imgs_train[char] = image_list
                self.char_2_imgs_test[char] = image_list
                self.char_2_imgs_val[char] = image_list
                continue
            train_idx = int(num_image * train_percent)
            val_idx = int(num_image * (train_percent+val_percent))
            self.char_2_imgs_train[char] = image_list[:train_idx]
            self.char_2_imgs_val[char] = image_list[train_idx:val_idx]
            self.char_2_imgs_test[char] = image_list[val_idx:]
        self.char_2_imgs = None

    def get_batch(self, batch_size, max_width=-1, mode=constants.TRAIN_MODE,
        **kwargs):
        """Get the batch image

        Subclass from super in order to determine which batch mode. The
        arguments can be seen in the subclass.

        # Arguments
            mode [int]: the mode of this batch
        """
        if mode == constants.TEST_MODE:
            self.char_2_imgs = self.char_2_imgs_test
        elif mode == constants.VALIDATION_MODE:
            self.char_2_imgs = self.char_2_imgs_val
        else:
            self.char_2_imgs = self.char_2_imgs_train

        return super(HandwrittenLineGenerator, self).get_batch(
            batch_size, max_width, **kwargs
        )

    def generate_images(self, start=0, end=None, save_dir=None,
        label_json=False, mode=constants.TRAIN_MODE, debug=True,
        text_lines=None, **kwargs):
        """Generate the images

        Subclass from super, in order to determine batch mode. Other
        arguments can be seen in the subclass

        # Arguments
            mode [int]: the mode of this generator
        """
        if mode == constants.TEST_MODE:
            self.char_2_imgs = self.char_2_imgs_test
        elif mode == constants.VALIDATION_MODE:
            self.char_2_imgs = self.char_2_imgs_val
        else:
            self.char_2_imgs = self.char_2_imgs_train

        return super(HandwrittenLineGenerator, self).generate_images(
            start, end, save_dir, label_json, debug=debug, text_lines=text_lines
        )

    def load_long_text_database(self, file_path, is_random=False,
      top_freq=False, max_len=-1, ignore_strange=True, combine_text=1,
      max_char_per_line=20, max_char_rand_range=2):
        """ Read the text file for synthetic generation

        # Arguments
            file_path [string]: path to txt file
            is_random [bool]: if true, choose random max_len lines in the file
            max_len [int]: max number of line chosen in txt file,
                default=-1 means getting all lines
            top_freq [bool]: get lines that contain top frequent words
            ignore_strange [bool]: if True, ignore the line that has first
                character not found in vocab
            combine_text [int]: combine many text in consecutive lines into
                one longer text
            max_char_per_line [int]: limit the length of text line
            max_char_rand_range [int]: range of random allowed around the
                max_char_per_line length
        """
        def split_bylen(item, maxlen):
            """Requires item to be sliceable (with __getitem__ defined)"""
            return [item[ind:ind + maxlen] for ind in
                range(0, len(item), maxlen)]

        nfname = file_path[:-4] + '_{}.txt'.format(max_len)
        all_stri = []
        if top_freq and not is_random and os.path.isfile(nfname):
            txt_file=nfname
            max_len=-1
        cline=0
        try:
            with open(txt_file) as f:
                for each_row in f:
                    if max_len>0 and cline>max_len*10:
                        break
                    l  = normalize_text(
                    each_row.strip(), self.conversion_table)
                    if ignore_strange and len(l)>0:
                        all_stri.append(str(l).strip())
                    cline+=1
        except:
            with open(file_path, 'rt', encoding='utf-8') as f:
                for each_row in f:
                    if max_len>0 and cline>max_len*10:
                        break
                    l = normalize_text(
                        each_row.strip(), self.conversion_table)
                    if ignore_strange and len(l) > 0:
                        all_stri.append(str(l).strip())
                    cline += 1
        print(len(all_stri))
        if max_len>0:
            if not top_freq:
                if not is_random:
                    all_stri=all_stri[:max_len]
                else:
                    all_stri=np.random.choice(all_stri, max_len)
                    #print(self.corpus_lines)
            else:
                cw={}
                for si, st in enumerate(all_stri):
                    # if si%100==0:
                    #     llprint('\rcounted {}/{}'.format(si, len(all_stri)))
                    for c in st:
                        if c not in cw:
                            cw[c]=0
                        cw[c]+=1
                list_score=[]
                print('\n-----')
                for si, st in enumerate(all_stri):
                    # if si%100==0:
                    #     llprint('\rscored {}/{}'.format(si, len(all_stri)))
                    cur_s=0
                    for c in st:
                        cur_s+=cw[c]
                    list_score.append(cur_s)
                print('\n-----')
                sort_strs=[x for _, x in sorted(zip(list_score, all_stri),
                    reverse=True)]
                all_stri = sort_strs[:max_len]
                with open(nfname, 'w', encoding='utf-8') as f:
                    for si, st in enumerate(all_stri):
                        f.write(st)
                        f.write('\n')
        # raise False
        # np.random.shuffle(all_stri)
        print('num name {}'.format(len(all_stri)))
        if combine_text>1:
            c=0
            new_all_stri=[]
            while c<len(all_stri)-combine_text:
                news=''
                for j in range(combine_text):
                    news+=all_stri[c+j]
                new_all_stri.append(news)
                c+=combine_text
            all_stri=new_all_stri

        for stri in all_stri:
            if len(stri)<max_char_per_line:
                self.corpus_lines.append(stri)
            else:
                rdl = np.random.randint(max_char_per_line-max_char_rand_range,
                    max_char_per_line+max_char_rand_range, 1)[0]
                stris=split_bylen(stri, rdl)
                for st in stris:
                    self.corpus_lines.append(st)
        print('num string {}'.format(len(self.corpus_lines)))


class PrintedLineGenerator(BaseOCRGenerator):
    """
    This generator assumes there are:
        (1) a collection of text files, from which the content of images will
            be generated;
        (2) a collection of font files

    Usage example:
    ```
    import os
    from dataloader.generate.image import PrintedLineGenerator

    lineOCR = PrintedLineGenerator()
    lineOCR.load_fonts('./fonts/')
    lineOCR.load_text_database('text.txt')

    # to get random characters
    X, widths, y = lineOCR.get_batch(4, 1800)

    # or to generate 100 images
    lineOCR.generate_images(start=0, end=100, save_dir='/tmp/Samples')
    ```
    """
    FONT_EXTENSIONS = ['otf', 'ttf', 'OTF', 'TTF']

    def __init__(self, height=64, num_workers=8, allowed_chars=None,
        helper=None, augmentor=None, is_binary=False, verbose=2):
        """Initialize the generator"""
        super(PrintedLineGenerator, self).__init__(height=height,
            helper=helper, allowed_chars=allowed_chars, verbose=verbose)

        # image configuration
        self.fonts = []

        # utility
        self.pool = (None if type(num_workers) != int
                     else multiprocessing.Pool(num_workers))
        self.augment = (
            PrintMildAugment(is_binary=is_binary, cval=255) if augmentor is None
            else augmentor(is_binary=is_binary, cval=255))
        self.helper = HandwritingHelper() if helper is None else helper

    def __getstate__(self):
        """Delete the pool from object state since pool cannot be picklable"""
        state = self.__dict__.copy()
        state.pop('pool', None)
        return state

    def _remove_unknown_characters(self, text):
        """Check for characters in the text that are not in database.

        # Arguments
            text [str]: the string to check

        # Returns
            [str]: the text that have missing characters removed
            [set of str]: missing characters
        """
        exist = []
        missing_chars = set([])
        for each_char in text:
            if not self.helper.is_char_exists(each_char):
                missing_chars.add(each_char)
            else:
                exist.append(each_char)

        return ''.join(exist), missing_chars

    def _is_char_supported_by_font(self, char, font_check):
        """Check whether the current font supported to draw `char`

        # Arguments
            char [str]: the character to check
            font_check [TTFont]: the font check object

        # Returns
            [bool]: True if the font support `char`, False otherwise
        """
        for cmap in font_check['cmap'].tables:
            if cmap.isUnicode():
                if ord(char) in cmap.cmap:
                    return True

        return False

    def _get_config(self, text, default_config={}):
        """Returns a configuration object for the text

        The idea is that each line of text will have a specific configuration,
        which then will be used during image generation. The configuration file
        has the following format: {
            config_key: config_value
            'text': [list of {}s with length == len(text), with each {} is a
                     config for that specific word]
        }

        The returning object contains these following keys:
            'text': a list of each character configuration

        Each character configuration object contains these following keys:
            'skewness': the skew angle of character
            'character_normalization_mode': an integer specifcy how to
                normalize resized characters
            'space': the space distance to the next character
            'bottom': the bottom padding
            'height': height of character (in pixel)
            'width_noise': the amount to multiply with character width

        # Arguments
            text [str]: the text to generate string
            default_config [dict]: the default config value

        # Returns
            [obj]: the configuration object
        """
        # config = {'angle': 0, 'text': []}
        # text_config = {'space': 0}

        ### General rules for each character
        is_space_close = random.random() > 0.8

        font = random.choice(self.fonts)
        font_draw = ImageFont.truetype(font, 60)
        font_check = TTFont(font)

        piecewise_augment = 1 if random.random() < 0.3 else 0
        if piecewise_augment == 1:
            piecewise_augment = 2 if random.random() < 0.5 else 1
        ### End general rules


        # each text configuration
        text_config = []
        last_bottom = 0
        for _idx, each_char in enumerate(text):

            # space
            if _idx == len(text) - 1:                   # last character
                space = 2
            elif _idx == 0:
                space = 0
            elif (not self._is_number(each_char)        # number separation
              and self._is_number(text[_idx-1])):
                space = random.randint(15, 25)
            elif ord(each_char) < 12800:                # non kanji chars
                space = random.randint(5, 15)
            else:
                if is_space_close:                      # other
                    space = random.randint(-4, 2)
                else:
                    space = random.randint(2, 25)

            text_config.append({
                'piecewise_augment': piecewise_augment,
                'space': space,
                'font_draw': font_draw,
                'font_check': font_check
            })

        return {
            'font': font,
            'text': text_config
        }

    def _generate_single_image(self, char, config):
        """Generate a character image

        # Arguments
            char [str]: a character to generate image
            config [obj]: a configuration object for this image

        # Returns
            [np array]: the generated image for that specific string
        """
        if not self._is_char_supported_by_font(char, config['font_check']):
            return None

        image = Image.new('L', (150, 100), 255)
        draw = ImageDraw.Draw(image)
        draw.text((15, 15), char, font=config['font_draw'], fill=0)

        image = np.array(image)

        if len(np.unique(image)) < 2:
            return None

        if config['piecewise_augment'] == 0:
            image = image
        elif config['piecewise_augment'] == 1:
            image = self.augment.char_piecewise_affine_1.augment_image(image)
            image = cv2.threshold(image, 0, 255,
                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        elif config['piecewise_augment'] == 2:
            image = self.augment.char_piecewise_affine_2.augment_image(image)
            image = cv2.threshold(image, 0, 255,
                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        image = trim_image_horizontally(image)
        return image

    def _generate_sequence_image(self, text, debug=True):
        """Generate string image of a given text

        # Arguments
            text [str]: the text that will be used to generate image

        # Returns
            [np array]: the image generated
            [str]: the text label
            [list of str]: list of missing characters
        """
        default_config = {}
        config = self._get_config(text, default_config)

        label = ''
        missing_chars = []
        base_image = np.zeros((100, len(text) * 100), np.uint8)
        current_horizontal = 0
        for _idx, each_char in enumerate(text):
            if each_char == ' ':
                current_horizontal += config['text'][_idx]['space']
                current_horizontal = max(0, current_horizontal)
                current_horizontal += random.randint(12, 25)
                label += each_char
                continue

            char_image = self._generate_single_image(each_char,
                config['text'][_idx])
            if char_image is not None:
                char_image = 255 - char_image

                current_horizontal += config['text'][_idx]['space']
                current_horizontal = max(0, current_horizontal)
                next_horizontal = current_horizontal + char_image.shape[1]
                base_image[:, current_horizontal:next_horizontal] = (
                    np.bitwise_or(
                        char_image,
                        base_image[:, current_horizontal:next_horizontal]))
                current_horizontal = next_horizontal
                label += each_char
            else:
                missing_chars.append(each_char)

        image = 255 - base_image
        if len(np.unique(image)) < 2:
            if debug:
                return None, label, {
                    'missing_chars': missing_chars
                }
            else:
                return None, label

        image = crop_image(image)
        image = self.augment.augment_line(image)
        if self.is_binary:
            image = cv2.threshold(
                image, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        if debug:
            return image, label, {
                'missing_chars': missing_chars
            }

        return image, label

    def _batching_image(self, image, new_width, max_width):
        """Prepare and normalize size image for batching

        # Arguments
            image [np array]: the image to process
            new_width [int]: the new width of the image (when image is resized
                to have `self.height` height)
            max_width [int]: the maximum width that an image can have

        # Returns
            [np array]: the image with fixed width (black stroke on white
                background)
        """
        resized_image = cv2.resize(image, (new_width, self.height),
                interpolation=cv2.INTER_LINEAR)
        image = (np.ones([self.height, max_width], dtype=np.uint8)
            * 255).astype(np.uint8)
        image[:, :new_width] = resized_image

        return image

    def load_fonts(self, font_folder):
        """Load all the font files recursively in the font folder

        # Arguments
            font_folder [str]: the folder containing all the fonts
        """
        self.fonts = []
        for each_ext in self.FONT_EXTENSIONS:
            self.fonts += glob.glob(
                os.path.join(font_folder, '**', '*.{}'.format(each_ext)),
                recursive=True)

        if len(self.fonts) == 0:
            print(':WARNING: no font loaded from {}'.format(font_folder))
        else:
            if self.verbose > 2:
                print(':INFO: {} fonts loaded'.format(len(self.fonts)))

    def load_text_database(self, file_paths):
        """Load the text database from which to generate line strings

        # Arguments
            file_paths [str]: the path to text file. If the path is a folder
                then load all `.txt` file in the folder path

        # Returns
            [int]: the number of text lines
        """
        if os.path.isfile(file_paths):
            file_paths = [file_paths]
        elif os.path.exists(file_paths):
            file_paths = glob.glob(os.path.join(file_paths, '*.txt'))
        else:
            raise ValueError('invalid `file_path`: {}'.format(file_paths))

        for each_path in file_paths:
            with open(each_path, 'r', encoding="utf-8") as f_text:
                for each_row in f_text:
                    if len(each_row) < 2:
                        continue

                    line_string = normalize_text(
                        each_row.strip(), self.conversion_table)
                    self.corpus_lines.append(line_string)

        return len(self.corpus_lines)

    def load_text_database_avd(self, file_path, is_random=False, top_freq=True,
        max_len=-1, ignore_strange=True, combine_text=1, max_char_per_line=20,
        max_char_rand_range=5, random_space_limit=5):

        def split_bylen(item, maxlen):
            """
            Requires item to be sliceable (with __getitem__ defined)
            """
            return [
                item[ind:ind + maxlen] for ind in range(0, len(item), maxlen)
            ]

        nfname = file_path[:-4] + '_{}.txt'.format(max_len)
        all_stri = []
        if top_freq and not is_random and os.path.isfile(nfname):
            txt_file=nfname
            max_len=-1
        cline=0
        try:
            with open(file_path) as f:
                for l in f:
                    if max_len > 0 and cline > max_len * random_space_limit:
                        break
                    cline+=1
                    l = unicodedata.normalize('NFKC', l)
                    if ignore_strange and len(l)>0 and l[0] in self.char_2_label:
                        all_stri.append(normalize_text(
                    l.strip(), self.conversion_table))
        except:
            with open(file_path, encoding='utf-8') as f:
                for l in f:
                    if max_len > 0 and cline > max_len * random_space_limit:
                        break
                    cline+=1
                    l = unicodedata.normalize('NFKC', l)
                    if ignore_strange and len(l) > 0 and l[0] in self.char_2_label:
                        all_stri.append(normalize_text(
                    l.strip(), self.conversion_table))
        if max_len>0:
            if not top_freq:
                if not is_random:
                    all_stri=all_stri[:max_len]
                else:
                    all_stri=np.random.choice(all_stri, max_len)
                    #print(self.all_stri)
            else:
                cw={}
                for si, st in enumerate(all_stri):
                    if si%100==0:
                        print('\rcounted {}/{}'.format(si, len(all_stri)))
                    for c in st:
                        if c not in cw:
                            cw[c]=0
                        cw[c]+=1
                list_score=[]
                print('\n-----')
                for si, st in enumerate(all_stri):
                    if si%100==0:
                        print('\rscored {}/{}'.format(si, len(all_stri)))
                    cur_s=0
                    for c in st:
                        cur_s+=cw[c]
                    list_score.append(cur_s)
                print('\n-----')
                sort_strs=[x for _, x in sorted(zip(list_score, all_stri), reverse=True)]
                all_stri = sort_strs[:max_len]
                with open(nfname, 'w', encoding='utf-8') as f:
                    for si, st in enumerate(all_stri):
                        if si % 100 == 0:
                            print('\rwrited {}/{}'.format(si, len(all_stri)))
                        f.write(st)
                        f.write('\n')
        # raise False
        # np.random.shuffle(all_stri)
        print('num name {}'.format(len(all_stri)))
        if combine_text>1:
            c=0
            new_all_stri=[]
            while c<len(all_stri)-combine_text:
                news=''
                for j in range(combine_text):
                    news+=all_stri[c+j]
                new_all_stri.append(news)
                c+=combine_text
            all_stri=new_all_stri

        for stri in all_stri:
            if len(stri)<max_char_per_line:
                self.corpus_lines.append(stri)
            else:
                rdl=np.random.randint(max_char_per_line-max_char_rand_range,
                                           max_char_per_line+max_char_rand_range, 1)[0]
                stris=split_bylen(stri, rdl)
                for st in stris:
                    self.corpus_lines.append(st)
        print('num string {}'.format(len(self.corpus_lines)))
        return len(self.corpus_lines)

    def load_background_image_files(self, folder_path):
        """Load background image files

        # Arguments
            folder_path [str]: the path to folder that contains background
        """
        if self.is_binary:
            print(':WARNING: background image files are not loaded for binary '
                  'generation mode.')
        else:
            self.augment.add_background_image_noises(folder_path)

    def initialize(self):
        """Initialize the generator

        This method will be called when all text and image files are loaded. It
        will:
            1. check for missing characters
            2. construct label_2_char, char_2_label
            3. randomize corpus_lines
        """
        if self.allowed_chars is None:
            print(':INFO: `ALLOWED_CHARS` is not supplied, automatically '
                  'becomes all characters in text corpus.')
            self.allowed_chars = set([])
            for each_line in self.corpus_lines:
                self.allowed_chars.update(set(each_line))

        super(PrintedLineGenerator, self).initialize()

    def get_batch(self, batch_size, max_width=-1, **kwargs):
        """Get the next data batch.

        # Arguments
            batch_size [int]: the size of the batch
            max_width [int]: the maximum width that an image can have. If this
                value is smaller than 1, then `max_width` is automatically
                calculated
            **append_channel [bool]: if True, then pad the image
            **label_converted_to_list [bool]: whether label should be in list
                form (converted from char_2_label). Otherwise label is just a
                text string

        # Returns
            [list 4D np.ndarray]: list of binary image (0: background)
            [list of strs]: the label of each image
            [list of ints]: the width of each image
        """
        if max_width <= 0:
            max_width = float('inf')

        # get the images and determine its eligibility
        if self.iterations >= len(self.corpus_lines):
            # shuffle the corpus every iterations
            random.shuffle(self.corpus_lines)
            self.corpus_lines.sort(key=lambda each_line: len(each_line))
            self.iterations = 0
        self.iterations += batch_size

        # get the images and determine their eligibility
        index = random.choice(self.corpus_size)
        _temp_store = []
        debug_info = []
        if self.pool is not None:
            _temp_store = self.pool.starmap(
                self._gen_image_each_process,
                [(index+_idx, batch_size, max_width)
                    for _idx in range(batch_size)]
            )
        else:
            while len(_temp_store) < batch_size:
                index = (index + 1) % self.corpus_size
                each_string = self.corpus_lines[index]
                image, chars, each_debug_info = self._generate_sequence_image(
                    text=each_string, debug=True)

                if image is None or len(chars) == 0 or len(np.unique(image))<2:
                    continue

                original_height, original_width = image.shape
                new_width = original_width * self.height // original_height
                if new_width > max_width:
                    continue

                _temp_store.append((image, chars, new_width))
                debug_info.append(each_debug_info)

        batch_images = []
        batch_labels = []
        batch_widths = []
        if max_width == float('inf'):
            if len(_temp_store) == 0:
                raise ValueError("_temp_store is empty, no image is appended")
            max_width = max(_temp_store, key=lambda obj: obj[2])[2] + 1

        # Process the image
        for each_image, each_label, new_width in _temp_store:
            image = self._batching_image(each_image, new_width, max_width)
            image = self.helper.postprocess_image(image, **kwargs)
            label = self.helper.postprocess_label(each_label, **kwargs)

            batch_images.append(image)
            batch_labels.append(label)
            batch_widths.append(new_width)

        if debug:
            return self.helper.postprocess_outputs(
                batch_images, batch_widths, batch_labels
            ), debug_info
        
        return self.helper.postprocess_outputs(
            batch_images, batch_widths, batch_labels
        )

    def load_long_text_database(self, file_path, is_random=False,
      top_freq=False, max_len=-1, ignore_strange=True, combine_text=1,
      max_char_per_line=20, max_char_rand_range=2):
        """ Read the text file for synthetic generation

        # Arguments
            file_path [string]: path to txt file
            is_random [bool]: if true, choose random max_len lines in the file
            max_len [int]: max number of line chosen in txt file,
                default=-1 means getting all lines
            top_freq [bool]: get lines that contain top frequent words
            ignore_strange [bool]: if True, ignore the line that has first
                character not found in vocab
            combine_text [int]: combine many text in consecutive lines into
                one longer text
            max_char_per_line [int]: limit the length of text line
            max_char_rand_range [int]: range of random allowed around the
                max_char_per_line length
        """
        def split_bylen(item, maxlen):
            """Requires item to be sliceable (with __getitem__ defined)"""
            return [item[ind:ind + maxlen] for ind in
                range(0, len(item), maxlen)]

        nfname = file_path[:-4] + '_{}.txt'.format(max_len)
        all_stri = []
        if top_freq and not is_random and os.path.isfile(nfname):
            txt_file=nfname
            max_len=-1
        cline=0
        try:
            with open(txt_file) as f:
                for each_row in f:
                    if max_len>0 and cline>max_len*10:
                        break
                    l  = normalize_text(
                    each_row.strip(), self.conversion_table)
                    if ignore_strange and len(l)>0:
                        all_stri.append(str(l).strip())
                    cline+=1
        except:
            with open(file_path, 'rt', encoding='utf-8') as f:
                for each_row in f:
                    if max_len>0 and cline>max_len*10:
                        break
                    l = normalize_text(
                        each_row.strip(), self.conversion_table)
                    if ignore_strange and len(l) > 0:
                        all_stri.append(str(l).strip())
                    cline += 1
        print(len(all_stri))
        if max_len>0:
            if not top_freq:
                if not is_random:
                    all_stri=all_stri[:max_len]
                else:
                    all_stri=np.random.choice(all_stri, max_len)
                    #print(self.corpus_lines)
            else:
                cw={}
                for si, st in enumerate(all_stri):
                    # if si%100==0:
                    #     llprint('\rcounted {}/{}'.format(si, len(all_stri)))
                    for c in st:
                        if c not in cw:
                            cw[c]=0
                        cw[c]+=1
                list_score=[]
                print('\n-----')
                for si, st in enumerate(all_stri):
                    # if si%100==0:
                    #     llprint('\rscored {}/{}'.format(si, len(all_stri)))
                    cur_s=0
                    for c in st:
                        cur_s+=cw[c]
                    list_score.append(cur_s)
                print('\n-----')
                sort_strs=[x for _, x in sorted(zip(list_score, all_stri),
                    reverse=True)]
                all_stri = sort_strs[:max_len]
                with open(nfname, 'w', encoding='utf-8') as f:
                    for si, st in enumerate(all_stri):
                        f.write(st)
                        f.write('\n')
        # raise False
        # np.random.shuffle(all_stri)
        print('num name {}'.format(len(all_stri)))
        if combine_text>1:
            c=0
            new_all_stri=[]
            while c<len(all_stri)-combine_text:
                news=''
                for j in range(combine_text):
                    news+=all_stri[c+j]
                new_all_stri.append(news)
                c+=combine_text
            all_stri=new_all_stri

        for stri in all_stri:
            if len(stri)<max_char_per_line:
                self.corpus_lines.append(stri)
            else:
                rdl = np.random.randint(max_char_per_line-max_char_rand_range,
                    max_char_per_line+max_char_rand_range, 1)[0]
                stris=split_bylen(stri, rdl)
                for st in stris:
                    self.corpus_lines.append(st)
        print('num string {}'.format(len(self.corpus_lines)))

    def _gen_image_each_process(self, start_text_index, batch_size, max_width):
        """Get an image given a text index

        The text index used to generated image will be determined by:
            text_index + {enum} * batch_size
        Rationale: since a text might not be valid to create an image, we must
        skip it and choose a text. The rule above ensures that the next chosen
        text will not duplicate with text of other process, and still be close
        enough (in length) with the initial assigned text.

        # Argument
            start_text_index [int]: the start index used to calculate string
            process_id [int]: the process id

        # Returns
            image [np array]: the image
            label [str]: the label
            new_width [int]: the width of the image
        """
        # to make sure randomization within child images
        random.seed()
        self.augment.reseed()

        _idx = 0
        while True:
            text_index = start_text_index + _idx * batch_size
            text_index = text_index % self.corpus_size
            text = self.corpus_lines[text_index]
            image, label, _ = self._generate_sequence_image(text, debug=True)

            if image is None or len(label) == 0 or len(np.unique(image)) < 2:
                _idx += 1
                continue

            original_height, original_width = image.shape
            new_width = original_width * self.height // original_height
            if new_width > max_width:
                _idx += 1
                continue

            return image, label, new_width


class FontLineGenerator(PrintedLineGenerator):

    def _is_char_supported_by_font(self, char, font_check, font_draw=None):
        """Check whether the current font supported to draw `char`

        # Arguments
            char [str]: the character to check
            font_check [TTFont]: the font check object
            font_draw [Image Font]: font object capable of drawing

        # Returns
            [bool]: True if the font support `char`, False otherwise
        """
        cmap_check = False
        for cmap in font_check['cmap'].tables:
            if cmap.isUnicode():
                if ord(char) in cmap.cmap:
                    cmap_check = True
                    break

        # skip this checking if font_draw is not supplied
        non_blank_image_check = True if font_draw is None else False
        if not non_blank_image_check:
            image = Image.new('L', (200, 200), 255)
            draw = ImageDraw.Draw(image)
            draw.text((15, 15), char, font=font_draw, fill=0)
            image = np.array(image, dtype=np.uint8)
            if len(np.unique(image)) > 1:
                non_blank_image_check = True

        return cmap_check and non_blank_image_check

    def _get_config(self, text, default_config={}):
        """Returns a configuration object for the text

        The idea is that each line of text will have a specific configuration,
        which then will be used during image generation. The configuration file
        has the following format: {
            config_key: config_value
            'text': [list of {}s with length == len(text), with each {} is a
                     config for that specific word]
        }

        # Arguments
            text [str]: the text to generate string
            default_config [dict]: the default config value

        # Returns
            [obj]: the configuration object
        """
        font = (random.choice(self.fonts)
            if 'font' not in default_config
            else default_config['font'])
        font_draw = ImageFont.truetype(font, 80)
        font_check = TTFont(font)

        return {
            'font': font,
            'font_draw': font_draw,
            'font_check': font_check
        }

    def _generate_sequence_image(self, text, debug=True, font=None):
        """Generate string image of a given text

        # Arguments
            text [str]: the text that will be used to generate image
            debug [bool]: whether to include debug information in the output
            font [idx]: the specific font to generate

        # Returns
            [np array]: the image generated
            [str]: the text label
            [list of str]: list of missing characters
        """
        default_config = {} if font is None else {'font': font}
        config = self._get_config(text, default_config)

        label = ''
        missing_chars = []

        # initial check if all characters in text is supported by fonts
        for each_char in text:
            if self._is_char_supported_by_font(
                char=each_char,
                font_check=config['font_check'],
                font_draw=config['font_draw']):
                label += each_char
            else:
                missing_chars.append(each_char)

        if len(label) == 0:
            if debug:
                return None, label, {
                    'missing_chars': missing_chars
                }
            return None, label

        # draw the image
        base_image = Image.new('L', (len(label) * 150, 800), 255)
        draw = ImageDraw.Draw(base_image)
        draw.text((10, 200), label, font=config['font_draw'], fill=0)

        image = np.asarray(base_image, dtype=np.uint8)
        image = crop_image(image)
        image = self.augment.augment_line(image)
        if self.is_binary:
            image = cv2.threshold(
                image, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        if debug:
            return image, label, {
                'font': config['font'],
                'missing_chars': missing_chars
            }

        return image, label


class HandwrittenCharacterGenerator(HandwrittenLineGenerator):
    """This generator outputs handwritten character"""

    def __init__(self, height=64, width=64, helper=None,
                 limit_per_char=float('inf'), verbose=2, allowed_chars=None,
                 is_binary=False, augmentor=None):
        """Intialize the generator object"""
        super(HandwrittenCharacterGenerator, self).__init__(
            height=height, helper=helper, limit_per_char=limit_per_char,
            verbose=verbose, allowed_chars=allowed_chars,
            is_binary=is_binary, augmentor=augmentor
        )
        self.width = 64
        self.augment = HandwritingCharacterAugment(is_binary=is_binary)

    def _get_config(self, text, default_config={}):
        """Returns a configuration object for the text.

        Not necessary for character generation

        # Arguments
            text [str]: the text to generate string
            default_config [dict]: the default config value

        # Returns
            [obj]: the configuration object
        """
        return {}

    def _generate_single_image(self, char):
        """Generate a character image

        # Arguments
            char [str]: a character to generate image

        # Returns
            [np array]: the generated image for that specific string
        """
        if char not in self.char_2_imgs.keys():
            raise ValueError(
                'invalid char {}. If you want to generate this character, '
                'add it to `allowed_char`'.format(char))

        choice = random.choice(len(self.char_2_imgs[char]))
        image = self.char_2_imgs[char][choice]
        # resize image
        if char not in constants.NOT_RESIZE:
            height, width = image.shape

            if char in ('゙', '゚'):
                desired_height = random.randint(
                    int(height / 2),
                    int(height * 1.5)
                )
            else:
                desired_height = random.randint(
                    max(int(height / 2), 22),
                    min(max(height * 1.5, 23), 50)
                )
            desired_width = int(
                width * (desired_height / height) * random.uniform(0.9, 1.0))

            if desired_width < 64:
                image = self._resize_character(
                    image, desired_height,
                    desired_width, random.randint(0, 2))

        try:
            image = self.augment.augment_character.augment_image(image)
        except:
            print("Ignoring augmentation for character {}".format(char))
        height, width = image.shape
        if max(height, width) >= 64:
            ratio = 50 / max(height, width)
            image = cv2.resize(image, None, fx=ratio, fy=ratio,
                interpolation=self.interpolation)

        # add horizontal space and bottom space
        height, width = image.shape
        top_pad = 0 if height == 64 else random.randint(0, 64 - height)
        bottom_pad = 64 - height - top_pad
        left_pad = 0 if width == 64 else random.randint(0, 64 - width)
        right_pad = 64 - width - left_pad
        image = np.pad(image, ((top_pad, bottom_pad), (left_pad, right_pad)),
                    'constant', constant_values=self.background_value)
        return image

    def _resize_character(self, image, desired_height, desired_width,
        character_normalization_mode=4):
        """Resize and normalize the character

        This method optionally normalizes the characters, so that the affect
        of resizing characters do not have a bias affects on the model.
        Sometimes we can skip normalization to provide more noise effects.

        # Arguments
            image [np array]: the character image to resize
            desired_height [int]: the desired height to resize the image into
            desired_width [int]: the desired width to resize the image into
            character_normalization_mode [int]: to have value of 0-3, variate
                the normalization scheme

        # Returns
            [np array]: the resized character image
        """
        original_height, original_width = image.shape
        ratio = desired_height / original_height

        # Resize the character
        image = cv2.resize(image, (desired_width, desired_height),
            interpolation=self.interpolation)

        # Adjust the stroke width, color, and deblur the result with some
        # randomness to enhance model robustness
        if character_normalization_mode > 0:
            image = adjust_stroke_width(image, ratio, is_binary=self.is_binary)

        # if character_normalization_mode > 1:
        #     image = normalize_grayscale_color(image)

        # if character_normalization_mode > 2:
        #     image = unsharp_masking(image)

        return image

    def load_text_database(self, file_path):
        """Need not loading the text database"""
        if verbose > 2:
            print(
                ':INFO: the character generator needs no text. The texts '
                'are given inside `allowed_chars`.'
            )

        return len(self.corpus_lines)

    def initialize(self, train_percent=0.8, val_percent=0.1):
        """Initialize the generator

        This method will be called when all text and image files are loaded. It
        will:
            1. check for missing characters
            2. construct validation and test characters
            3. corpus_lines are basically allowed characters

        # Arguments
            train_percent [float]: percentage of training characters
            val_percent [float]: percentage of validation characters
        """
        super(HandwrittenCharacterGenerator, self).initialize()
        self.corpus_lines = list(self.allowed_chars)

    def get_batch(self, batch_size=32, char=None,
        mode=constants.TRAIN_MODE):
        """Get a batch of images

        # Arguments
            batch_size [int]: the size of batch
            char [str]: whether to generate a specific character. If None random
        """
        if mode == constants.TEST_MODE:
            self.char_2_imgs = self.char_2_imgs_test
        elif mode == constants.VALIDATION_MODE:
            self.char_2_imgs = self.char_2_imgs_val
        else:
            self.char_2_imgs = self.char_2_imgs_train

        images, labels = [], []
        while len(images) < batch_size:
            label = random.choice(self.corpus_lines) if char is None else char
            image = self._generate_single_image(label)
            if image is None:
                print(':WARNING: problem with generating char {}'.format(label))
                continue
            images.append(image)
            labels.append(label)

        return self.helper.postprocess_outputs(images, labels)

    def get_batch_from_files(self, folder_path, label_path, binarized=False):
        """Get a batch of images from files

        # Arguments
            folder_path [str]: the path to folder
            label_path [str]: the path to label file
            binarized [bool]: whether the file images are already binarized

        # Returns
            [list of np images]: the list of images
            [list of str]: the list of corresponding labels
        """

        image_files = glob.glob(os.path.join(folder_path, '*.png'))
        images, labels = [], []
        with open(label_path, 'rb') as f:
            label_obj = json.load(f)

        for each_image_file in image_files:
            filename = os.path.basename(each_image_file)
            image = cv2.imread(each_image_file, cv2.IMREAD_GRAYSCALE)

            if not binarized:
                image = cv2.GaussianBlur(image, (3,3), 0)
                image = cv2.threshold(image, 0, 1,
                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            height, width = image.shape
            if max(height, width) >= 64:
                ratio = 50 / max(height, width)
                image = cv2.resize(image, None, fx=ratio, fy=ratio,
                    interpolation=self.interpolation)

            height, width = image.shape
            top_pad = int((64 - height) / 2)
            bottom_pad = 64 - height - top_pad
            left_pad = int((64 - width) / 2)
            right_pad = 64 - width - left_pad
            image = np.pad(
                image, ((top_pad, bottom_pad), (left_pad, right_pad)),
                'constant', constant_values=self.background_value)

            images.append(image)
            labels.append(label_obj[filename])

        return self.helper.postprocess_outputs(images, labels)

    def generate_images(self):
        pass


# for backward compatibility
LineOCRGenerator = HandwrittenLineGenerator


def generate_printed(fonts, corpus, output_folder=None):
    """Quickly generate images using printed scheme

    # Arguments
        fonts [str]: path to folder containing fonts
        corpus [str]: path to corpus strings
        output_folder [str]: path to output folder
    """
    generator = PrintedLineGenerator()
    generator.load_fonts(fonts)
    generator.load_text_database(corpus)
    generator.initialize()
    generator.generate_images(save_dir=output_folder, label_json=True)


def generate_handwritten(chars, corpus, output_folder=None):
    """Quick generate images using handwritten scheme

    # Arguments
        chars [str]: path to pkl file or folder containing pkl char files
        corpus [str]: path to corpus string
        output_folder [str]: path to output folder
    """
    generator = HandwrittenLineGenerator()

    if os.path.isfile(chars):
        generator.load_character_database(chars)
    else:
        for each_pkl in glob.glob(os.path.join(chars, '*.pkl')):
            generator.load_character_database(each_pkl)

    generator.load_text_database(corpus)
    generator.initialize()
    generator.generate_images(save_dir=output_folder, label_json=True)


if __name__ == '__main__':
    """Run this block if the code is directly called"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help='Generation mode: printed or handwritten')
    parser.add_argument('--output-folder', default='',
        help='Folder containing generated images')
    parser.add_argument('--chars', default='',
        help='Pkl file or folder of pkl files containing character images')
    parser.add_argument('--fonts', default='',
        help='Folder containing font')
    parser.add_argument('--corpus', default='',
        help='Path to text file containing the text content to generate '
               'images')

    args = parser.parse_args()
    args.output_folder = args.output_folder if args.output_folder else None
    if args.mode == 'printed':
        generate_printed(args.fonts, args.corpus, args.output_folder)
    elif args.mode == 'handwritten':
        generate_handwritten(args.chars, args.corpus, args.output_folder)
    else:
        raise AttribtueError('the mode must either be `printed` or '
                             '`handwritten`')
