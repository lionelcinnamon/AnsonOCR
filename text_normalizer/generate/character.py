# @author: John Nguyen
# TODO: support for checking these font types: ttc,
# =============================================================================
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import glob
import os
import uuid
from collections import defaultdict

import numpy as np
from fontTools.ttLib import TTFont
from fontTools.ttLib.sfnt import readTTCHeader
from PIL import Image, ImageDraw, ImageFont

from dataloader.utils.misc import dump_json

CHAR_LIST = ["搗", "搦", "搶", "摑", "摠", "摯", "摶", "撈", "撓", "撥", "擅"]

# FONT_EXTENSIONS = ['eot', 'otf', 'ttf', 'ttc', 'woff']
FONT_EXTENSIONS = ['otf', 'ttf']


class FontGenerator(object):
    """Class to generate character image from font

    Current supported font type: ttf, otf.
    # TODO: support for ttc

    # Example:
        f_gen = FontGenerator('fonts/YuMincho.ttf')
        image = f_gen.generate_image('辨', to_file=True)
    """
    def __init__(self, font_path, output_dir='.', point_size=44,
      image_size=(64,64)):
        """Initialize an image generator from font.

        Note: `font_draw` and `font_check` are lists because certain
        fonts contain more than 1 character sets.

        # Arguments
            font_path [str]: the path to font file
            output_dir [str]: the path to store generated image
            point_size [int]: font size
            image_size [tuple of 2 ints]: the image size
        """
        self.point_size = point_size
        self.image_size = image_size
        if min(self.image_size) < self.point_size:
            print(':WARNING: font {} is potentially larger than image {}'
                .format(self.point_size, self.image_size))

        self._font_path = font_path
        self.font_draw = ImageFont.truetype(font_path, self.point_size)
        self.font_check = TTFont(font_path)
        self.output_dir = output_dir

    # def _load_font(self, font_path):
    #     """Load the font into `font_draw` and `font_check`

    #     # Arguments
    #         font_path [str]: the string path to font
    #     """
    #     self._font_draw, self._font_check = [], []
    #     if os.path.splitext(font_path)[1].lower() == 'ttc':
    #         with open(font_path, 'rb') as f:
    #             num_fonts = readTTCHeader(f).
    #     else:
    #         self._font_draw.append(
    #             ImageFont.truetype(font_path), self.point_size)
    #         self._font_check.append(TTFont(font_path))


    def check_font_support(self, char):
        """Check whether the current font supported to draw `char`

        # Arguments
            char [str]: the character to check

        # Returns
            [bool]: True if the font support `char`, False otherwise
        """
        for cmap in self.font_check['cmap'].tables:
            if cmap.isUnicode():
                if ord(char) in cmap.cmap:
                    return True

        return False

    def generate_image(self, char, to_file=False, sep='_'):
        """Generate image of a character given a character

        # Arguments
            char [str]: the character text (should have length == 1)
            to_file [bool]: whether to save to external file
            sep [chr]: whether to check

        # Returns
            [np array]: the image as numpy array
            [str]: if to_file is true, retrieve the filename, else empty string
        """
        if len(char) != 1:
            raise ValueError('`char` should be a character, but get {} instead'
                .format(char))

        if not self.check_font_support(char):
            print(':WARNING: {} does not support {}. SKIP'
                .format(self._font_path, char))
            return None, ''

        image = Image.new('RGB', (64, 64), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), char, font=self.font_draw, fill=(0,0,0))
        image_np = np.asarray(image, dtype=np.uint8)

        if to_file:
            if len(np.unique(image_np)) <= 1:
                print(':WARNING: image only have 1 unique pixel value')
                return image_np, ''

            filename = os.path.join(
                self.output_dir,
                '{}{}{}.png'.format(char, sep, uuid.uuid4().hex))
            image.save(filename)

            return image_np, filename

        return image_np, ''


def generate_images_from_font_folder(char_list, font_folder, output_dir,
    debug=False):
    """Generate a list of characters from all fonts in a font folder

    # Arguments
        char_list [list of str]: list of characters
        font_folder [str]: path to folder containing fonts
        output_dir [str]: path to folder containing the output image
        debug [bool]: whether to save image information (default False)

    # Returns
        [list of np array]: the list of images
        [list of str]: the list of corresponding characters
        [list of str]: the list of missing characters (characters which are
            not supported by all fonts in the `font_folder`)
    """
    missing_chars_count = defaultdict(int) # initialize counting to 1
    for each_char in char_list: missing_chars_count[each_char] += 1

    fonts = []
    for each_font in FONT_EXTENSIONS:
        fonts += glob.glob(
            os.path.join(font_folder, '**', '*.{}'.format(each_font)),
            recursive=True)

    images, labels = [], []
    debug_info = {}

    for each_font in fonts:
        print('Generating from font {}...'.format(each_font))
        f_gen = FontGenerator(each_font, output_dir=output_dir)
        for each_char in char_list:
            image, filepath = f_gen.generate_image(each_char, to_file=True)
            debug_info[filepath] = each_font

            if image is not None:
                missing_chars_count[each_char] += 1
                images.append(image)

    missing_chars = [
        char for char, count in missing_chars_count.items()
        if count == 1]

    if debug:
        dump_json(debug_info, os.path.join(output_dir, 'debug.json'))
        if len(missing_chars) > 0:
            print('Missing {} chars, which are: {}'.format(
                len(missing_chars), missing_chars))


    return images, labels, missing_chars

