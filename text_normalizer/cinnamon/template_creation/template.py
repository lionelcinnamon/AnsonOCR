"""Generate template files, along with the corresponding labels

@author: John
"""

import json
import math
import os
import subprocess
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


class Elements(object):
    """Base class to create elements"""

    @staticmethod
    def create_single_box(width, height, dot_length=None, dot_space=None):
        """Create character box.

        Suggested argument values:
        - For single char: width 31, height 40, dot_length 6, dot_space 4
        - For a line: width varying, height 60, dot length 5, dot space 4

        # Arguments
            width [int]: the base width
            height [int]: the base height
            dot_length [int]: if dotted, the length of each dot
            dot_space [int]: if dotted, the space between dots

        # Returns
            [np array]: the dotted images
        """
        image = (np.ones((height, width)) * 255).astype(np.uint8)
        image = np.pad(image, [[2, 2], [2, 2]], mode='constant',
                       constant_values=0)

        if (dot_length is None and dot_space is not None or
                dot_length is not None and dot_space is None):
            raise AttributeError('both `dot_length` and `dot_space` must be '
                                 'set or unset')

        if dot_length is None and dot_space is None:
            return image

        height, width = image.shape
        for pixel in range(dot_length, width-dot_length, dot_length+dot_space):
            space_range = min(pixel+dot_space, width-2)
            image[:, pixel:space_range] = 255

        for pixel in range(dot_length, height-dot_length, dot_length+dot_space):

            space_range = min(pixel+dot_space, height-2)
            image[pixel:space_range, :] = 255

        return image

    @staticmethod
    def group_boxes_horizontally(box, num_boxes, space=6):
        """Create a group of boxes

        Suggested argument space 6, box created by default `create_single_box`

        # Arguments
            box [np image]: each box
            num_boxes [int]: number of boxes in each group
            space [int]: the space between characters

        # Returns
            [np image]: the group of boxes
        """
        elements = []
        height, _ = box.shape
        for _idx in range(num_boxes):
            elements.append(box)
            if _idx != num_boxes - 1:
                elements.append((np.ones((height, space)) * 255)
                                .astype(np.uint8))
        image = np.concatenate(elements, axis=1)
        return image


class BaseTemplate(object):
    """Initialize the template"""

    def __init__(self, text_file, n_rows, n_columns, n_pages=None,
                 font_cjk=None, font_latin=None,
                 size=(1651, 1275), original=None, header=None,
                 footer=None):
        """Initialize the object"""

        self.text = []
        with open(text_file, 'r') as f_in:
            for each_row in f_in:
                self.text.append(each_row.strip())

        self.n_rows = n_rows
        self.n_columns = n_columns
        self.n_pages = (
            math.ceil(len(self.text) / (n_rows * n_columns))
            if n_pages is None else n_pages)

        self.header = header
        self.footer = footer

        self.page = (
            original if original is not None else
            (np.ones(size) * 255).astype(np.uint8))

        self.artifact_dir = os.path.join(
            os.path.expanduser('~'),
            '.dataloader/template_creation')
        os.makedirs(self.artifact_dir, exist_ok=True)

        font_cjk = (os.path.join(self.artifact_dir, 'NotoSansCJKjp-Light.otf')
            if font_cjk is None
            else font_cjk)
        font_latin = (os.path.join(self.artifact_dir, 'NotoSans-Light.ttf')
            if font_latin is None
            else font_latin)
        self.font_cjk = ImageFont.truetype(font_cjk, 20)
        self.font_latin = ImageFont.truetype(font_latin, 20)

    def _basic_char_block(self, index):
        """Ready-to-use template for char block with label

        # Arguments
            index [int]: the block index

        # Returns
            [np array]: the constructed block
        """
        # Generate the box
        small_box = Elements.create_single_box(
            width=31, height=40, dot_length=6, dot_space=4)
        box_group = Elements.group_boxes_horizontally(small_box, 7)
        box_group = np.pad(box_group, [[50, 0], [0, 0]], mode='constant',
                           constant_values=255)

        # Generate the text
        text = self.text[index]
        image = Image.fromarray(box_group)
        draw = ImageDraw.Draw(image)
        draw.text((5, 5), text, 0, self.font_cjk)

        return np.asarray(image, dtype=np.uint8), text

    def _basic_single_block(self, index, height=60):
        """Ready-to-use template for single block with label

        # Arguments
            index [int]: the block index
            height [int]: the height for text box (optional)

        # Returns
            [np array]: the constructed block
        """
        _, width = self.page.shape

        # Generate the box
        small_box = Elements.create_single_box(
            width=int((width - 40) / self.n_columns),
            height=height)
        box_group = np.pad(small_box, [[50, 0], [0, 0]], mode='constant',
                           constant_values=255)

        # Generate the text
        text = self.text[index]
        image = Image.fromarray(box_group)
        draw = ImageDraw.Draw(image)
        draw.text((5, 5), text, 0, self.font_cjk)

        return np.asarray(image, dtype=np.uint8), text

    def create_basic_block(self, index):
        """Create the basic block

        # Examples
        ```
        # Generate the box
        small_box = Elements.create_single_box(
            width=31, height=40, dot_length=6, dot_space=4)
        box_group = Elements.group_boxes_horizontally(small_box, 7)
        box_group = np.pad(box_group, [[50, 0], [0, 0]], mode='constant',
                           constant_values=255)

        # Generate the text
        text = self.text[index]
        image = Image.fromarray(box_group)
        draw = ImageDraw.Draw(image)
        draw.text((5, 5), text, 0, self.font_cjk)

        return np.asarray(image, dtype=np.uint8), text
        ```

        # Argument
            index [int]: the index of block on the page, so that label can be
                accessed

        # Returns
            [np array]: the basic block
        """
        raise NotImplementedError('use Elements')

    def construct_header(self, text, header_width, header_height):
        """Construct the header

        # Arguments
            text [str]: header text
            header_width [int]: the width of header
            header_height [int]: the height of header

        # Returns
            [np array]: the image containing text
        """
        image = (np.ones((header_height, header_width)) * 255).astype(np.uint8)
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        draw.text((10, 30), text, 0, self.font_latin)

        return np.asarray(image, dtype=np.uint8)

    def construct_footer(self, text, footer_width, footer_height):
        """Construct the footer

        # Arguments
            text [str]: header text
            footer_width [int]: the width of header
            footer_height [int]: the height of header

        # Returns
            [np array]: the image containing text
        """
        image = (np.ones((footer_height, footer_width)) * 255).astype(np.uint8)
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        draw.text((10, 30), text, 0, self.font_latin)

        return np.asarray(image, dtype=np.uint8)

    def create(self, output_folder, to_pdf=False, verbose=True):
        """Create the result

        # Arguments
            output_folder [str]: path to output folder

        # Returns
            [dict]: the label
        """
        # Calculate prelimiary information
        temp_box, _ = self.create_basic_block(0)
        box_height, box_width = temp_box.shape
        image_height, image_width = self.page.shape

        margin_top = int(image_height * 0.1)
        margin_left = min(int(image_width * 0.1), 10)

        coeff = 0
        if self.header is not None:
            coeff += 1
        if self.footer is not None:
            coeff += 1
        valid_height = image_height - coeff * margin_top
        valid_width = image_width - 2 * margin_left

        space_vertical = valid_height - (self.n_rows * box_height)
        space_vertical = (
            int(space_vertical / (self.n_rows - 1))
            if self.n_rows > 1
            else 0)
        space_horizontal = valid_width - (self.n_columns * box_width)
        space_horizontal = (
            int(space_horizontal / (self.n_columns - 1))
            if self.n_columns > 1
            else 0)

        # Official works
        os.makedirs(output_folder, exist_ok=True)
        fill_values = len(str(self.n_pages + 1))
        if self.header is not None:
            header_h, header_w = int(margin_top * 0.8), valid_width
            header = self.construct_header(self.header, header_w, header_h)
        if self.footer is not None:
            footer_h, footer_w = int(margin_top * 0.9), valid_width
            footer = self.construct_footer(self.footer, footer_w, footer_h)

        project_labels = {}
        index = 0
        for each_page in tqdm(range(1, self.n_pages + 1)):
            time.sleep(0.5)           # Wait for 1 second, computer heats up!

            # page creation
            image = Image.fromarray(self.page)
            draw = ImageDraw.Draw(image)
            draw.text((0,0), str(each_page), 0, self.font_latin)
            image = np.asarray(image, dtype=np.uint8)
            image.setflags(write=True)

            # header & footer
            if self.header is not None:
                image[
                    int(margin_top * 0.2):header_h+int(margin_top * 0.2),
                    margin_left:margin_left+header_w] = header
            if self.footer is not None:
                image[-footer_h:, margin_left:margin_left+footer_w] = footer

            # construct each page content
            page_labels = {}
            for each_column in range(self.n_columns):
                column_labels = []
                left = margin_left + each_column * (box_width + space_horizontal)
                for each_row in range(self.n_rows):

                    if index >= len(self.text):
                        # might get index out-of-range error if the the number
                        # of lines does not fully fill the last page
                        break

                    top = margin_top + each_row * (box_height + space_vertical)
                    element, label = self.create_basic_block(index)
                    index += 1
                    image[top:top+box_height, left:left+box_width] = element
                    column_labels.append(label)
                page_labels[each_column] = column_labels

            project_labels[each_page] = page_labels
            cv2.imwrite(
                os.path.join(
                    output_folder,
                    '{0:0{space}d}.png'.format(each_page, space=fill_values)),
                image)

        with open(os.path.join(output_folder, 'labels.json'), 'w') as f_out:
            json.dump(project_labels, f_out, indent=4, separators=(',', ': '),
                      ensure_ascii=False, sort_keys=False)

        if to_pdf:
            if verbose:
                print('Creating single pdf file...')
            subprocess.run(['convert', os.path.join(output_folder, '*.png'),
                            os.path.join(output_folder, 'result.pdf')])

        return project_labels


if __name__ == '__main__':
    HEADER = (
        'Thank you for participating!\n' +
        'Please fill in handakuten and dakuten characters in separate box')
    FOOTER = (
        'Thank you for participating!\n' +
        'Please fill in handakuten and dakuten characters in separate box')

    CREATOR = BaseTemplate('/Users/ducprogram/cinnamon/misc/name_kana_1165.txt',
        n_rows=11, n_columns=3, n_pages=50,
        header=HEADER, footer=FOOTER)
    CREATOR.create(output_folder='new_output')
