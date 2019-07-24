#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import os
import sys
import io
import glob
import pandas as pd
import numpy as np
import copy
import json
import tqdm
import re

XML_EXT = '.xml'
CSV_EXT = '.csv'
ENCODE_METHOD = 'utf-8'


class InvoiceIO(object):
    """
    class to read write invoice format
    support : VIA , Althelia, inhouse format
    """

    def __init__(self):
        pass


class InvoiceWriter(object):
    def __init__(self, data):
        self.elements_dict = data

    def update_json(self, json_file, new_json_file):
        """
        update json label file
        """
        with open(new_json_file) as f:
            new_data = json.load(f)
        if os.path.exists(json_file):
            with open(json_file) as f:
                data = json.load(f)
                for element in new_data.keys():
                    if element not in data.keys():
                        data[element] = new_data[element]
                    else:
                        print("There are conflicts!! cannot update")
        else:
            print("new file")
            data = new_data
        self._write_json(json_file, data)

    def merge_label(self, file_to_update):
        reader = InvoiceReader(file_to_update)
        old_elements_dict = reader.get_elements_dict()
        old_elements_dict.update(self.elements_dict)
        self.elements_dict = old_elements_dict
        self.write_label(file_to_update)


#    @classmethod

    def write_label(self, filepath):
        shapes = []
        for element in self.elements_dict:
            for item in self.elements_dict[element]:
                shape = {
                    "label": item["label"],
                    "points": item["points"],
                    "line_color": None,
                    "fill_color": None
                }
                shapes.append(shape)
        # get file image name from file path
        filename_img = str.split(os.path.basename(filepath), '.')[0]+".png"

        data = {"imageData": None,
                "imagePath": filename_img,
                "lineColor": [0, 255, 0, 128],
                "fillColor": [255, 0, 0, 128],
                "shapes": shapes,  # polygonal annotations
                "flags": {}   # image level flags
                }
        with open(filepath, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _write_json(self, outfile, data):
        with open(outfile, 'w') as outfile:
            json.dump(data, outfile, sort_keys=True,
                      indent=4, separators=(',', ': '))


class InvoiceReader(object):

    def __init__(self, path, types="label_invoice"):
        if not os.path.exists(path):
            raise IOError("InvoiceReader: ", "{} not exists".format(path))
#        self.elements_list = ["textline", "stamps"]
        self.elements_dict = {}
        self.data = self._parse_json(path)
        self.elements_dict = self._parse_label_invoice(self.data)

    def get_elements_dict(self):
        return self.elements_dict

    def get_element(self, element):
        if element in self.elements_dict.keys():
            return self.elements_dict[element]
        else:
            return []

    def _parse_json(self, json_file):
        with open(json_file) as f:
            data = json.load(f)
            # clean unwanted elements
            data.pop("imageData", None)  # heavy image data
        return data

    def _parse_label_invoice(self, data_in):
        """
        to parse element list in label_invoice
        """
        elements_dict = {}

        if "shapes" in data_in.keys():
            # parse item label to element
            for item in data_in['shapes']:

                label = item['label']

                element = label
                # check for ocr class
                p = re.match("^TextLine: *", label)
                if p is not None:
                    ocr = label[p.span()[1]:]
                    element = 'textline'
                    record = {'ocr': ocr,
                              'points': item['points'],
                              'label': label,
                              'textclass': item['textclass'] if 'textclass' in item.keys() else 'None'}

                p = re.match("^cell", label)
                if p is not None:
                    parent = label[p.span()[1]:]
                    element = 'cell'
                    record = {'parent': parent,
                              'points': item['points'],
                              'label': label}

                # update dict with element founds
                if element not in elements_dict.keys():
                    elements_dict.update({element: []})
                if element not in ['textline', "cell"]:
                    record = {'points': item['points'], 'label': label}

                elements_dict[element].append(record)

        return elements_dict


def parse_via():
    label_path = "./lines_gt.json"
    # reader = InvoiceReader(label_path,"via")
    with open(label_path) as f:
        data = json.load(f)
        # clean unwanted elements
    for file in data['lines']:
        lines = data['lines'][file]
        elements_dict = {'textline': []}
        label = 'TextLine: '
        ocr = 'None'
        for line in lines:
            points = [[line['x'], line['y']],
                      [line['x']+line['width'], line['y']],
                      [line['x']+line['width'], line['y'] + line['height']],
                      [line['x'], line['y'] + line['height']]
                      ]
            record = {'ocr': ocr,
                      'points': points, 'label': label}
            elements_dict['textline'].append(record)
        print(file, len(elements_dict['textline']))
        writer = InvoiceWriter(elements_dict)
        json_file = str.split(file, '.')[0]+'.json'
        writer.merge_label(json_file)
#        writer.write_label(file)

    """
    for file in glob.glob("./*.json"):
        file_name = os.path.basename(file)
        path = os.path.join("../datasets/000_mizuho", file_name)
        print(path)
        writer = InvoiceWriter("./")
        writer.update_json(path, file)
    """


def main():
    img_dir = "~/Github/00_cinnamon/rnd_invoice_analysis/images/"
    label_dir = "~/Github/00_cinnamon/rnd_invoice_analysis/label"
    iio = InvoiceIO("./")
    json_list = glob.glob("../test/*.json")
    dataset = "000_mizuho"
    dataset_dir = "../datasets"
    for json_file in json_list:
        reader = InvoiceReader(json_file)
        file_name = os.path.basename(json_file)
        print(file_name[2:])
        output = os.path.join(dataset_dir, dataset, file_name)
        reader._write_json(output)


def test_write_label():
    json_file = "./calbee_10_1.json"
    reader = InvoiceReader(json_file)
    writer = InvoiceWriter(reader.get_elements_dict())
    writer.write_label("dummy.json")


if __name__ == "__main__":
    # main()
    parse_via()
    # test_write_label()
