import os
import numpy as np
from PIL import Image, ImageOps

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import utils.normalize_japanses_char as standard_normalizer
import utils.normalize_japanses_char_printed as printed_normalizer
import utils.normalize_invoice as invoice_normalizer
from utils.common import InputType


def read_default(root, label_file, normalize_label=None, alphabet=None, separated_char='|'):
    """
    Read label file

    :param root: folder path contains images
    :param label_file:  label file path contains label
    :param normalize_label: normalize method for label
    :param alphabet: charset
    :param separated_char: separator for image path and label in label_file
    :return:
    image_list: [(image_path, image_label), ...]
    """
    image_list = []

    with open(label_file, encoding='utf-8') as rf:
        for line in rf.readlines():
            arr = line.strip().split(separated_char, 1)
            if len(arr) != 2:
                continue

            image_path, image_label = arr

            # normalize japanese characters
            if normalize_label == 'SHO':
                image_label = standard_normalizer.normalize_text(image_label)
                image_label = standard_normalizer.combine_diacritic_characters(image_label)
                image_label = standard_normalizer.normalize_text(image_label)
                image_label = image_label.replace('.', '')
            elif normalize_label == 'PRINTED':
                image_label = printed_normalizer.normalize_text(image_label)
                image_label = printed_normalizer.combine_diacritic_characters(image_label)
                image_label = printed_normalizer.normalize_text(image_label)
            elif normalize_label == 'INVOICE':
                image_label = invoice_normalizer.normalize_text(image_label)
                image_label = invoice_normalizer.normalize_text(image_label)
            else:
                print("[WARN] normalize_label is NULL")

            if not os.path.exists(os.path.join(root, image_path)):
                print("[Skip] Non exists image path: {} {}".format(os.path.join(root, image_path), image_label))
                continue

            if os.path.getsize(os.path.join(root, image_path)) <= 0:
                print("[Skip] Zero byte image path: {} {}".format(image_path, image_label))
                continue

            if os.path.getsize(os.path.join(root, image_path)) <= 100:
                print("[Skip] Too-small (<=100 bytes) image path: {} {}".format(image_path, image_label))
                continue

            if image_label == '':
                print("[Skip] Blank label image path: {} {}".format(image_path, image_label))
                continue

            if alphabet is not None:
                skip = False
                char = ""
                for x in image_label:
                    if x not in alphabet:
                        skip = True
                        char = x
                        break
                if skip:
                    print("[Skip] Not in Charset: {} - Char \"{}\" - {}".format(image_path, char, image_label))
                    continue

            image_list.append((image_path, image_label))

        print("====> Total available {} images from {}".format(len(image_list), label_file))
    return image_list


def load_default(path, input_type):
    """
    Load image
    :param path: path of image
    :param input_type: input type
    :return:
    image as PIL
    """
    if input_type == InputType.binary.value:
        im = Image.open(path)
        im = im.convert('L')
        im = ImageOps.invert(im)
        im = im.convert('1')
    elif input_type == InputType.rgb.value:
        im = Image.open(path)
        im = im.convert('RGB')
    elif input_type == InputType.gray_scale.value:
        im = Image.open(path)
        im = im.convert('L')
    else:
        raise ValueError("Only support input type: Gray scale, Binary, RGB")

    return im


class DatasetLoader(Dataset):
    def __init__(self, root, label_file, normalize_label, input_type=1, alphabet=None, separated_char="|",
                 read_func=read_default, load_func=load_default):
        self.root = root
        self.load_func = load_func
        self.input_type = input_type
        self.image_list = read_func(root, label_file, normalize_label, alphabet, separated_char)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path, image_label = self.image_list[idx]
        image_full_path = os.path.join(self.root, image_path)
        image = self.load_func(image_full_path, self.input_type)
        return image, image_label


class ResizeTransformer(object):
    def __init__(self, size, input_type=1, interpolation=Image.LANCZOS):
        self.size = size
        self.interpolation = interpolation
        self.ToTensor = transforms.ToTensor()
        self.input_type = input_type

    def __call__(self, image):
        if self.input_type == InputType.binary.value:
            background_image = Image.new('1', self.size, color=0)
        elif self.input_type == InputType.rgb.value:
            background_image = Image.new('RGB', self.size, color=(255, 255, 255))
        elif self.input_type == InputType.gray_scale.value:
            background_image = Image.new('L', self.size, color=255)
        else:
            raise ValueError("Only support input type: Gray scale, Binary, RGB")
        w, h = image.size
        w = min(int(np.floor(w / h * self.size[1])), self.size[0])
        image = image.resize((w, self.size[1]), self.interpolation)
        background_image.paste(image, (0, 0))
        new_image = self.ToTensor(background_image)
        return new_image


class AlignCollate(object):
    def __init__(self, img_h=32, img_w=100, keep_ratio=True, input_type=1, min_width=48, background_value=1.0):
        self.img_h = img_h
        self.img_w = img_w
        self.keep_ratio = keep_ratio
        self.input_type = input_type
        self.min_width = min_width
        self.background_value = background_value

    def __call__(self, batch):
        images, labels = zip(*batch)
        img_w = self.img_w
        img_h = self.img_h
        # if true: resize width along to height by max ratio of w/h
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            img_w = int(np.floor(max_ratio * img_h))
        resize_transformer = ResizeTransformer((img_w, img_h), input_type=self.input_type)
        resized_images = [resize_transformer(image) for image in images]
        images = torch.cat([img.unsqueeze(0) for img in resized_images], 0)

        b, c, h, w = images.size()
        if w < self.min_width:
            padded_images = torch.ones(b, c, h, self.min_width) * self.background_value
            padded_images[:, :, :, :w] = images
            return padded_images, labels
        return images, labels


def load_training_data(configs):
    training_ds = DatasetLoader(configs["train_data_folder"], configs["train_data_file"], configs["normalize_label"],
                                input_type=configs["input_type"], alphabet=configs["character"],
                                separated_char=configs["separator"])
    align_collator = AlignCollate(configs["img_h"], configs["img_w"], keep_ratio=True, input_type=configs["input_type"])
    training_loader = DataLoader(training_ds, shuffle=True, pin_memory=True, batch_size=configs["batch_size"],
                                 num_workers=configs["workers"], collate_fn=align_collator)
    return training_loader


def load_validation_data(configs):
    validation_ds = DatasetLoader(configs["valid_data_folder"], configs["valid_data_file"], configs["normalize_label"],
                                  input_type=configs["input_type"], alphabet=configs["character"],
                                  separated_char=configs["separator"])
    align_collator = AlignCollate(configs["img_h"], configs["img_w"], keep_ratio=True, input_type=configs["input_type"])
    validation_loader = DataLoader(validation_ds, shuffle=True, pin_memory=True, batch_size=configs["batch_size"],
                                   num_workers=configs["workers"], collate_fn=align_collator)
    return validation_loader


def load_alphabet(character):
    if not os.path.exists(character):
        alphabet = character
    else:
        with open(character, 'r', encoding='utf-8') as rf:
            alphabet = rf.read().replace('\r', '').replace('\n', '')
    return alphabet


class InferenceDataset(Dataset):
    def __init__(self, images):
        self.image_list = images

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.image_list[idx]
        return image, ""


def load_batch_data(images, configs):
    inference_ds = InferenceDataset(images)
    align_collator = AlignCollate(configs["img_h"], configs["img_w"], input_type=configs["input_type"])
    inference_loader = DataLoader(inference_ds, shuffle=False, pin_memory=True, batch_size=configs["batch_size"],
                                  num_workers=configs["workers"], collate_fn=align_collator)
    return inference_loader
