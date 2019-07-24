import io
import os
import re
import numpy as np
import cairocffi as cairo
import pickle
import random

NORMALIZE_TEXT = {
    '１': '1',
    '２': '2',
    '３': '3',
    '４': '4',
    '５': '5',
    '６': '6',
    '７': '7',
    '８': '8',
    '９': '9',
    '０': '0',
    'ａ': 'a',
    'ｂ': 'b',
    'ｃ': 'c',
    'ｄ': 'c',
    'ｅ': 'e',
    'ｆ': 'f',
    'ｇ': 'g',
    'ｈ': 'h',
    'ｉ': 'i',
    'ｊ': 'j',
    'ｋ': 'k',
    'ｌ': 'l',
    'ｍ': 'm',
    'ｎ': 'n',
    'ｏ': 'o',
    'ο': 'O',
    'ｐ': 'p',
    'ｑ': 'q',
    'ｔ': 't',
    'ｓ': 's',
    'ｒ': 'r',
    'ｕ': 'u',
    'ｖ': 'v',
    'ｗ': 'w',
    'ｘ': 'x',
    'ｙ': 'y',
    'ｚ': 'z',
    'А' : 'A',
    'Ａ': 'A',
    'Ｂ': 'B',
    'Ｃ': 'C',
    'Ｄ': 'D',
    'Ｅ': 'E',
    'Ｇ': 'G',
    'Ｈ': 'H',
    'Н': 'H',
    'Ｉ': 'I',
    'Ｊ': 'J',
    'Ｋ': 'K',
    'Ｌ': 'L',
    'Ｍ': 'M',
    'Ｎ': 'N',
    'Ν': 'N',
    'Ｏ': 'O',
    'Ｐ': 'P',
    'Р': 'P',
    'Ｑ': 'Q',
    'Ｔ': 'T',
    'Ｓ': 'S',
    'Ｒ': 'R',
    'Ｕ': 'U',
    'Ｖ': 'V',
    'Ｗ': 'W',
    'Ｘ': 'X',
    'Х': 'X',
    'Ｙ': 'Y',
    'Ｚ': 'Z',
    '！': '!',
    '＠': '@',
    '＄': '$',
    '％': '%',
    '＾': '^',
    '＆': '&',
    '＊': '*',
    '（': '(',
    '）': ')',
    '＿': '_',
    '＋': '+',
    '－': '-',
    '＝': '=',
    '［': '[',
    '］': ']',
    '｜': '|',
    '│': '|',
    'ǀ': '|',
    "’" : "'",
    '¥' : '￥',
    '；': ';',
    '：': ':',
    '／': '/',
    '．': '.',
    '，': ',',
    '？': '?',
    '＞': '>',
    '＜': '<',
    '‐': '-',
    '―': '-',
    'Ｆ': 'F',
    '“': '"',
    '×': 'X',
    '　': ' '
}

class WordList(object):
    def __init__(self, training_text,monogram_text,unichar_text,poc_text,absolute_max_string_len=16,normalize_text=NORMALIZE_TEXT):
        self.training_text = training_text
        self.poc_text = poc_text
        self.monogram_text = monogram_text
        self.unichar_text = unichar_text
        self.absolute_max_string_len = absolute_max_string_len
        self.vocab = []
        self.unichar_set = [" "]
        self.vocab_poc = []
        self.normalize_text = normalize_text
        # self.vocab_path = os.path.join('.', 'vocab.pickle')
        # self.unichar_set_path = os.path.join('.', 'unichar.pickle')
        # self._build_word_array()
        # self._build_uni_charset()
        # self._load_uni_charset()
        # self._load_poc_vocab()
        self._text_to_label()
        self._label_to_text()
        self.n_class = len(self.unichar_set)

    def _build_word_array(self):
        """
        split training text into work chunks and store into vocab
        :return: array of words
        """
        if not os.path.exists(self.vocab_path):
            print("vocab not found ... building vocab")
            with io.open(self.training_text, mode='rt', encoding='utf-8') as f:
                for line in f:
                    word = line.rstrip().split(" ")[0]
                    if word not in self.vocab:
                        self.vocab.append(word)
                with open("vocab.pickle", "wb") as output_file:
                    pickle.dump(self.vocab, output_file)
        else:
            print("loading vocab from {}".format(self.vocab_path))
            with open(self.vocab_path, "rb") as input_file:
                self.vocab = pickle.load(input_file)

    # Swap the last slot in unichar_set for " " character
    def swap_space(self):
        self.unichar_set[0] = self.unichar_set[-1]
        self.unichar_set[-1] = " "

        # Update corresponding lists and n_classes
        self._text_to_label()
        self._label_to_text()
        self.n_class = len(self.unichar_set)

    # Update unichar_set with character labels set
    def update_uni_charset(self, labels):
        for line in labels:
            for char in list(self._normalize_text_line(line)):
                if char not in self.unichar_set and self._is_valid_unichar(char):
                    self.unichar_set.append(char)

    # Create unichar_set from standrad, provided set
    def _build_uni_charset(self):
        with io.open(self.monogram_text, mode='rt', encoding='utf-8') as f:
            for line in f:
                char = line.rstrip().split(" ")[0]

                if char not in self.unichar_set:
                    self.unichar_set.append(char)
        

    def get_line(self, max_string_len=None,insert_space=False):
        self.vocab = np.random.permutation(self.vocab)
        string_list = [" "]*max_string_len
        for i in range(len(string_list)):
            string_list[i] = self.vocab[i][0]
        if insert_space:
            num_space = np.random.randint(0,max_string_len)
            space_indice = np.random.choice(len(string_list),num_space,replace=False)
            for index in space_indice:
                if index != 0 and index != (len(string_list) - 1):
                    string_list[index] = " "
        render_text = "".join(string_list)
        train_text = self._normalize_text_line(' '.join((render_text.lstrip().rstrip().split())))
        # train_text = train_text.ljust(self.absolute_max_string_len)
        label_list = [float(self.text_label[char]) for char in train_text]
        return (render_text,train_text,label_list)

    def _load_poc_vocab(self):
        with io.open(self.poc_text, mode='rt', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                if line not in self.vocab_poc:
                    self.vocab_poc.append(line)

    def get_poc_line(self,max_string_len=None,insert_space=False):
        # self.vocab_poc = np.random.permutation(self.vocab_poc)
        line_array = [" "]*max_string_len
        line = " "*100
        while len(line) > max_string_len:
            line = random.choice(self.vocab_poc)
        for i,w in enumerate(line):
            line_array[i] = w
        # check if space is many then shuffle it
        first = line_array[0]
        rest = line_array[1:]
        if line_array.count(" ") > (max_string_len - 3):
            random.shuffle(rest)

        render_text = "".join([first] + rest)
        train_text = self._normalize_text_line(' '.join(render_text.lstrip().rstrip().split()))
        label_list = [float(self.text_label[char]) for char in train_text]
        return (render_text, train_text, label_list)
    def _is_valid_str(self,in_str):
        search = re.compile(r'^[a-z ]+$', re.UNICODE).search
        return bool(search(in_str))

    def _is_valid_unichar(self,unichar, font='meiryo'):
        # check if char can be renderd by font
        surface = cairo.ImageSurface(cairo.FORMAT_RGB24, 48, 48)
        with cairo.Context(surface) as context:
            context.select_font_face(font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            context.set_font_size(38)
            box = context.text_extents(unichar)
            if (box[2] == 0.0 or box[3] == 0.0):
                #raise IOError("ERROR Cannot render char {} with font {}".format(unichar,font))
                return False
            else:
                return True
    def _normalize_text_line(self,text_line):
        for w in text_line:
            if w in self.normalize_text:
                text_line = text_line.replace(w,self.normalize_text[w])
        return text_line

    def _text_to_label(self):
        print(len(self.unichar_set))
        self.text_label = {v: k for k, v in enumerate(self.unichar_set)}

    def _label_to_text(self):
        self.label_text = {v: k for k, v in self.text_label.items()}

    def get_text_to_label(self):
        return self.text_label

    def get_label_to_text(self):
        return self.label_text

    def get_label(self,text):
        return [float(self.text_label[char]) for char in self._normalize_text_line(text)]

    def get_nclass(self):
        return self.n_class
