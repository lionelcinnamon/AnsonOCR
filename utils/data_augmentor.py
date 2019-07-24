# Configurate paths
import os, sys
sys.path.insert(
    0,
    os.path.dirname(sys.path[0])
)

from utils.config import get_config_from_json
from utils.arg_utils import get_args
from data_loader.DataGenerator.MachineTextUtilities import MachineTextUtilities as Util

from skimage.util import random_noise
from skimage.transform import pyramid_reduce, resize
from keras.preprocessing import image
from PIL import Image

import cairocffi as cairo
import cv2
import random
import numpy as np
import csv, io, glob

def get_config():
    try:
        args = get_args()
        config, _ = get_config_from_json(args.config)

        return config
    except:
        print("Missing configuration file")
        exit(0)

class Augmentor:
    def __init__(self, config):
        self.names = []
        self.labels = []
        self.config = config
        self.poc_imgs, self.poc_labels = self._acquire_real_data()
        self.training_text = self.get_training_text()

        # Create font pool
        font_path = os.path.join(
            self.config.font_dir
        )
        assert os.path.exists(font_path)
		
        font_types = ['*.ttf', '*.ttc', '*.otf']
        self.fonts = []

        for types in font_types:
            self.fonts.extend(
                glob.glob(
                    os.path.join(
                        font_path,
                        types)
                    )
                )

        assert len(self.fonts) > 0
        print("Found {} font ".format(len(self.fonts)))

        # Create directory to dumb images
        self.path = os.path.join(
            'datasets',
            config.name
        )

        if not os.path.isdir(self.path):
            os.mkdir(self.path)

            os.mkdir(
                os.path.join(
                    self.path,
                    "raw"
                )
            )

    # Augment image
    def augment(self, img):
        if len(self.labels) % 1000 == 0:
            print(len(self.labels))
        w, h = img.size
        assert w*h > 0
        img = np.array(img, dtype=float) / 255

        # Shear
        sheared = random.random() < self.config.blur_prob
        	
        if sheared:
            rows,cols,_ = img.shape
            offset = random.choice([-10, 10])
            pts1 = np.float32([
                [10,10],
                [1900,10],
                [10,30]
            ])
            pts2 = np.float32([
                [10 + offset,10],
                [1890 + offset,10],
                [10,30]
            ])
            
            M = cv2.getAffineTransform(pts1,pts2)
            
            img = cv2.warpAffine(
                img,
                M,
                (cols,rows),
                borderValue = [255,255,255]
            )

        # Rotate
        # TODO: double check params
        # rotated = random.random() < self.config.rotate_prob
        # if rotated:
        #     order = np.random.uniform(-0.05, 0.05)
        #     img = image.random_rotation(
        #         img, 
        #         order * np.arctan(h/w) * 57.3
        #     )

        # Degrade
        degraded = random.random() < self.config.degrade_prob
        if degraded :
            degrade_level = np.random.uniform(1.01, 1.1)

            img = pyramid_reduce(
                img, 
                downscale=degrade_level,
                mode='reflect'
            )

            img = resize(img, (h,w),mode='reflect')

        # Noise
        noised = random.random() < self.config.noise_prob
        if noised:
            noise_type  = np.random.choice([
                'gaussian',
                's&p',
                'speckle',
                'poisson',
                'localvar'
            ])

            img = random_noise(img, mode=noise_type)
        
        # Invert
        inverted = random.random() < self.config.invert_prob
        if inverted:
            img = 1 - img

        # Blur
        # TODO: double check params
        blurred = random.random() < self.config.blur_prob
        if blurred:
            img = cv2.blur(img, (3, 3))

        # Brightness
        img = np.clip(np.random.uniform(0.8,1.5) * img, 0, 1)
        
        # Translation
        img = 255 * img

        img = np.array(img, dtype=np.uint8)
        img = Image.fromarray(img, mode='RGB')
        
        tx = random.randint(-4, 4)
        ty = random.randint(-4, 4)

        img = img.transform(
            img.size, 
            Image.AFFINE, 
            [1, 0, tx, 0, 1, ty]
        )

        return img.convert('L')

    # Merge csv of all datasets in config
    def _acquire_real_data(self):
        names = []
        labels = []

        for item in self.config.datasets:
            csv_path = os.path.join(
                'datasets',
                item,
                'label.csv'
            )

            raw_path = os.path.join(
                'datasets',
                item,
                'raw'
            )

            assert os.path.exists(csv_path)
            assert os.path.exists(raw_path)

            with open(csv_path, 'r') as csvFile:
                reader = csv.reader(csvFile)
                next(reader)

                for row in reader:
                    img_path = os.path.join(
                        raw_path, 
                        row[0]
                    )

                    assert os.path.exists(img_path)

                    names.append(img_path)
                    labels.append(row[1])

            csvFile.close()

        return names, labels
    
    def clone_real_data(self, n_real):
        indexes = np.arange(len(self.poc_imgs))
        np.random.shuffle(indexes)
        
        # If number of samples is sufficient
        if n_real <= len(self.poc_labels):
            indexes = indexes[0:n_real]
        else:
            print(n_real)
            # Append random index
            while len(indexes) < n_real:
                indexes = np.append(
                    indexes,
                    random.randint(0, len(self.poc_imgs) - 1)
                )
                
        for i in indexes:
            img_name = self.poc_imgs[i].split('/')[-1]

            # Augment image
            img = Image.open(self.poc_imgs[i]).convert('RGB')
            img = self.augment(img)

            # Dumb image
            save_path = os.path.join(
                self.path,
                "raw",
                img_name
            )
            
            while (os.path.exists(save_path)):
                img_name = "d" + img_name
                save_path = os.path.join(
                    self.path,
                    "raw",
                    img_name
                )
            

            # Append to csv list
            self.names.append(img_name)
            self.labels.append(self.poc_labels[i])

            # Make sure there is no duplication
            assert not os.path.exists(save_path)

            img.save(save_path)

    # Merge trainging text files
    # TODO: merge multiple text files
    def get_training_text(self):
        train_path = os.path.join(
            'datasets',
            self.config.training_text
        )

        with io.open(train_path, 'r', encoding='utf8') as f:
            training_text = [
                word for line in f for word in line.split()
            ]
            
            print("Initialise with {} word(s) in training_text".format(
                len(training_text)))

        return training_text

    def get_text_line(self, poc, train, max_length):
        text = ""

        # Poc_text or training_text
        if random.random() < self.config.real_text_ratio:
            text = random.choice(self.poc_labels)
        else:
            # Random number of words
            length = random.randint(1, max_length)

            for i in range(length):
                text = text + " " + random.choice(self.training_text)

            # Remove first " "
            text = text[1:]

            # Clip if exceeds max_length
            if len(text) > max_length:
                text = text[:max_length]

        return text
    
    def render_line(self, text):
        time_out = 0

        while (True):
            img = Util.render(
                text, 
                random.choice(self.fonts), 
                self.config.height, 
                normalize=True, 
                normalize_size=(self.config.height * len(text), self.config.height)
            )

            if img != False:
                break
            else:
                time_out += 1
                if (time_out > 10):
                    print('Timeout cannot fix font and text render')
                    return False
                continue

        return img

    def generate_fake_data(self, n_fake):
        for i in range(n_fake):
            while (True):
                text = self.get_text_line(
                    self.poc_labels, 
                    self.training_text,
                    self.config.max_length
                )

                img = self.render_line(text)
                
                if img != False:
                    img = self.augment(img)

                    img_name = "fake_" + str(i) + ".png"
                    self.names.append(img_name)
                    self.labels.append(text)

                    # Dumb image
                    save_path = os.path.join(
                        self.path,
                        "raw",
                        img_name
                    )

                    # Make sure there is no duplication
                    assert not os.path.exists(save_path)
                    img.save(save_path)

                    break
                else:
                    continue

    # Generate dataset
    def generate(self):
        # Real data
        n_real = int(
            self.config.n_images * self.config.real_ratio
        )
        self.clone_real_data(n_real)

        # Synthesized data
        n_fake = self.config.n_images - n_real
        self.generate_fake_data(n_fake)
        
        return self.names, self.labels

def main():
    # Get configuration
    config = get_config()

    # Generate datset
    augmentor = Augmentor(config)
    names, labels = augmentor.generate()

    # Export information to csv
    path = os.path.join(
        'datasets',
        config.name,
        'label.csv'
    )

    with open(path, 'w') as csvfile:
        writer = csv.writer(
            csvfile, 
            dialect='excel', 
            delimiter=','
        )
        
        writer.writerow(
            ['Filename', 'Label']
        )
        
        for name, label in zip(names,labels):
            writer.writerow([name, label])
    
    csvfile.close()

if __name__ == '__main__':
    main()
