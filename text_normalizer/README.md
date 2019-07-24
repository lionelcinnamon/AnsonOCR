# Information
This generater generate japanese text line, of the following version:
- Handwritten line (both binary and grayscale)
- Printed line (both binary and grayscale)

Generated images using handwritten characters:

![handwrriten1](docs/handwritten1.png)
![handwritten2](docs/handwritten2.png)
![handwritten3](docs/handwritten3.png)

Generated images using fonts (augmented to make it less clean):

![printed1](docs/printed1.png)
![printed2](docs/printed2.png)
![printed3](docs/printed3.png)

# Get started

Due to a plethora of Open-CV Python libraries, which might result in namespace conflict, you should manually install Open-CV. After that, install this package depedencies:

```bash
# If python opencv is not installed, you can choose 1 of these 2 options
$ conda install -c conda-forge opencv
$ pip install opencv-python

# dataloader other dependencies
$ pip install -r requirements.txt
```

# Prerequisites
This generator assumes several files (e.g. font files, handwriting character images,...) in order to work:

- Charset: is a text file that contains characters that will be used for synthetic line generation. The rationale for this text file is many Kanji characters are uncommon, hence, include those characters in text line will dramatically increase the number of character classes, while adding very small benifits. This text file will tells generator to only generate line images that characters are included within this charset. The charset text file can be constructed based on tasks (using `dataloader.misc.dump_allowed_char_file`, but some common versions are stored [here](https://drive.google.com/open?id=1OO54wMy9e18reXur9qJpARopbTP__qHT).
- Text files: these text file determine the content of generated line image. If the text file contains character not included in the charset above, then those characters will be omitted during generation. Some text files can be obtained [here](https://drive.google.com/open?id=1b9-YiJB27wVshEww4znGGKGjPtpLBLNq)
- (For handwritten line generation) Character files: these files contain the images of each character - [binary](https://drive.google.com/open?id=1MmIn0Zoj3ucv-6LnVQcxVu-jQpyZDayo), [grayscale](https://drive.google.com/open?id=1quJ1pZyR8xxhxDZERCTysUHI63JdXXbY). A lighter version for development purpose is stored [here](https://drive.google.com/open?id=1GdyLtZXQ9fz91xerT1Np7rcFaJkEjGcO) (use `gray_images.pkl` for grayscale generation and `imagse.pkl` for binary generation).
- (For printed line generation) Font folder: these are the fonts that can be used to generate text line from fonts [here](https://drive.google.com/open?id=17wQLvs6w1xmEhmS255ylc8NNK-vZavQG).


# Usage


### Generate images from text

Generating line images requires:

1. Text file, where each line represents a line to generate into image.
2. (Only for generation using font) Folder containing .ttf and .otf fonts (can be retrieved according to the above section).
3. (Only for generation using images of characters) File containing image characters (can be retrieved according to the above section).

If the above conditions are satisfied, you can generate images by going to the `generate` folder and run:

```bash
# using handwritten scheme
$ python dataloader/generate/image.py handwritten --output-folder [path] --corpus [path.txt] --chars [path.pkl or /path/*.pkl]

# using printed scheme
$ python dataloader/generate/image.py printed --output-folder [path] --corpus [path.txt] --fonts [path]
```


### On-the-spot generation

Images can be generated into numpy array, eliminating the needs for saving and loading `.png` from disk. This feature is useful to quickly generate on-the-fly data for training.

```python
import glob
import os
import pickle
from dataloader.generate.image import HandwrittenLineGenerator, PrintedLineGenerator

TEST_PKL_FILE='dev/images.pkl'
TEST_TXT_FILE='dev/address.txt'
ALLOWED_CHARS='dev/charset_codes.txt'
CHEM_CHARS = 'dev/chemical_formulas.txt'
BACKGROUND_NOISES = 'dev/background'
FONT_FOLDER = 'dev/fonts'
OUTPUT_DIR='output'

# 1. Initialize the generator
lineOCR = HandwrittenLineGenerator(allowed_chars=ALLOWED_CHARS)
printOCR = PrintedLineGenerator(allowed_chars=ALLOWED_CHARS, is_binary=False)


# 2. Load the pkl and text files
lineOCR.load_character_database(TEST_PKL_FILE)
lineOCR.load_text_database(TEST_TXT_FILE)
lineOCR.load_chemical_formulas(CHEM_CHARS)
lineOCR.load_background_image_files(BACKGROUND_NOISES)
lineOCR.initialize()

printOCR.load_fonts(FONT_FOLDER)
printOCR.load_text_database(TEST_TXT_FILE)
printOCR.load_background_image_files(BACKGROUND_NOISES)
printOCR.initialize()


# 3a. Generate images and save to external folder
lineOCR.generate_images(start=0, end=-6000, save_dir=OUTPUT_DIR)

# 3b. Generate into batches
BATCH_SIZE = 4
for _idx in range(10):
    X, widths, y = lineOCR.get_batch(BATCH_SIZE)
    print('{} ==== {}, {}, {}'.format(_idx, len(X), widths, len(y)))
    for each_y in y:
        print(each_y)

# 3c. Get images from folder
X, width, y = lineOCR.get_batch_from_files(batch_size=32, path=OUTPUT_DIR,
        append_channel=True, label_converted_to_list=True,
        mode=constants.TRAIN_MODE, skip_invalid_image=False)
```

### Modifications to the generated images/labels

Due to differences in training procedures, generated images and labels might need modifications (in piece or in batch) before used in training. This can be done without modifications of the generator by subclassing the `generate.image.TrainingHelper` class and define these methods inside the helper:
- `postprocess_image`: allows processing each image individually
- `postprocess_label`: allows processing each label individually
- `postprocess_outputs`: which takes a list of images, a list of labels and a list of widths, allows processing everything in batch

After that, initialize the helper and pass it into generator during creation. Example:

```
helper = TrainingHelper()
generator = HandwrittenLineGenerator(helper=helper,..)
```

# Development path
- Make the generation multiprocess
- Utility: automatically combine separate diacritic characters into base characters
- A scheme to use both handwritten and printed generator

# Coding convention
- PEP8, summarized in this [file](https://drive.google.com/file/d/1kizy-UTHJ9qbrBmbPFE_xCqYHW_7p8qU/view)
- Docstring: should be included with all functions and methods; with the following format:
```python
def function_name(arg1, arg2):
    """One-line description

    Multi-line elaboration (optional).

    # Arguments
        arg1 [type of arg1]: description of arg1
        arg2 [type of arg2]: description of arg2

    # Returns
        [type of first return]: description of first return
        [type of second return]: description of second return

    # Raises
        SomeError: description of the error
    """
    return 1, 'a'
```