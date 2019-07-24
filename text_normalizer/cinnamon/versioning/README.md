# Official data versioning

At Cinnamon, the state of AI advancement always continues, and the amount of data keeps increasing. As a result, it is necessary to keep versions of training and testing dataset, so that we know which data a model is trained on, which data a model is tested on. This mini-system of data versioning attempts to:
- assist in adding all images in a central repository
- create necessary information regarding the images
- assist managing train/test data versions

This mini-system requires `rsycn` and an account at Hanoi server to function properly.

## Folder and file structure

The module includes these folders:
- `data`: the central folder containing all images. The images inside can be organized into subfolders.
- `metadata`: this folder contains json files that contain all information of all the images in the `data` folder. The first name of the first column of these json files is the image relative path to the `data` folder.
    + `metadata/.bad_files.txt`: files to skip by default when creating versions
- `scripts`: contains scripts to interact with this data versioning mini-system
- `versions`: contains the version text file, each text file contains the path to image in the `data` folder and the associating label


## Getting started

The file `scripts/connections.py` contains quick functions to set up:

```bash
# fresh setup
$ python scripts/connections.py --mode fresh

# then update from server
$ python scripts/connections.py --mode pull
```


Refer to [*Actions*](#actions) section for more detailed information. Short sample codes are presented here (`cd scripts`)

```python
import subprocess
import add_new_data
import generate_version

# create a temporary tests1 folder
subprocess.run(['cp', '-r', '../tests', '../tests1'])

# change filename to conform with desired format. Look at the initial content of the `input_folder`
# and `input_json` before and after this method is called
add_new_data.adjust_filename(input_folder='../tests1/adjust_filename/test',
                             input_json='../tests1/adjust_filename/labels.json',
                             project='toppan')

# add images to central data.
# @WARNING: this action will paste images and metadata into the working data and metadata folder
add_new_data.add_images(input_folder='../tests1/add_images/test',
                        input_json='../tests1/add_images/labels.json',
                        project='toppan')

# create version, look the result at `versions` folder
generate_version.generate_new_version()
```


## Actions

- add new images in the central image folders and update the json metadata accordingly
- edit images
- create major and minor version text files
- load necessary images and labels from the version text file
- backup images, metadata and version files

### Add new images into central repo

To allow for better examination and maintenance process, the addition of new files into central repo requires that

- a folder containing images and corresponding json file (should not be inside the folder containing images)
- the filename should follow format [ProjectInCamelCase]\_[YYYYMMDD]\_[IdxInteger].[ext]

The json format should be:

```json
{ 
    "relative/path/to/file1" : {
        "label": "",
        "from": "",
        "field": "",
        "char_type": [""]
    },

    "relative/path/to/file2": {}
}
```

Each key is path to file, each value is an object containing information relating to the image. Each object must contain keys `label` and `origin`. Even though data creator can add any information regarding the image, the mini-system supports certain image information. Full description of supported keys below:
- `label` [str]: the label of the image
- `origin` [str]: choose 1 from ['poc', 'collect', 'synthetic']
- `field` [str]: choose 1 from ['address', 'name', 'datetime']
- `char_type` [list of str]: can contain ['kanji', 'kana', 'hira', 'alpha', 'number', 'symbol']


When the above requirement is satisfied, you can add the images and json files to central repo using:

```
$ python scripts/add_new_data.py <path/to/input/folder> <path/to/json/file.json> <ProjectName>
```

The script `add_new_data.py` contains the following helpful functions:
- `adjust_filename`: adjust the filename of images in folder and key in json file to match the desired format
- `add_images`: process input folder and json file and copy them into the central repository

After successfully adding images, json metadata, versions or updating .bad_files.txt, user should update these new files into folder `/data/fixed_form_hw_data` in Hanoi server, so that other members can update these new files. It is also to run backup for this folder as detailed in the later section.

As the number of files increases, it becomes necessary to regularly check the sanity of central repo data. To do so, run:

```
# this will print out problem
python scripts/sanity_check.py --mode repo
```


### Create version text files

A version text file contains links and labels of images corresponding to a specific dataset version, where the links are relative paths in the `data` folder, and labels can be retrieved in the json files in `metadata` folder. Example:

```
/path/to/image1.png,label1
/path/to/image2.png,label2
/path/to/image3.jpg,label3
```

By default, images in the version text file = all images - bad images from `/metadata/.bad_files.txt`. Since we are working on *fixed form* handwriting OCR, this method support automatically creating version for name, address, or all fields. To create train and version text file under this default method, run:

```
$ python scripts/generate_version.py <pick between name/address/all>
```

### Edit/Remove images

Treat editting images the same ways as adding new images. Specifically, instead of deleting the bad images or replacing the bad images with the correct images and/or changing all the metadata and versions accordingly to that change, do:

1. Create new image folder and json metadata (skip for image removal)
2. Add that new folder and json metadata into central repo (as specified above, skip for image removal)
3. Specify the `/path/to/old/image` as bad in `metadata/.bad_files.txt`
4. Generate new version (as specified above)


### Backup

In the data versioning minisystem, only images, json and versions text files need backup. The below script provides easy walkthrough for automatic backup

```
$ python scripts/connections.py --mode push --backup_folder </path/to/backup/folder>
```

where the `/path/to/backup/folder` must contains these 3 folders:

- `data_zipped`: contains smaller zipped files
- `versions`: contains folders of text files
- `metadata`: contains files

By default, this backup folder is `/data/fixed_form_hw_data` in Hanoi server. You can refer to this folder to get better idea of folder structure.


## Developments

This system is developed and tested on Python 3.6. To contribute in development, clone this folder, install the prerequisite and continue developing:

```
pip install -r requirements.txt
```

### Roadmaps

- (for multiple users) Upload new data, metadata and version files (maybe using rsync)
- (nice-to-have) convert from json -> csv for better viewing
    + csv is less convenient than json because we do not have a fixed number of features and attributes, while csv requires to have fixed number of columns; json does not have the same problem
- ability to trace original files
- handle duplicate data
- support faster picking bad files
- single character data
