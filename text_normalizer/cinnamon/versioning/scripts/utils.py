"""Script contains small utility operations"""
import glob
import os
import json
import subprocess


FILE_USER_CONFIG = 'user_config.json'
FILE_SYSTEM_CONFIG = 'system_config.json'

def get_base_path(folder=None):
    """Get base folder"""
    base_path = os.path.realpath(__file__)
    base_path = os.path.dirname(base_path)
    base_path = os.path.dirname(base_path)

    if folder is None:
        return base_path

    return os.path.join(base_path, folder)


with open(get_base_path(FILE_SYSTEM_CONFIG), 'r') as f_in:
    _config = json.load(f_in)


SERVER_IP = _config['SERVER_IP']
SERVER_PORT = (
    ''
    if _config.get('SERVER_PORT', '') == ''
    else ' -p {}'.format(_config['SERVER_PORT']))
SERVER_FOLDER = _config['SERVER_FOLDER']
BOX_FOLDER_BASE = _config['BOX_FOLDER_BASE']

FOLDER_DATA = 'data'
FOLDER_WORK = '.work'
FOLDER_DATA_ZIPPED = '.work/data_zipped'
FOLDER_METADATA = 'metadata'
FOLDER_SCRIPT = 'scripts'
FOLDER_VERSIONS = 'versions'
FOLDER_TEST = 'tests'

FILE_BAD = '.bad_files.txt'

SERVER_FOLDER_DATA = os.path.join(SERVER_FOLDER, 'data')
SERVER_FOLDER_DATA_ZIPPED = os.path.join(SERVER_FOLDER, 'data_zipped')
SERVER_FOLDER_METADATA = os.path.join(SERVER_FOLDER, 'metadata')
SERVER_FOLDER_VERSIONS = os.path.join(SERVER_FOLDER, 'versions')
SERVER_FILE_TESTS = os.path.join(SERVER_FOLDER, 'tests.zip')

KEY = 'KEYS'
KEY_LABEL = 'label'
KEY_ORIGIN = 'origin'
VALUE_ORIGIN_SYNTHETIC = 'synthetic'
VALUE_ORIGIN_POC = 'poc'
VALUE_ORIGIN_COLLECTION = 'collection'

IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'tif', 'tiff']
SKIPS = ['.DS_Store']

BOX_FOLDER_DATA_ZIPPED = 'data_zipped'
BOX_FOLDER_METADATA = 'metadata'
BOX_FOLDER_VERSIONS = 'versions'

BOX_API_CLIENT_ID = 'ov450518faxmbskxmg489u6eujldrtjf'
BOX_API_CLIENT_SECRET = 'ylPy1l25DiSzgKK2gFDbjIwr2WUA140X'
BOX_API_RETURN_URI = 'https://127.0.0.1:5000/return'



def create_backup(input_folder, input_json, backup_name):
    """Create backup folder

    # Arguments
        input_folder [str]: the folder
        input_json [str]: the json file
        backup_name [str]: the backup folder name
    """
    backup_folder = os.path.join(
        os.path.dirname(input_folder),
        backup_name)
    print('Creating backup folder at {}...'.format(backup_folder))

    os.mkdir(backup_folder)
    subprocess.run(['cp', '-r', input_folder, backup_folder])
    subprocess.run(['cp', input_json, backup_folder])

    return backup_folder


def get_all_metadata():
    """Get all json information

    # Returns
        [dict]: the metadata from all json
    """
    json_files = glob.glob(
        os.path.join(get_base_path(FOLDER_METADATA), '*.json'))
    features = {}
    for each_json_file in json_files:
        with open(each_json_file, 'r') as f_in:
            each_features = json.load(f_in)
            features.update(each_features)

    return features


def update_user_config(update_dictionary):
    """Update the user configuration

    # Arguments
        update_dictionary [dict]: the dictionary to update from old one

    # Returns
        [dict]: the updated configuration
    """
    folder_base = get_base_path()
    file_user_config = os.path.join(folder_base, FILE_USER_CONFIG)

    # create the config file if not existed
    if not os.path.exists(file_user_config):
        with open(file_user_config, 'w') as f_out:
            json.dump({}, f_out)

    with open(file_user_config, 'r') as f_in:
        configs = json.load(f_in)

    configs.update(update_dictionary)
    with open(file_user_config, 'w') as f_out:
        json.dump(configs, f_out, indent=4, separators=(',', ': '))

    return configs


def get_key_value():
    """Get the key and value options

    # Returns
        [dict]: dictionary of key and expected values
    """
    with open(get_base_path(FILE_SYSTEM_CONFIG), 'r') as f_in:
        _config = json.load(f_in)

    return _config[KEY]


def get_bad_files():
    """Get bad files

    These bad files are obtained from `FILE_BAD`. Each line in file bad can be
    either (1) a path to a bad file, or (2) or globable pattern to obtain file
    paths.

    # Returns
        [list of str]: a list of bad images
    """
    folder_data = get_base_path(FOLDER_DATA)
    bad_images = []
    with open(os.path.join(get_base_path(FOLDER_METADATA), FILE_BAD)) as f_in:
        lines = f_in.read().splitlines()
        for each_line in lines:
            filepath = os.path.join(folder_data, each_line)
            if '*' not in each_line:
                bad_images.append(filepath)
            else:
                bad_images += glob.glob(filepath, recursive=True)

    return bad_images
