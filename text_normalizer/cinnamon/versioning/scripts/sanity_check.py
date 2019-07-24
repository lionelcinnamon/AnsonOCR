"""Check the validity of the minisystem"""
import glob
import json
import os

import arrow

import utils


def check_good_input_source(input_folder, input_json):
    """Check whether the input source is good for copy

    Currently, this methods check for:
        - filename convention: [project_name]_[YYYYMMDD]_[counting_idx].[xyz]
        - the number of entries in `input_json` is the same as the number
        of files in `input_folder`
        - the filepath key in `input_json` links to legit file in `input_folder`
        - (nice to have) warnings about keys that similar to keys already in
            other json files

    # Arguments
        input_folder [str]: the folder containing images
        input_json [str]: the path to json file

    # Exceptions
        [Exception]: raise error when the input folder and json is bad
    """

    # check for equal amount of files and entries in `input_folder` and
    # `input_json`
    print('Checking number of images is equal to number of keys in json...')
    files = glob.glob(os.path.join(input_folder, '**', '*.*'), recursive=True)
    with open(input_json, 'r') as f_in:
        features = json.load(f_in)
    if len(files) != len(features):
        raise Exception('the amount of files in folders is different from the '
                        'amount of files in json: {} vs {}'
                        .format(len(files), len(features)))

    # check that filepath in `input_json` is legit
    print('Checking the filepath in json file is legit...')
    base_path = input_folder + '/' if input_folder[-1] != '/' else input_folder
    files = set([each_file.replace(base_path, '') for each_file in files])
    features_keys = set(features.keys())
    only_features_keys = features_keys.difference(files)
    only_files = files.difference(features_keys)
    if len(only_features_keys) > 0:
        raise Exception('there are {} files in json but does not in folder, '
                        'they are {}'
                        .format(len(only_features_keys), only_features_keys))
    if len(only_files) > 0:
        raise Exception('there are {} files in folder but not in json, they '
                        'are {}'.format(len(only_files), only_files))

    # check that each value in `input_json` has the label and origin key
    print('Checking that each value in json file has label and origin key...')
    for filename, file_attribute in features.items():
        if utils.KEY_LABEL not in file_attribute:
            raise Exception('{} does not contain `{}` key'.format(
                filename, utils.KEY_LABEL))

        if utils.KEY_ORIGIN not in file_attribute:
            raise Exception('{} does not contain `{}` key'.format(
                filename, utils.KEY_ORIGIN))

    # check good filename format
    print('Checking filename format...')
    for each_file in list(files):
        filename = os.path.basename(each_file)
        filename, _ = os.path.splitext(filename)

        try:
            _, date, idx = filename.split('_')
        except ValueError:
            raise Exception('the number of "_" is not 3 with this filename {}, '
                            'please make sure that filename has '
                            '[ProjectInCamelCase]_[YYYYMMDD]_[IdxInteger].[ext]'
                            ' format.'
                            .format(each_file))

        try:
            date = arrow.get(date, 'YYYYMMDD')
        except arrow.parser.ParserError:
            raise Exception('the date does not have YYYYMMDD format for file {}'
                            .format(each_file))

        try:
            idx = int(idx)
        except ValueError:
            raise Exception('the index is not an integer for file {}'
                            .format(each_file))

    print('Checked!')


def check_central_repo():
    """Check that there is no problem in central repo

    This method can confirm that:
        - the number of files in json is equal to the number of files in folder
        - the filepath in json is legit
        - the filename format is <Project>_<YYYYMMDD>_<index>.<ext>
        - important files and folders exist
    """
    # check important files and folders exist
    print('\n=> Check important files and folders exist')
    for each_folder in [utils.FOLDER_DATA, utils.FOLDER_WORK,
        utils.FOLDER_DATA_ZIPPED, utils.FOLDER_METADATA, utils.FOLDER_SCRIPT,
        utils.FOLDER_VERSIONS]:
        folder_path = utils.get_base_path(each_folder)
        if not os.path.exists(folder_path):
            print(':WARNING: folder {} does not exist'.format(folder_path))

    for each_file in [os.path.join(utils.FOLDER_METADATA, utils.FILE_BAD),
        utils.FILE_USER_CONFIG]:
        file_path = utils.get_base_path(each_file)
        if not os.path.exists(file_path):
            print(':WARNING: file {} does not exist'.format(file_path))


    features = utils.get_all_metadata()
    folder_data = utils.get_base_path(utils.FOLDER_DATA) + '/'
    images = glob.glob(os.path.join(folder_data, '**', '*.*'), recursive=True)
    images = set([each_file.replace(folder_data, '') for each_file in images])

    # check the number of files is equal to the number of files in folder
    print('\n=> Check the number of file in json is equal to that in folder')
    if len(features) == len(images):
        print('Equal files ({})'.format(len(features)))
    else:
        print(':WARNING: not equal: json {}, files {}'
            .format(len(features), len(images)))


    # check filename format is <Project>_<YYYYMMDD>_<index>.<ext>
    print('\n=> Check filename format is <Project>_<YYYYMMDD>_<index>.<ext>')
    for each_file in list(images):
        filename = os.path.basename(each_file)
        filename, _ = os.path.splitext(filename)
        bad = False

        try:
            _, date, idx = filename.split('_')
        except ValueError:
            bad = True

        try:
            date = arrow.get(date, 'YYYYMMDD')
        except arrow.parser.ParserError:
            bad = True

        try:
            idx = int(idx)
        except ValueError:
            bad = True

        if bad:
            print(':WARNING: {}'.format(each_file))


    # check filepath in json is legit
    print('\n=> Check file path in json is legit')
    features_keys = set(features.keys())
    only_features_keys = features_keys.difference(images)
    only_files = images.difference(features_keys)
    if len(only_features_keys) > 0:
        print('there are {} files in json but does not in folder, they are {}'
            .format(len(only_features_keys), only_features_keys))
    if len(only_files) > 0:
        print('there are {} files in folder but not in json, they are {}'
            .format(len(only_files), only_files))

    print('\nFinished!')


def query_contain(key, matching_value, features=None):
    """Query the metadata for files using `in` scheme

    Includes a file if `matching_value in features[filename][key]`

    # Arguments
        key [str]: contain key
        matching_value [str]: matching value
        features [dict]: all features (allow concatenating query)

    # Returns
        [dict]: all matched results
    """
    features = utils.get_all_metadata() if features is None else features
    results = {}

    # Iterate each element to see if it matches
    for filepath, feature in features.items():
        value = feature.get(key, None)
        if value is not None and matching_value in value:
            results[filepath] = feature

    return results


def query_exact(key, matching_value, features=None):
    """Query the metadata for exact matching

    Includes a file if `matching_value == features[filename][key]`

    # Arguments
        key [str]: contain key
        matching_value [str]: matching value
        features [dict]: all features (allow concatenating query)

    # Returns
        [dict]: all matched results
    """
    features = utils.get_all_metadata() if features is None else features
    results = {}

    # Iterate each element to see if it matches
    for filepath, feature in features.items():
        value = feature.get(key, None)
        if value is not None and matching_value == value:
            results[filepath] = feature

    return results



if __name__ == '__main__':
    """Run this block when the script is directly called"""

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='repo',
        help='Whether sanity check the whole repo')
    args = parser.parse_args()

    if args.mode == 'repo':
        check_central_repo()
