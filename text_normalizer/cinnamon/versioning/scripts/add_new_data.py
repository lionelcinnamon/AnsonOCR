"""Add new data into the repository and update csv data at the same time"""
import copy
import glob
import os
import json
import subprocess

import arrow

import sanity_check
import utils


def add_clean_images(input_folder, input_json, force=False):
    """Add the images into data folder

    @NOTE: this method assumes that the images in `input_folder` and the
    content in `input_json` is good to copy.

    # Arguments
        input_folder [str]: the path to input image folder
        input_json [str]: the path to base json information
        force [bool]: if True, then will force copy even if the output folder
            and output json exist

    # Returns
        [str]: the path to output folder image
    """

    # retrieve output folder name
    input_folder = input_folder[:-1] if input_folder[-1] == '/' else input_folder
    data_folder = utils.get_base_path(utils.FOLDER_DATA)
    folder_name = os.path.basename(input_folder)
    output_folder = os.path.join(data_folder, folder_name)

    # check if the output folder currently exists
    if os.path.exists(output_folder) and not force:
        raise AttributeError('the `output_folder` already exists: {}'
            .format(output_folder))

    # retrieve output metadata name
    metadata_folder = utils.get_base_path(utils.FOLDER_METADATA)
    output_json = os.path.join(metadata_folder,os.path.basename(input_json))

    # check if the output json file currently exists
    if os.path.exists(output_json) and not force:
        raise AttributeError('the `output_json` already exists: {}'
            .format(output_json))

    subprocess.run(['cp', '-r', input_folder, output_folder])
    subprocess.run(['cp', input_json, output_json])

    print('The output folder is {}'.format(output_folder))
    print('The output json is {}'.format(output_json))

    return output_folder, output_json


def adjust_filename(input_folder, input_json, project, time=None):
    """Modify images and json into correct filename convention

    This method automatically adjust filename into desired format. It will also
    create a backup version in `dataversioning_backup_adjust_filename`.

    # Arguments
        input_folder [str]: the path to input folder
        input_json [str]: the path to input json
        project [str]: the project name
        time [str]: the time in YYYYMMDD format, if is None, then take the
            current time
        append_root [bool]: whether to create a root folder

    # Returns
        [str]: project name
        [str]: time
    """
    input_folder = input_folder[:-1] if input_folder[-1] == '/' else input_folder
    _ = utils.create_backup(
        input_folder, input_json, 'dataversioning_backup_adjust_filename')

    print('Loading images and labels...')
    with open(input_json, 'r') as f_in:
        features = json.load(f_in)
        features_new = {}

    if time is None:
        time = arrow.now().format('YYYYMMDD')
    else:
        _ = arrow.get(time, 'YYYYMMDD')

    digits = len(str(len(features))) + 1
    print('Modifying images and labels...')
    for idx, each_file in enumerate(features.keys()):
        directory = os.path.dirname(each_file)
        _, ext = os.path.splitext(each_file)

        filename_new = '{project}_{time}_{idx:0{digits}d}{ext}'.format(
            project=project, time=time, idx=idx, digits=digits, ext=ext)
        filename_new = os.path.join(directory, filename_new)

        features_new[filename_new] = features[each_file]
        subprocess.run(['mv', os.path.join(input_folder, each_file),
                        os.path.join(input_folder, filename_new)])

    with open(input_json, 'w') as f_out:
        json.dump(features_new, f_out, indent=4, separators=(',', ': '),
                  sort_keys=True, ensure_ascii=False)

    print('Finish!')

    return project, time


def batch_json_modification(input_json):
    """Modify json so that it conforms to desired json structure format
    
    # Arguments
        input_json [str]: the path to json file
    
    # Returns
        [dict]: the json file or None if some check is invalid
    """
    with open(input_json, 'r') as f_in:
        features = json.load(f_in)
    
    proceed = True
    features_update = {}
    keys = utils.get_key_value()

    # ask about origin
    while True:
        supported_origins = [
            utils.VALUE_ORIGIN_SYNTHETIC,
            utils.VALUE_ORIGIN_POC,
            utils.VALUE_ORIGIN_COLLECTION]

        user_origin = input(
            'Dataset **origin**.Supported {}: '.format(
                supported_origins
            )).strip()
        
        if user_origin in supported_origins:
            features_update[utils.KEY_ORIGIN] = user_origin
            break
        else:
            print('Invalid, expect 1 of {} but receive {}'.format(
                supported_origins, user_origin
            ))

    # ask about other user-defined information
    for key, supported_values in keys.items():
        if isinstance(supported_values, list):
            user_input = 'Please specify **{}**. Supported {}: '.format(
                key, supported_values
            )
        else:
            user_input = 'Please specify **{}**: '.format(key)
        
        user_input = input(user_input).strip()

        if user_input and isinstance(supported_values, str):
            # free-form input (except when user leaves blank)
            features_update[key] = user_input
        elif user_input in supported_values:
            # select
            features_update[key] = user_input
        else:
            print('Invalid, expect 1 of {} but receive {}'.format(
                supported_values, user_input
            ))
            proceed = False
            break

    if not proceed:
        return

    features_new = {}
    for each_filename, each_feature_ in features.items():
        each_feature = ({utils.KEY_LABEL: each_feature_} 
            if isinstance(each_feature_, str)
            else copy.deepcopy(each_feature_))
        each_feature.update(features_update)

        if utils.KEY_LABEL not in each_feature:
            print(':WARNING: {}\'s object does not contain key label'
                .format(each_filename))
        
        features_new[each_filename] = each_feature
    
    with open(input_json, 'w') as f_out:
        json.dump(features_new, f_out, indent=4, separators=(',', ': '),
                  ensure_ascii=False, sort_keys=True)
    
    return features_new
        
    
def add_root(root, input_json, input_folder):
    """Append the root for json

    This method modifies the json and (optinally) folder, such that:
        - `root` will be prepend to filepath keys in `input_json`
        - the `input_json` file will be renamed into `root`.json
        - `input_folder` will be renamed into `root`

    # Arguments
        root [str]: the rootname
        input_json [str]: path to input json
        input_folder [str]: path to input folder

    # Returns
        [str]: target output folder
        [str]: target output json
    """
    input_folder = (input_folder[:-1] if input_folder[-1] == '/'
                    else input_folder)
    _ = utils.create_backup(
        input_folder, input_json, 'dataversioning_backup_add_root')

    with open(input_json, 'r') as f_in:
        features = json.load(f_in)
        features_new = {}

    print('Adding the root to json keys...')
    for each_file in features.keys():
        filename_new = os.path.join(root, each_file)
        features_new[filename_new] = features[each_file]

    target_folder = os.path.join(os.path.dirname(input_folder), root)
    subprocess.run(['mv', input_folder, target_folder])
    print('Moved the folder into {}'.format(target_folder))

    target_json = os.path.join(
        os.path.dirname(input_json),
        '{}.json'.format(root)
    )

    with open(target_json, 'w') as f_out:
        json.dump(features_new, f_out, ensure_ascii=False, indent=4,
                  separators=(',', ': '), sort_keys=True)
    _ = subprocess.run(['rm', input_json])
    print('Dumped the new json file into {}'.format(target_json))

    return target_folder, target_json


def add_images(input_folder, input_json, project, time=None, force=False):
    """Add the images into repo

    This method will sequentially:
        1. Change filename format in the json
        2. Create and append the root folder
        3. Copy the folder and json into repo

    # Arguments
        input_folder [str]: the path to folder containing images
        input_json [str]: the path to json file
        project [str]: the project name
        force [bool]: if True, then will force adding images even when the
            output folder and output json exists
    """
    user_batch_json_info = input('\n=>Add batch information to json [Y/n]: ')
    user_batch_json_info = user_batch_json_info.strip().lower()
    if user_batch_json_info in ['', 'y', 'yes']:
        _ = batch_json_modification(input_json)

    print('\n=> Adjust filename')
    project, time = adjust_filename(input_folder, input_json, project, time)
    root_dir = '{}_{}'.format(project, time)

    print('\n=> Sanity check')
    sanity_check.check_good_input_source(input_folder, input_json)

    print('\n=> Append root')
    input_folder, input_json = add_root(root_dir, input_json, input_folder)

    print('\n=> Copy images')
    output_folder, output_json = add_clean_images(
        input_folder, input_json, force)

    return output_folder, output_json


if __name__ == '__main__':
    """Run the script is directly called"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', help='the folder containing images')
    parser.add_argument('input_json', help='the json containing information')
    parser.add_argument('project', help='the project name (CamelCase)')
    parser.add_argument('--time', default=None,
        help='the time in YYYYMMDD; leave out to use current time')
    parser.add_argument('--force', action='store_true',
        help='force copying images even when files already existed')
    args = parser.parse_args()

    add_images(input_folder=args.input_folder, input_json=args.input_json,
               project=args.project, time=args.time, force=args.force)
