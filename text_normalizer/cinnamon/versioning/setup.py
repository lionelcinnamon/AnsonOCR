# Setup data versioning mini-system locally for any project
# @author: _john
# ==============================================================================
import argparse
import json
import os
import subprocess
from pprint import pprint

import utils


def base_path():
    """Get the base folder"""
    return os.path.dirname(os.path.realpath(__file__))


def copy_files(target_folder):
    """Copy all data versioning scripts to `target_folder`

    # Arguments
        target_folder [str]: the base data versioning folder
    """
    data_versioning_folder = os.path.join(target_folder, 'data_versioning')
    os.makedirs(data_versioning_folder, exist_ok=True)

    subprocess.run([
        'cp',
        '-r',
        os.path.join(base_path(), 'scripts'), data_versioning_folder
    ])

    subprocess.run([
        'cp', os.path.join(base_path(), 'README.md'), data_versioning_folder
    ])


def create_system_config(target_folder):
    """Create a system config file in the target folder

    # Arguments
        target_folder [str]: the base data versioning folder
    """
    data_versioning_folder = os.path.join(target_folder, 'data_versioning')
    os.makedirs(data_versioning_folder, exist_ok=True)

    print(
        'Please answer these following questions, or you can edit the '
        'system_config.json file later')
    SERVER_IP_INPUT = input('Please specify the server IP: ').strip()
    SERVER_PORT_INPUT = input('Please specify the server port: ').strip()
    SERVER_FOLDER_INPUT = input(
        'Please specify the server folder directory: ').strip()
    BOX_FOLDER_BASE_INPUT = input(
        'Please specify the folder ID of Box backup folder: ').strip()
    KEYS = {}
    while True:
        KEY_INPUT = input(
            'Please specify targeted key (blank to skip): ').strip()
        if not KEY_INPUT:
            break
        VALUE_INPUT = input(
            'Please specify expected values (blank for unexpected): ').strip()
        if VALUE_INPUT:
            VALUE_INPUT = VALUE_INPUT.split(', ')

        KEYS[KEY_INPUT] = VALUE_INPUT

    # obtain the result and set up
    obj = {
        'SERVER_IP': SERVER_IP_INPUT,
        'SERVER_PORT': SERVER_PORT_INPUT,
        'SERVER_FOLDER': SERVER_FOLDER_INPUT,
        'BOX_FOLDER_BASE': BOX_FOLDER_BASE_INPUT,
        'KEYS': KEYS
    }
    output_json = os.path.join(data_versioning_folder, 'system_config.json')
    with open(output_json, 'w') as f_out:
        json.dump(obj, f_out, indent=4, separators=(',', ': '))

    print('This is the config:')
    pprint(obj)


def create_gitignore(target_folder):
    """Add .gitignore

    This is important in code versioning, as we want to mask sensitive
    information as well as ignore large data

    # Arguments
        target_folder [str]: the base data versioning folder
    """
    data_versioning_folder = os.path.join(target_folder, 'data_versioning')
    os.makedirs(data_versioning_folder, exist_ok=True)

    ignores = [
        "# data in /data_versioning/",
        "/data_versioning/data/",
        "/data_versioning/metadata/",
        "/data_versioning/versions/",
        "/data_versioning/tests/",
        "/data_versioning/user_config.json",
        "/data_versioning/.work",
        "",
        "# python compiled code",
        "*.pyc",
        "__pycache__",
        "",
        "# os and other tools",
        ".DS_Store",
        ".ipynb_checkpoints",
        "*.swp",
        ".vscode/",
        ".idea/**",
        "*.pbxproj",
        "*.xcworkspacedata"
    ]

    with open(os.path.join(data_versioning_folder, '.gitignore'), 'w') as f_out:
        f_out.write('\n'.join(ignores))


if __name__ == '__main__':
    """Simplify the initialization process"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'target_folder', help='The folder containing data versioning')
    args = parser.parse_args()

    print('\n==> Copying scripts...')
    copy_files(args.target_folder)

    print('\n==> Constructing system configuration...')
    create_system_config(args.target_folder)

    print('\n==> Adding .gitignore...')
    create_gitignore(args.target_folder)

    print()
    print('Everything is set up at {}'.format(args.target_folder))
