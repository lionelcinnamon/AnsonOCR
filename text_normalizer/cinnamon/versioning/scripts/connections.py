"""Initialize the minisystem"""

import glob
import json
import os
import subprocess
import zipfile

import arrow
from boxsdk import Client, OAuth2

import utils


def fresh_initialize():
    """Initialize the minisystem"""

    # Working files and directories
    folder_base = utils.get_base_path()
    folder_data = utils.get_base_path(utils.FOLDER_DATA)
    folder_metadata = utils.get_base_path(utils.FOLDER_METADATA)
    folder_versions = utils.get_base_path(utils.FOLDER_VERSIONS)
    folder_work = utils.get_base_path(utils.FOLDER_WORK)
    folder_data_zipped = utils.get_base_path(utils.FOLDER_DATA_ZIPPED)
    file_bad = os.path.join(folder_metadata, utils.FILE_BAD)
    file_user_config = os.path.join(folder_base, utils.FILE_USER_CONFIG)

    # Creating folders
    os.makedirs(folder_data, exist_ok=True)
    os.makedirs(folder_metadata, exist_ok=True)
    os.makedirs(folder_versions, exist_ok=True)
    os.makedirs(folder_work, exist_ok=True)
    os.makedirs(folder_data_zipped, exist_ok=True)

    # Creating files
    with open(file_bad, 'w') as f_out:
        f_out.write('')
    with open(file_user_config, 'w') as f_out:
        json.dump({}, f_out)

    # Set up
    username = input('Username to access Hanoi server (to sync image data): ')
    utils.update_user_config({'username': username})

    print('Initialized!')


def pull_system_config_from_server():
    """Automatically system configuration from server"""

    file_user_config = os.path.join(
        utils.get_base_path(), utils.FILE_USER_CONFIG
    )
    folder_base = utils.get_base_path()

    username = None
    if os.path.exists(file_user_config):
        with open(file_user_config, 'r') as f_in:
            configs = json.load(f_in)
            username = configs.get('username', None)

    # Ask for username if not available
    if username is None:
        username = input('Username to access Hanoi server: ')
        utils.update_user_config({'username': username})

    print('Loading system configuration from server...')
    commands = [
        'rsync',
        '-vur',
        '-e',
        'ssh{}'.format(utils.SERVER_PORT),
        '--delete',
        '--progress',
        '{}@{}:{}'.format(
            username, utils.SERVER_IP, utils.FILE_SYSTEM_CONFIG),
        '{}/'.format(folder_base)]
    subprocess.run(commands)
    print('Finished!')


def pull_metadata_from_server():
    """Automatically metadata from server"""

    file_user_config = os.path.join(
        utils.get_base_path(), utils.FILE_USER_CONFIG
    )
    folder_base = utils.get_base_path()

    username = None
    if os.path.exists(file_user_config):
        with open(file_user_config, 'r') as f_in:
            configs = json.load(f_in)
            username = configs.get('username', None)

    # Ask for username if not available
    if username is None:
        username = input('Username to access Hanoi server: ')
        utils.update_user_config({'username': username})

    print('Loading metadata folder from server...')
    commands = [
        'rsync',
        '-vur',
        '-e',
        'ssh{}'.format(utils.SERVER_PORT),
        '--delete',
        '--progress',
        '{}@{}:{}'.format(
            username, utils.SERVER_IP, utils.SERVER_FOLDER_METADATA),
        '{}/'.format(folder_base)]
    subprocess.run(commands)
    print('Finished!')


def pull_versions_from_server():
    """Automatically update version information from server"""
    file_user_config = os.path.join(
        utils.get_base_path(), utils.FILE_USER_CONFIG
    )
    folder_base = utils.get_base_path()

    username = None
    if os.path.exists(file_user_config):
        with open(file_user_config, 'r') as f_in:
            configs = json.load(f_in)
            username = configs.get('username', None)

    # Ask for username if not available
    if username is None:
        username = input('Username to access Hanoi server: ')
        utils.update_user_config({'username': username})

    print('Loading version folder from server...')
    commands = [
        'rsync',
        '-vur',
        '-e',
        'ssh{}'.format(utils.SERVER_PORT),
        '--delete',
        '--progress',
        '{}@{}:{}'.format(
            username, utils.SERVER_IP, utils.SERVER_FOLDER_VERSIONS),
        '{}/'.format(folder_base)]
    subprocess.run(commands)

    print('Finished!')


def pull_data_from_server():
    """Automatically update image data from server"""

    file_user_config = os.path.join(
        utils.get_base_path(), utils.FILE_USER_CONFIG
    )
    folder_data = utils.get_base_path(utils.FOLDER_DATA)
    folder_work = utils.get_base_path(utils.FOLDER_WORK)
    folder_data_zipped = utils.get_base_path(utils.FOLDER_DATA_ZIPPED)

    username = None
    if os.path.exists(file_user_config):
        with open(file_user_config, 'r') as f_in:
            configs = json.load(f_in)
            username = configs.get('username', None)

    # Ask for username if not available
    if username is None:
        username = input('Username to access Hanoi server: ')
        utils.update_user_config({'username': username})

    print('Loading image folder from server...')
    os.makedirs(folder_data, exist_ok=True)
    commands = [
        'rsync',
        '-vur',
        '-e',
        'ssh{}'.format(utils.SERVER_PORT),
        '--delete',
        '--progress',
        '{}@{}:{}'.format(
            username, utils.SERVER_IP, utils.SERVER_FOLDER_DATA_ZIPPED),
        '{}'.format(folder_work)]
    subprocess.run(commands)
    zip_data = set([os.path.splitext(os.path.basename(each_file))[0]
        for each_file in glob.glob(os.path.join(folder_data_zipped, '*.zip'))])
    image_data = set([os.path.basename(each_file[:-1])
        for each_file in glob.glob(os.path.join(folder_data, '*/'))])
    update_data = list(zip_data.difference(image_data))
    for each_file in update_data:
        source_zip_file = os.path.join(
            folder_data_zipped, '{}.zip'.format(each_file))

        with zipfile.ZipFile(source_zip_file, 'r') as f_zipped:
            print('Unzipping {}...'.format(source_zip_file))
            f_zipped.extractall(folder_data)


def update_from_server():
    """Automatically pull everything from server"""
    pull_data_from_server()
    pull_metadata_from_server()
    pull_versions_from_server()
    pull_system_config_from_server()


def update_to_server():
    """Automatically udpate to server"""
    pass


def punch():
    """Notify server that this user is in charge of the next update"""
    pass


"""
Backup to box interaction
"""
def box_authenticate():
    """Authenticate to Box

    # Returns
        [Box Client]: the authenticated client
    """
    oauth = OAuth2(
        client_id=utils.BOX_API_CLIENT_ID,
        client_secret=utils.BOX_API_CLIENT_SECRET
    )

    auth_url, _ = oauth.get_authorization_url(utils.BOX_API_RETURN_URI)
    auth_code = input(
        'Go to this url {} and paste the value of `code` in redirected URL: '
        .format(auth_url))
    access_token, refresh_token = oauth.authenticate(auth_code)

    return Client(oauth)


def box_upload_file_to_folder(filepath, folder, client=None):
    """Upload a file to box folder

    # Arguments
        filepath [str]: the path to file
        folder [str]: folder id
        client [Box Client]: if None, create one

    # Returns
        [Box File]: the file
    """
    if client is None:
        client = box_authenticate()

    filename = os.path.basename(filepath)
    box_file = client.folder(folder).upload(filepath, filename)

    return box_file


def box_backup(base_path):
    """This code should be run on Hanoi server to upload to box

    # Arguments
        base_path [str]: base path containing folders `data_zipped`, `metadata`,
            `versions`

    # Returns
        [str]: backup folder sharable link
    """
    client = box_authenticate()

    folder_backup_data = os.path.join(base_path, utils.BOX_FOLDER_DATA_ZIPPED)
    folder_backup_metadata = os.path.join(base_path, utils.BOX_FOLDER_METADATA)
    folder_backup_versions = os.path.join(base_path, utils.BOX_FOLDER_VERSIONS)

    if not (os.path.exists(base_path) and os.path.exists(folder_backup_data) and
            os.path.exists(folder_backup_metadata) and
            os.path.exists(folder_backup_versions)):
        raise Exception('all of these folders must exist {}'.format(
            [base_path, folder_backup_data, folder_backup_metadata,
            folder_backup_versions]
        ))

    box_folder_base = client.folder(utils.BOX_FOLDER_BASE)
    box_folder_backup = box_folder_base.create_subfolder(
        arrow.now().format('YYYYMMDD-HHmmss'))
    box_folder_backup_data = box_folder_backup.create_subfolder(
        utils.BOX_FOLDER_DATA_ZIPPED)
    box_folder_backup_metadata = box_folder_backup.create_subfolder(
        utils.BOX_FOLDER_METADATA)
    box_folder_backup_versions = box_folder_backup.create_subfolder(
        utils.BOX_FOLDER_VERSIONS)

    print('\n=> Backup zipped data folder...')
    files = glob.glob(os.path.join(folder_backup_data, '*.zip'))
    for each_file in files:
        print(each_file)
        filename = os.path.basename(each_file)
        _ = box_folder_backup_data.upload(each_file, filename)

    print('\n=> Backup metadata folder...')
    files = glob.glob(os.path.join(folder_backup_metadata, '*'))
    for each_file in files:
        print(each_file)
        filename = os.path.basename(each_file)
        _ = box_folder_backup_metadata.upload(each_file, filename)

    print('\n=> Backup versions folder...')
    folder_versions = glob.glob(os.path.join(folder_backup_versions, '*/'))
    for each_folder in folder_versions:
        box_each_version = box_folder_backup_versions.create_subfolder(
            os.path.basename(each_folder[:-1])
        )
        files = glob.glob(os.path.join(each_folder, '*.txt'))
        for each_file in files:
            print(each_file)
            filename = os.path.basename(each_file)
            _ = box_each_version.upload(each_file, filename)

    share_link = box_folder_backup.get_shared_link()
    print('\n\nBackup to {}'.format(share_link))

    return share_link


if __name__ == '__main__':
    """Run this block if the script is directly called"""

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='pull',
        help='Pick from: pull, fresh, push')
    parser.add_argument('--pull', default='all',
        help='Pick from: all, data, metadata, versions')
    parser.add_argument('--backup_folder', default='/data/fixed_form_hw_data',
        help='Folder in server containing `data_zipped`, `versions`, and '
             '`metadata`')
    args = parser.parse_args()

    if args.mode == 'pull':
        if args.pull == 'all':
            update_from_server()
        elif args.pull == 'data':
            pull_data_from_server()
        elif args.pull == 'metdata':
            pull_metadata_from_server()
        elif args.pull == 'versions':
            pull_versions_from_server()
    elif args.mode == 'fresh':
        fresh_initialize()
    elif args.mode == 'push':
        _ = box_backup(args.backup_folder)
    else:
        print('Please pick --mode from ["pull", "fresh", "push"]')
