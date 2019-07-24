"""Generate version text file"""
import argparse
import glob
import os

import numpy.random as random

import utils


def generate_new_version(version=None, name='version'):
    """Default choice for generating new text file version

    The default behavior is: All images - Bad images

    # Argumments
        version [str]: type <major>.<minor> . If None, <minor> will be
            increased by 1
        name [str]: name of the text file

    # Returns
        [str]: the path to new version text file
    """
    folder_version = utils.get_base_path(utils.FOLDER_VERSIONS)
    folder_data = utils.get_base_path(utils.FOLDER_DATA)
    folder_metadata = utils.get_base_path(utils.FOLDER_METADATA)

    # get the next minor version
    if version is None:
        olds = glob.glob(os.path.join(folder_version, '{}*.*.txt'.format(name)))

        if len(olds) == 0:
            version = '1.0'
        else:
            olds = [os.path.basename(each_file) for each_file in olds]
            olds = [os.path.splitext(each_file)[0] for each_file in olds]
            olds = [each_file.replace(name, '') for each_file in olds]
            olds = [each_file.split('.') for each_file in olds]

            old_versions = []
            for major, minor in olds:
                try:
                    old_versions.append((int(major), int(minor)))
                except ValueError:
                    continue

            old_versions.sort(key=lambda obj: obj[0])
            old_versions.sort(key=lambda obj: obj[1])

            major, minor = old_versions[-1]
            version = '{}.{}'.format(major, minor + 1)

    # get valid images
    all_images = set(glob.glob(
        os.path.join(folder_data, '**', '*.*'), recursive=True))

    bad_images = set(utils.get_bad_files())
    out_images = list(all_images.difference(bad_images))
    out_images.sort()

    # get label
    features = utils.get_all_metadata()
    folder_data += '/'
    finals = []
    for each_file in out_images:
        filename = each_file.replace(folder_data, '')
        label = (
            features[filename] if isinstance(features[filename], str) else
            features[filename][utils.KEY_LABEL]
        )
        finals.append('{},{}'.format(filename, label))

    # dump the version
    target_txt = os.path.join(folder_version, '{}{}.txt'.format(name, version))
    with open(target_txt, 'w') as f_out:
        f_out.write('\n'.join(finals))

    print('The generated version {}'.format(target_txt))

    return target_txt


def generate_train_test(version_name, condition, version=None):
    """Generate the train and test

    Under this method:
        - POC samples have 80% probability in test set
        - collection samples have 20% probability in test set
        - synthetic samples have 0% probabilty in test set

    # Arguments
        field [str]: the field. Currently support name, address, all
        version [str]: <major>.<minor>

    # Returns
        [str]: path to train version
        [str]: path to test version
    """
    folder_version = os.path.join(
        utils.get_base_path(utils.FOLDER_VERSIONS), version_name)
    folder_data = utils.get_base_path(utils.FOLDER_DATA)
    folder_metadata = utils.get_base_path(utils.FOLDER_METADATA)

    # sanity check
    os.makedirs(folder_version, exist_ok=True)

    # get the next minor version
    if version is None:
        olds = glob.glob(os.path.join(
            folder_version, '{}-train*.*.txt'.format(version_name)))

        if len(olds) == 0:
            version = '1.0'
        else:
            olds = [os.path.basename(each_file) for each_file in olds]
            olds = [os.path.splitext(each_file)[0] for each_file in olds]
            olds = [each_file.replace('{}-train'.format(version_name), '')
                for each_file in olds]
            olds = [each_file.split('.') for each_file in olds]

            old_versions = []
            for major, minor in olds:
                try:
                    old_versions.append((int(major), int(minor)))
                except ValueError:
                    continue

            old_versions.sort(key=lambda obj: obj[0])
            old_versions.sort(key=lambda obj: obj[1])

            major, minor = old_versions[-1]
            version = '{}.{}'.format(major, minor + 1)

    # get valid images
    all_images = set(glob.glob(
        os.path.join(folder_data, '**', '*.*'), recursive=True))

    # get bad images
    bad_images = set(utils.get_bad_files())
    out_images = list(all_images.difference(bad_images))
    out_images.sort()

    # get label
    features = utils.get_all_metadata()
    folder_data += '/'
    finals_train, finals_test = [], []
    for each_file in out_images:
        filename = each_file.replace(folder_data, '')

        proceed = True
        for key, value in condition.items():
            if value != features[filename].get(key, None):
                proceed = False
        if not proceed:
            continue

        label = features[filename][utils.KEY_LABEL]
        origin = features[filename][utils.KEY_ORIGIN]

        # train/test split based on origin is poc, collection or synthetic
        if origin == utils.VALUE_ORIGIN_POC:
            if random.random() > 0.8:
                finals_train.append('{},{}'.format(filename, label))
            else:
                finals_test.append('{},{}'.format(filename, label))
        elif origin == utils.VALUE_ORIGIN_COLLECTION:
            if random.random() > 0.2:
                finals_train.append('{},{}'.format(filename, label))
            else:
                finals_test.append('{},{}'.format(filename, label))
        else:
            finals_train.append('{},{}'.format(filename, label))

    # dump the versions
    target_train = os.path.join(
        folder_version, '{}-train{}.txt'.format(version_name, version))
    target_test = os.path.join(
        folder_version, '{}-test{}.txt'.format(version_name, version))

    with open(target_train, 'w') as f_out:
        f_out.write('\n'.join(finals_train))
    with open(target_test, 'w') as f_out:
        f_out.write('\n'.join(finals_test))

    print('The generated train {}'.format(target_train))
    print('The generated test {}'.format(target_test))

    return target_train, target_test


if __name__ == '__main__':
    """Execute when this file is directly called"""

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('name', help='the name')
    parser.add_argument('condition', help='key1:value1,key2:value2..')
    parser.add_argument('--version', default=None,
        help='the next version to generate, should be <major>.<minor>; leave '
             'out for auto-increment the last minor version')
    args = parser.parse_args()

    condition_list = args.condition.split(',')
    condition = {
        each.split(':')[0]: each.split(':')[1]
        for each in condition_list
    }

    _ = generate_train_test(version_name=args.name,
        condition=condition, version=args.version)
