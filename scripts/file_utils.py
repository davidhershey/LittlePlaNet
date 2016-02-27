
import os

import aws_s3_utility


def is_valid(key):
    return key.replace('-','').isalnum()

def load_key(filepath):
    assert os.path.exists(filepath), 'filepath: {} not found'.format(filepath)
    
    key = None
    with open(filepath, 'rb') as f:
        key = f.readline()
    if is_valid(key):
        return key
    else:
        raise ValueError('invalid key: {}'.format(key))

def upload_directory_to_aws(directory):
    ak = load_key('../access_key.key')
    sk = load_key('../secret_key.key')
    bucket = 'littleplanet'
    aws_util = aws_s3_utility.S3Utility(ak, sk, bucket)
    aws_util.upload_directory(directory)

if __name__ == '__main__':
    directory_names = ['Barcelona', 'DC', 'Detroit', 'London', 'Moscow', 'NYC', 'Paris', 'Rio', 'SanFran', 'Sydney']
    for directory_name in directory_names:
        directory_path = os.path.join('../imgs', directory_name)
        upload_directory_to_aws(directory_path)
