# -*- coding=utf-8
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
secret_id = 'AKIDTmqdlKteW9uPglwmTu6JFCxYxIRWnu76'     
secret_key = 'sUw7FBsLZYJr9MM6ctwkLhyFKAkOp34R'   
region = 'ap-beijing'      
config = CosConfig(Region=region, 
                   SecretId=secret_id, 
                   SecretKey=secret_key, 
                   Token=None, 
                   Scheme='https')
client = CosS3Client(config)


def cos_upload_file(path_file):
    response = client.upload_file(
        Bucket='data-behaviour-1254003045',
        LocalFilePath=path_file,
        Key=path_file,
        PartSize=1,
        MAXThread=10,
        EnableMD5=False
    )
    
def cos_upload_dir(path_dir):
    for root, dirs, files in os.walk(path_dir):
        for i, f in enumerate(files):
            path_file = os.path.join(root, f)
            print(path_file)
            cos_upload_file(path_file)


# upload_path = "behaviour_dataset"

# cos_upload_file("out/totalz.npz")
# cos_upload_file("out/totaly.npy")