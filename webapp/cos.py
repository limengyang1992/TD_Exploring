# -*- coding=utf-8
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import os
import sys
import logging
from glob import glob

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

def cos_download(cos_path, local_path):
    response = client.get_object(
        Bucket='data-behaviour-1254003045',
        Key=cos_path,
    )
    response['Body'].get_stream_to_file(local_path)


#下载 训练数据
def get_train_dataset(root_path="behaviour_dataset"):
    
    if not os.path.exists(root_path):
        os.makedirs(root_path)
        
    cifar10_x = f"{root_path}/cifar10_x_merge.npy"
    cifar10_y = f"{root_path}/cifar10_y_merge.npy"
    cifar100_x = f"{root_path}/cifar100_x_merge.npy"
    cifar100_y = f"{root_path}/cifar100_y_merge.npy"
    cos_download(cifar10_x, cifar10_x)
    cos_download(cifar10_y, cifar10_y)
    cos_download(cifar100_x, cifar100_x)
    cos_download(cifar100_y, cifar100_y)
    

#下载 行为数据
def get_behaviour_dataset(dirs,epochs=200):
    if dirs[:4] == "exps":
        dirs = dirs[5:]
    
    root_dir = os.path.join("exps",dirs)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        os.makedirs(os.path.join(root_dir,"runs"))
    
    log_path = 'exps/' + dirs + '/log.csv'
    args_path = 'exps/' + dirs + '/args.txt'
    logit_path = ['exps/' + dirs + '/runs/'+ f"logit_{x}.npy" for x in range(epochs)]
    topk_path = ['exps/' + dirs + '/runs/'+ f"topk_{x}.npy" for x in range(epochs)]
    
    cos_download(log_path,log_path)
    cos_download(args_path,args_path)

    # for logit_p in logit_path:
    #     cos_download(logit_p,logit_p)
        
    for topk_p in topk_path:
        try:
            cos_download(topk_p,topk_p)  
        except:
            pass
            
            
def get_task_list():
    result_list = []
    response = client.list_objects(
    Bucket='data-behaviour-1254003045', Prefix='exps/', Delimiter='/')
    if 'CommonPrefixes' in response:
        for folder in response['CommonPrefixes']:
            result_list.append(folder['Prefix'])
    
    return result_list

        
if __name__ == "__main__":
    
    get_train_dataset()
    
    result = get_task_list()
    print(result)
    
    for choose_task in result:
        get_behaviour_dataset(choose_task)
    