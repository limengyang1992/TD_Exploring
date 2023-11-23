import os
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from cos import get_task_list,get_behaviour_dataset

def choose_type(type,label_df):
    df_type = label_df[label_df["type"] == type]
    df_index = list(df_type.index)
    return df_index


def to_pil(data):
    data = data.astype(np.uint8)
    r = Image.fromarray(data[0]).convert('L')
    g = Image.fromarray(data[1]).convert('L')
    b = Image.fromarray(data[2]).convert('L')
    pil_img = Image.merge('RGB', (r, g, b))
    p = pil_img.resize((256, 256))
    return p


def plt_line(img, topk):
    list_ccc = st.columns(topk)

    for col in list_ccc:
        with col:
            st.image(img, use_column_width=True)
            st.write("label")


def show_epoch_sims(index, epoch,root_topk):
    numpy_sims = np.load(os.path.join(root_topk,f"topk_{epoch}.npy"))
    df_sims = pd.DataFrame(numpy_sims, columns=["index", "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9",
                                                "score", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"])

    sim_index = [int(x)
                 for x in df_sims[df_sims["index"] == index].values[0]][0:10]
    sim_score = [x for x in df_sims[df_sims["index"] == index].values[0]][10:]
    return sim_index, sim_score


########################################################################################################################

# st.sidebar.title(f'数据行为可视化')

total_task_list = get_task_list()
choose_task = st.sidebar.selectbox("任务列表:", total_task_list)
model_type = choose_task[5:].split("_")[1]
dataset_type = choose_task[5:].split("_")[2]

st.sidebar.write(f"模型:{model_type[2:]},数据:{dataset_type[2:] }")


# st.sidebar.title(pd.DataFrame({'模型': [model_type[2:]],
#                                '数据': [dataset_type[2:]] 
#                                } ))
                                                    

genre = st.sidebar.radio("展示类型:",('训练日志', '数据行为'),horizontal=True)

if genre == "训练日志":
    
    log_path = choose_task+"log.csv"
    
    chart_data = pd.read_csv(log_path,usecols=["acc","val_acc"])
    st.line_chart(chart_data)

    chart_data = pd.read_csv(log_path,usecols=["loss","val_loss"])
    st.line_chart(chart_data)
  
    chart_data = pd.read_csv(log_path,usecols=["lr"])
    st.line_chart(chart_data)  
  
########################################################################################################################


else:
    
    print("dataset_type:",dataset_type)
    root_topk = choose_task + '/runs/'
    if "cifar100" not in dataset_type:
        dataset_img = np.load(r"behaviour_dataset/cifar10_x_merge.npy")
        dataset_img = (255*(dataset_img/2+0.5)).astype(int).clip(min=0, max=255)
        dataset_label = np.load(r"behaviour_dataset/cifar10_y_merge.npy")

        dataset_label_name = {0: 'airplane', 1: 'automobile', 2: 'brid', 3: 'cat',
                            4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
        
    else:
        dataset_img = np.load(r"behaviour_dataset/cifar100_x_merge.npy")
        dataset_img = (255*(dataset_img/2+0.5)).astype(int).clip(min=0, max=255)
        dataset_label = np.load(r"behaviour_dataset/cifar100_y_merge.npy")
        dataset_label_name = {19: 'cattle', 29: 'dinosaur', 0: 'apple', 11: 'boy', 1: 'aquarium_fish', 86: 'telephone',
                            90: 'train', 28: 'cup', 23: 'cloud', 31: 'elephant',
                            39: 'keyboard', 96: 'willow_tree', 82: 'sunflower', 17: 'castle', 71: 'sea', 8: 'bicycle',
                            97: 'wolf', 80: 'squirrel', 74: 'shrew',
                            59: 'pine_tree', 70: 'rose', 87: 'television', 84: 'table', 64: 'possum', 52: 'oak_tree',
                            42: 'leopard', 47: 'maple_tree', 65: 'rabbit',
                            21: 'chimpanzee', 22: 'clock', 81: 'streetcar', 24: 'cockroach', 78: 'snake', 45: 'lobster',
                            49: 'mountain', 56: 'palm_tree', 76: 'skyscraper',
                            89: 'tractor', 73: 'shark', 14: 'butterfly', 9: 'bottle', 6: 'bee', 20: 'chair', 98: 'woman',
                            36: 'hamster', 55: 'otter', 72: 'seal', 43: 'lion',
                            51: 'mushroom', 35: 'girl', 83: 'sweet_pepper', 33: 'forest', 27: 'crocodile', 53: 'orange',
                            92: 'tulip', 50: 'mouse', 15: 'camel',
                            18: 'caterpillar', 46: 'man', 75: 'skunk', 38: 'kangaroo', 66: 'raccoon', 77: 'snail',
                            69: 'rocket', 95: 'whale', 99: 'worm',
                            93: 'turtle', 4: 'beaver', 61: 'plate', 94: 'wardrobe', 68: 'road', 34: 'fox', 32: 'flatfish',
                            88: 'tiger', 67: 'ray',
                            30: 'dolphin', 62: 'poppy', 63: 'porcupine', 40: 'lamp', 26: 'crab', 48: 'motorcycle',
                            79: 'spider', 85: 'tank', 54: 'orchid',
                            44: 'lizard', 7: 'beetle', 12: 'bridge', 2: 'baby', 41: 'lawn_mower', 37: 'house',
                            13: 'bus', 25: 'couch', 10: 'bowl', 57: 'pear',
                            5: 'bed', 60: 'plain', 91: 'trout', 3: 'bear', 58: 'pickup_truck', 16: 'can'}


    ########################################################################################################################

    label_df = pd.DataFrame(dataset_label, columns=[ "label", "label_p", "lam", "type"])
    list_init = choose_type(0,label_df)
    list_adver = choose_type(1,label_df)
    list_cutout = choose_type(2,label_df)
    list_mixup = choose_type(3,label_df)
    list_noise = choose_type(4,label_df)

    data_type = st.sidebar.selectbox("数据类型:", ["原始数据", "对抗数据", "cutout数据", "mixup数据", "噪声数据"])
    if data_type == "对抗数据":
        data_index = st.sidebar.selectbox("Index:", list_adver)

    elif data_type == "cutout数据":
        data_index = st.sidebar.selectbox("Index:", list_cutout)

    elif data_type == "mixup数据":
        data_index = st.sidebar.selectbox("Index:", list_mixup)

    elif data_type == "噪声数据":
        data_index = st.sidebar.selectbox("Index:", list_noise)

    else:
        data_index = st.sidebar.number_input("Index", min_value=0, max_value=50000)

    ########################################################################################################################

    topk = st.sidebar.slider('Topk:', 1, 10, 5)
    list_ccc = st.columns(topk)
    
    show = st.sidebar.slider('Epoch:', 1, 200, 20)

    data_index_label = dataset_label_name[int(label_df.loc[data_index]["label"])]
    data_index_label_p = dataset_label_name[int(label_df.loc[data_index]["label_p"])]
    data_index_label_lam = round(label_df.loc[data_index]["lam"],2) 

    st.sidebar.text(f"label:{data_index_label}")
    st.sidebar.text(f"pseudo:{data_index_label_p}")
    st.sidebar.text(f"lambda:{data_index_label_lam}")


    exits_data_flag = False
    if os.path.exists(choose_task):
        exits_data_flag = True
        
    else:
        st.sidebar.text("please wating load topk dataset...")
        get_behaviour_dataset(choose_task)

    if exits_data_flag:
        for epoch in range(0, show):
            sim_index, sim_score = show_epoch_sims(data_index, epoch, root_topk)
            sim_index_topk = sim_index[:topk]
            sim_score_topk = sim_score[:topk]
            for i, col in enumerate(list_ccc):
                with col:
                    sim_index = sim_index_topk[i]
                    score = round(sim_score_topk[i], 2)
                    label = dataset_label_name[int(label_df.loc[sim_index]["label"])]
                    img_one = to_pil(dataset_img[sim_index])
                    st.image(img_one, use_column_width=True)
                    st.write(pd.DataFrame({'label': [label],
                                        'score': [score],
                                        'index': [sim_index],
                                        'epoch': [epoch]}))
                                                    
