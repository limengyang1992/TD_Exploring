# Official PyTorch implementation for "Delving into the Training Dynamics  for Image Classification". 

![](figures/logo.png)

## 0. Requirements

- Python 3.6+
- torch=1.8.0+cu111
- torchvision+0.9.0+cu111
- tqdm=4.26.0
- PyYAML=6.0
- einops
- torchsummary


## 1. Implements

### 1.1 Models

vision Transformer:

| Model              | GPU Mem | Top1:train | Top1:val | weight:M |
| ------------------ | ------- | ---------- | -------- | -------- |
| VGG16           | 2869M   | 68.96      | 69.02    | 47.6     |
| ResNet18        | 2009M   | 98.83      | 92.50    | 19.2     |
| ResNet34        | 1681M   | 98.22      | 91.77    | 7.78     |
| DenseNet        | 1175M   | 96.40      | 90.17    | 4.0      |


## 2. Task Training

```
bash run_task.sh
```

## 3. TD Extraction

```
bash run_td.sh
```



