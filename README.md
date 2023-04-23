# Learning Second-Order Attentive Context for Efficient Correspondence Pruning (AAAI2023)


## Requirements & Compilation

Required packages are listed in [requirements.txt](requirements.txt). 

The code is tested using Python-3.7.10 with PyTorch 1.7.1.

2. Compile extra modules

```shell script
cd utils/extend_utils
python build_extend_utils_cffi.py
```
According to your installation path of CUDA, you may need to revise the variables cuda_version in [build_extend_utils_cffi.py](utils/extend_utils/build_extend_utils_cffi.py).

## Datasets & Pretrain Models

1. Download the YFCC100M dataset and the SUN3D dataset from the [OANet](https://github.com/zjhthu/OANet) repository.

2. Download pretrained  models from [here]() 
3. Unzip and arrange all files like the following.
```
data/
    ├── model/
        ├── ANANet/
                ├── build_model.yaml
                └── model_best.yaml
        .....
        └── your model/
    ├── yfcc100m/
    ├── sun3d_test/
    └── pair/
 
```
It should be noted that if you have downloaded YFCC100M or SUN3D on another path, you can redefine  ''data_root'' in [pose_dataset.py](dataset/pose_dataset.py).
## Evaluation

Evaluate on the YFCC100M :
```shell script
python eval.py --name yfcc --cfg configs/eval/ANANet/yfcc.yaml
```

Evaluate on the SUN3D:
```shell script
python eval.py --name sun3d --cfg configs/eval/ANANet/sun3d.yaml
```



## Citation

```bibtex
@inproceedings{

}
```

## Acknowledgement

We have used codes from the following repositories, and we thank the authors for sharing their codes.



OANet: [https://github.com/zjhthu/OANet](https://github.com/zjhthu/OANet)

LMCNet:[https://github.com/liuyuan-pal/LMCNet](https://github.com/liuyuan-pal/LMCNet)

SuperGlue: [https://github.com/magicleap/SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork)

