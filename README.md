# SOLIDER on [Person Re-identification]

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beyond-appearance-a-semantic-controllable/person-re-identification-on-msmt17)](https://paperswithcode.com/sota/person-re-identification-on-msmt17?p=beyond-appearance-a-semantic-controllable)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/beyond-appearance-a-semantic-controllable/person-re-identification-on-market-1501)](https://paperswithcode.com/sota/person-re-identification-on-market-1501?p=beyond-appearance-a-semantic-controllable)

This repo provides details about how to use [SOLIDER](https://github.com/tinyvision/SOLIDER) pretrained representation on person re-identification task.
We modify the code from [TransReID](https://github.com/damo-cv/TransReID), and you can refer to the original repo for more details.

## Installation and Datasets

We use python version 3.7, PyTorch version 1.7.1, CUDA 10.1 and torchvision 0.8.2. More details of installation and dataset preparation can be found in [TransReID-SSL](https://github.com/damo-cv/TransReID-SSL).

## Prepare Pre-trained Models 
You can download models from [SOLIDER](https://github.com/tinyvision/SOLIDER), or use [SOLIDER](https://github.com/tinyvision/SOLIDER) to train your own models.
Before training, you should convert the models first.

```bash
python convert_model.py path/to/SOLIDER/log/lup/swin_tiny/checkpoint.pth path/to/SOLIDER/log/lup/swin_tiny/checkpoint_tea.pth
```

## Training

We utilize 1 GPU for training. Please modify the `MODEL.PRETRAIN_PATH`, `DATASETS.ROOT_DIR` and `OUTPUT_DIR` in the config file.

```bash
sh run.sh
```

## Test

```bash
sh runtest.sh
```

## Performance

| Method | Model | MSMT17<br>(w/o RK) | Market1501<br>(w/o RK) | MSMT17<br>(with RK) | Market1501<br>(with RK) |
| ------ | :---: | :---: | :---: |
| SOLIDER | Swin Tiny | 67.4/85.9 | 91.6/96.1 | 81.5/89.2 | 95.3/96.6 |
| SOLIDER | Swin Small | 76.9/90.8 | 93.3/96.6 | 86.5/91.7 | 95.4/96.4 |
| SOLIDER | Swin Base | 77.1/90.7 | 93.9/96.9 | 86.5/91.7 | 95.6/96.7 |

- mAP/Rank1 are used as evaluation metric, RK indicates whether re-ranking is involved.
- We use the pretrained models from [SOLIDER](https://github.com/tinyvision/SOLIDER).
- The semantic weight is set to 0.2 in these experiments.

## Citation

If you find this code useful for your research, please cite our paper

```
@inproceedings{chen2023beyond,
  title={Beyond Appearance: a Semantic Controllable Self-Supervised Learning Framework for Human-Centric Visual Tasks},
  author={Weihua Chen and Xianzhe Xu and Jian Jia and Hao Luo and Yaohua Wang and Fan Wang and Rong Jin and Xiuyu Sun},
  booktitle={The IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023},
}
```
