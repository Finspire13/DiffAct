# [DiffAct](https://finspire13.github.io/DiffAct-Project-Page/)

Code for [Diffusion Action Segmentation](https://arxiv.org/abs/2303.17959) (ICCV 2023).



## Setup

* Recommended Environment: Python 3.9.2, Cuda 11.4, PyTorch 1.10.0
* Install dependencies: `pip3 install -r requirements.txt`

## Data

* Download features of 50salads, GTEA and Breakfast provided by [MS-TCN]() and [ASFormer](https://github.com/ChinaYi/ASFormer): [[Link1]](https://mega.nz/#!O6wXlSTS!wcEoDT4Ctq5HRq_hV-aWeVF1_JB3cacQBQqOLjCIbc8) [[Link2]](https://zenodo.org/record/3625992#.Xiv9jGhKhPY)
* Unzip the data, rename it to "datasets" and put into the current directory
```
DiffAct/
├── datasets
│   ├── 50salads
│   │   ├── features
│   │   ├── groundTruth
│   │   ├── mapping.txt
│   │   └── splits
│   ├── breakfast
│   │   ├── features
│   │   ├── groundTruth
│   │   ├── mapping.txt
│   │   └── splits
│   └── gtea
│       ├── features
│       ├── groundTruth
│       ├── mapping.txt
│       └── splits
├── main.py
├── model.py
└── ...
```

## Run

* Generate config files by `python3 default_configs.py`
* Simply run `python3 main.py --config configs/some_config.json --device gpu_id`
* Trained models and logs will be saved in the `result` folder

## Trained Models 

* We provide some trained models in the `trained_models` folder

## Citation

```
@inproceedings{liu2023diffusion,
  title={Diffusion Action Segmentation},
  author={Liu, Daochang and Li, Qiyue and Dinh, Anh-Dung and Jiang, Tingting and Shah, Mubarak and Xu, Chang},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2023}
}
```

## License
MIT
