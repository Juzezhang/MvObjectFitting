# MvObjectFitting

Welcome to the MvObjectFitting repository, a comprehensive toolkit designed for object fitting from multi-view images.

## Updates
- **April 2, 2024**: The source code has been fully uploaded!

## Environment Setup

To set up your environment, follow these steps:

```bash
conda create -n MvObjectFitting python=3.8 -y
conda activate MvObjectFitting
conda install --file conda_install_cuda117_package.txt -c nvidia
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
git clone https://github.com/JiangWenPL/multiperson.git && cd multiperson/neural_renderer
python setup.py install
```
Note: If you are using the latest versions of PyTorch, you may encounter an error related to the use of "AT_CHECK". To resolve this issue, you should replace instances of AT_CHECK with TORCH_CHECK in your code.

## Data Structure


Organize your dataset as follows:

```
├─ Path_of_Folder
    ├─ calibration.json        # camera intrinsics and world-to-cam extrinsics
    ├─ object_id.txt
    ├─ mask
        ├─ 0
            ├─ 000000.jpg
            ├─ 000001.jpg
            ├─ 000003.jpg
                ...
            ...
    ├─ videos
        ├─ data1.mp4
        ├─ data2.mp4
        ...
        ├─ data76.mp4
    ├─ Object Template
        ├─ object_name_1.obj
        ├─ object_name_2.obj
        ...
    ...
```

## Single Person and Single Object Fitting (SPSO)


```bash
python object_fitting_256_spso.py
```

## Multiple Person and Multiple Object Fitting (MPMO)


```bash
python object_fitting_256_mpmo.py
```

## Licenses
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.