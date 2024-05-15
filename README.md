# Split Estimation
This repo contains the code for calculating the angle of splits given an image of a gymnast. The repo implements the FastPose model from [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) and the [UniPose](https://github.com/bmartacho/UniPose) model (does not work lol). 

## Environment Requirements
I used a Python environment with CUDA 11.8 and Pytorch 2.1.0. 
```
python3 -m venv env

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install tqdm
pip install opencv-python
pip install numpy
```

## Data
There are 2 different datasets that were used in training the models: the [Halpe-FullBody](https://github.com/Fang-Haoshu/Halpe-FullBody) dataset and the [Leeds Sports Pose Extended (LSP)](https://github.com/axelcarlier/lsp) dataset. The data was structured as follows:
```
Split Estimation Repo
    |_ data
        |_ halpe
            |_ images
            |_ 	halpe_train_v1.json
        |_ lsp
            |_ images
            |_ 	joints.mat
    ...
```

## Pretrained Models
The FastPose model trained on the Halpe-FullBody dataset with 26 joint keypoints can be found [here](https://drive.google.com/file/d/1ek66o0_Ivk_9-KfD9qfdwohEnk5rK0B6/view?usp=sharing). The FastPose model trained on the LSP extended dataset can be found here [here](https://drive.google.com/file/d/1Q37oh2hLQsA_DXImmwq1herQ3xJJBAb4/view?usp=sharing). 