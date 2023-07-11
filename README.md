# ESTrack: Efficient Siamese Tracker


+ Green, red, blue, yellow  -> Groundtruth, ESTrack-repvgg, SiamAPN++, TCTrack

![ESTrack](/home/zxh/ESTrack.gif)

## Getting started

+ Install

  + create and activate a conda environment

    ```
    conda create -n estrack python=3.7
    conda activate estrack
    ```

  + intall pytorch

    ```
    pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
    ```

  + install other packages

    ```
    conda install matplotlib pandas tqdm
    pip3 install opencv-python visdom scikit-image tikzplotlib gdown
    conda install cython scipy
    sudo apt-get install libturbojpeg
    pip3 install pycocotools jpeg4py
    pip3 install wget yacs
    pip3 install shapely==1.6.4.post2
    pip3 install tensorboardX
    ```

  + clone the repo

    ``` 
    git clone https://github.com/zhanghan0112/ESTrack.git
    ```

## Quick start

+ Training

  + Modify [local.py](https://github.com/zhanghan0112/ESTrack/blob/main/ltr/admin/local.py) to set the paths to training datasets, results paths etc.

  + Running the following commands to train the ESTrack with DDP mode.

    ```
    conda activate estrack
    cd ESTrack/ltr
    python -m torch.distributed.launch --nproc_per_node 2 run_training_ddp.py ESTrack estrack_ddp
    ```

+ Evaluation

  + We integrated [PySOT](https://github.com/zhanghan0112/ESTrack/tree/main/pysot_toolkit) for evaluation. You need to specify the path of the model and dataset in the [test_one_param.py](https://github.com/zhanghan0112/ESTrack/blob/main/pysot_toolkit/test_one_param.py) or the [test.py](https://github.com/zhanghan0112/ESTrack/blob/main/pysot_toolkit/test.py) for evaluation.
  + You can also use [pytracking](https://github.com/zhanghan0112/ESTrack/tree/main/pytracking) to test and evaluate tracker. The results might be slightly different with [PySOT](https://github.com/STVIR/pysot) due to the slight difference in implementation (pytracking saves  results as integers, pysot toolkit saves the results as decimals).

## Acknowledgement

This is a modified version of the python framework [PyTracking](https://github.com/visionml/pytracking) based on **Pytorch**, also borrowing from [PySOT](https://github.com/STVIR/pysot) and [TransT](https://github.com/chenxin-dlut/TransT). We would like to thank their authors for providing great frameworks and toolkits.

## Contact

+ Xiaohan Zhang (email: zxiaohan@mail.dlut.edu.cn)

  Feel free to contact me if you have additional questions.

## 
