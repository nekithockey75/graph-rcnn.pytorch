# 1. graph-rcnn.pytorch

Pytorch code for our ECCV 2018 paper ["Graph R-CNN for Scene Graph Generation"](https://arxiv.org/pdf/1808.00191.pdf)

<div style="color:#0000FF" align="center">
<img src="figures/teaser_fig.png" width="850"/>
</div>

<!-- :balloon: 2019-06-04: Okaaay, time to reimplement Graph R-CNN on pytorch 1.0 and release a new benchmark for scene graph generation. It will also integrate other models like IMP, MSDN and Neural Motif Network. Stay tuned!

:balloon: 2019-06-16: Plan is a bit delayed by ICCV rebuttal, but still on track. Stay tuned! -->

## Introduction

This project is a set of reimplemented representative scene graph generation models based on Pytorch 1.0, including:

- [Graph R-CNN for Scene Graph Generation](https://arxiv.org/pdf/1808.00191.pdf), our own. ECCV 2018.
- [Scene Graph Generation by Iterative Message Passing](https://arxiv.org/pdf/1701.02426.pdf), Xu et al. CVPR 2017
- [Scene Graph Generation from Objects, Phrases and Region Captions](https://arxiv.org/pdf/1707.09700.pdf), Li et al. ICCV 2017
- [Neural Motifs: Scene Graph Parsing with Global Context](https://arxiv.org/pdf/1711.06640.pdf), Zellers et al. CVPR 2018
- [Graphical Contrastive Losses for Scene Graph Generation](https://arxiv.org/pdf/1903.02728.pdf), Zhang et al, CVPR 2019

Our reimplementations are based on the following repositories:

- [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
- [faster-rcnn](https://github.com/jwyang/faster-rcnn.pytorch)
- [scene-graph-TF-release](https://github.com/danfeiX/scene-graph-TF-release)
- [MSDN](https://github.com/yikang-li/MSDN)
- [neural-motifs](https://github.com/rowanz/neural-motifs)
- [Graphical Contrastive Losses](https://github.com/NVIDIA/ContrastiveLosses4VRD/tree/pytorch1_0)

## Why we need this repository?

The goal of gathering all these representative methods into a single repo is to establish a more fair comparison across different methods under the same settings. **As you may notice in recent literatures, the reported numbers for IMP, MSDN, Graph R-CNN and Neural Motifs are usually confusing, especially due to the big gap between IMP style methods (first three) and Neural Motifs-style methods (neural motifs paper and other variants built on it)**. We hope this repo can establish a good benchmark for various scene graph generation methods, and contribute to the research community!

## Checklist

- [x] Faster R-CNN Baseline (:balloon: 2019-07-04)
- [x] Scene Graph Generation Baseline (:balloon: 2019-07-06)
- [x] Iterative Message Passing (IMP) (:balloon: 2019-07-07)
- [x] Multi-level Scene Description Network (MSDN:no region caption) (:balloon: 2019-08-24)
- [x] Neural Motif (Frequency Prior Baseline) (:balloon: 2019-07-08)
- [x] Graph R-CNN (w/o relpn, GCNs) (:balloon: 2019-08-24)
- [x] Graph R-CNN (w relpn, GCNs) (:balloon: 2020-01-13)
- [ ] Graph R-CNN (w relpn, aGCNs)
- [ ] Neural Motif
- [ ] RelDN (Graphical Contrastive Losses)

## Benchmarking

### Object Detection

| source                                                                          | backbone | model        | bs  |  lr  | lr_decay | mAP@0.5 | mAP@0.50:0.95 |
| ------------------------------------------------------------------------------- | -------- | ------------ | :-: | :--: | :------: | :-----: | :-----------: |
| [this repo](https://drive.google.com/open?id=1THLvK8q2VRx6K3G7BGo0FCe-D0EWP9o1) | Res-101  | faster r-cnn |  6  | 5e-3 | 70k,90k  |  24.8   |     12.8      |

### Scene Graph Generation (Frequency Prior Only)

| source                                                                          | backbone     | model |  bs  |     lr     | lr_decay | sgdet@20 | sgdet@50 | sgdet@100 |
| ------------------------------------------------------------------------------- | ------------ | ----- | :--: | :--------: | :------: | :------: | :------: | :-------: |
| [this repo](https://drive.google.com/open?id=1Vb-gX3_OLhzgdNseXgS_2DiLmJ8qiG8P) | Res-101      | freq  |  6   |    5e-3    | 70k,90k  |   19.4   |   25.0   |   28.5    |
| [motifnet](https://github.com/rowanz/neural-motifs)                             | VGG-16       | freq  |  -   |     -      |    -     |   17.7   |   23.5   |   27.6    |
<!-- | Resnet-101                                                                 | freq-overlap | 6     | 5e-3 | (70k, 90k) |   100k   |    -     |    -     |   -       |-->

\* freq = frequency prior baseline

### Scene Graph Generation (Joint training)

| source                                                                               | backbone | model   | bs  |  lr  | lr_decay | sgdet@20 | sgdet@50 | sgdet@100 |         |
| ------------------------------------------------------------------------------------ | -------- | ------- | :-: | :--: | :------: | :------: | :------: | :-------: | ------- |
| [this repo](https://drive.google.com/open?id=1Vb-gX3_OLhzgdNseXgS_2DiLmJ8qiG8P)      | Res-101  | vanilla |  6  | 5e-3 | 70k,90k  |   10.4   |   14.3   |   16.8    |
| [this repo](https://drive.google.com/open?id=1Vb-gX3_OLhzgdNseXgS_2DiLmJ8qiG8P)      | Res-101  | freq    |  6  | 5e-3 | 70k,90k  |   100k   |   19.4   |   25.0    | 28.5    |

### Scene Graph Generation (Step training)

| source                                                                          | backbone | model   | bs  |  lr  | mAP@0.5 | sgdet@20 | sgdet@50 | sgdet@100 |
| ------------------------------------------------------------------------------- | -------- | ------- | :-: | :--: | :-----: | :------: | :------: | :-------: |
| [this repo](https://drive.google.com/open?id=1EoAVYsvOk5narI6iIoyOBb8jXDS7Retf) | Res-101  | vanilla |  8  | 5e-3 |  24.2   |   10.5   |   13.8   |   16.1    |
| [this repo](https://drive.google.com/open?id=1tAJAhrq8EP86PZJYfwsjtQ-k1RJt1p7q) | Res-101  | imp     |  8  | 5e-3 |  24.2   |   16.7   |   21.7   |   25.2    |
| [motifnet](https://github.com/rowanz/neural-motifs)                             | VGG-16   | imp     |  -  |  -   |    -    |   14.6   |   20.7   |   24.5    |
<!--|     this repo                                                                   | Res-101  | msdn    |  8  | 5e-3 | 20k,30k |    -     |    -     |     -     | -     |-->
<!--|     this repo                                                                   | Res-101  | grcnn   |  8  | 5e-3 | 20k,30k |    -     |    -     |     -     | -     |-->

\* you can click 'this repo' in above table to download the checkpoints.

The above table shows that our reimplementation of baseline and imp algorithm match the performance reported in mofitnet.

### Comparisons with other Methods

| model                                                                                    | bs  |  lr  | mAP@0.5 | sgdet@20 | sgdet@50 | sgdet@100 |
| ---------------------------------------------------------------------------------------- | :-: | :--: | :-----: | :------: | :------: | :-------: |
| [vanilla](https://drive.google.com/open?id=1EoAVYsvOk5narI6iIoyOBb8jXDS7Retf)            |  8  | 5e-3 |  24.2   |   10.5   |   13.8   |   16.1    |
| [imp](https://drive.google.com/open?id=1tAJAhrq8EP86PZJYfwsjtQ-k1RJt1p7q)                |  8  | 5e-3 |  24.2   |   16.7   |   21.7   |   25.2    |
| [msdn](https://drive.google.com/open?id=1xTl_o7D0oozDArjWl1p7g5jiH32jcdBS)               |  8  | 5e-3 |  24.2   |   18.3   |   23.6   | **27.1**  |
| [graph-rcnn](https://drive.google.com/open?id=1HSrJBK7qUxjvs-ccDlD7DKA7pW49hUcx)(no att) |  8  | 5e-3 |  24.2   | **18.8** | **23.7** |   26.2    |

\* you can click 'model' in above table to download the checkpoints.

Accordingly, all models achieved significantly better numbers compared with those reported in the original papers. The main reason for these consistant improvements are due to the per-class NMS of object proposals before sending to relationship head. Also, we found the gap between different methods are also reduced significantly. Our model has similar performance to msdn, while better performance than imp.

### Adding RelPN to other Methods

We added our RelPN to various algorithms and compared with the original version.

| model                                                                         | relpn | bs  |  lr  | mAP@0.5 | sgdet@20 | sgdet@50 | sgdet@100 |
| ----------------------------------------------------------------------------- | ----- | :-: | :--: | :-----: | :------: | :------: | :-------: |
| [vanilla](https://drive.google.com/open?id=1EoAVYsvOk5narI6iIoyOBb8jXDS7Retf) | no    |  8  | 5e-3 |  24.2   |   10.5   |   13.8   |   16.1    |
| [vanilla](https://drive.google.com/open?id=1Rp-cBVHGrmX993-oD05GuOXgLjhV_TNs) | yes   |  8  | 5e-3 |  24.2   |   12.3   |   15.8   |   17.7    |
| [imp](https://drive.google.com/open?id=1tAJAhrq8EP86PZJYfwsjtQ-k1RJt1p7q)     | no    |  8  | 5e-3 |  24.2   |   16.7   |   21.7   |   25.2    |
| [imp](https://drive.google.com/open?id=1XxQN3YRC5I8goTUO8rGVtlYdnofmSFZr)     | yes   |  8  | 5e-3 |  24.2   |   19.2   |   23.9   |   26.3    |
| [msdn](https://drive.google.com/open?id=1xTl_o7D0oozDArjWl1p7g5jiH32jcdBS)    | no    |  8  | 5e-3 |  24.2   |   18.3   |   23.6   |   27.1    |
| [msdn](https://drive.google.com/open?id=1ZMnyNotb-t5ty7vqX6IdQd7_S15iFIGg)    | yes   |  8  | 5e-3 |  24.2   |   19.2   |   23.8   |   26.2    |

\* you can click 'model' in above table to download the checkpoints.

Above, we can see consistant improvements for different algorithms, which demonstrates the effeciveness of our proposed relation proposal network (RelPN).

**Also, since much less object pairs (256, originally > 1k) are fed to relation head for predicate classification, the inference time for the models with RelPN is reduced significantly (~2.5 times faster)**

## Tips and Tricks

Some important observations based on the experiments:

- **Using per-category NMS is important!!!!**. We have found that the main reason for the huge gap between the imp-style models and motif-style models is that the later used the per-category nms before sending the graph into the scene graph generator. Will put the quantitative comparison here.

- **Different calculations for frequency prior result in differnt results\***. Even change a little bit to the calculation fo frequency prior, the performance of scene graph generation model vary much. In neural motiftnet, we found they turn on filter_non_overlap, filter_empty_rels to filter some triplets and images.

## Installation

### Prerequisites

- Python 3.6+
- Pytorch 1.0
- CUDA 8.0+

### Dependencies

Install all the python dependencies using pip:

```
pip install -r requirements.txt
```

and libraries using apt-get:

```
apt-get update
apt-get install libglib2.0-0
apt-get install libsm6
```

### Data Preparation

- Visual Genome benchmarking dataset:

| Annotations  | Object | Predicate |
| ------------ | ------ | --------- |
| \#Categories | 150    | 50        |

First, make a folder in the root folder:

```
mkdir -p datasets/vg_bm
```

Here, the suffix 'bm' is in short of "benchmark" representing the dataset for benchmarking. We may have other format of vg dataset in the future, e.g., more categories.

Then, download the data and preprocess the data according following this [repo](https://github.com/danfeiX/scene-graph-TF-release). Specifically, after downloading the [visual genome dataset](https://visualgenome.org/), you can follow this [guidelines](https://github.com/danfeiX/scene-graph-TF-release/tree/master/data_tools) to get the following files:

```
datasets/vg_bm/imdb_1024.h5
datasets/vg_bm/bbox_distribution.npy
datasets/vg_bm/proposals.h5
datasets/vg_bm/VG-SGG-dicts.json
datasets/vg_bm/VG-SGG.h5
```

The above files will provide all the data needed for training the object detection models and scene graph generation models listed above.

- Visual Genome bottom-up and top-down dataset:

| Annotations  | Object | Attribute | Predicate |
| ------------ | ------ | --------- | --------- |
| \#Categories | 1600   | 400       | 20        |

Soon, I will add this data loader to train [bottom-up and top-down model](https://arxiv.org/pdf/1707.07998.pdf) on more object/predicate/attribute categories.

- Visual Genome extreme dataset:

| Annotations  | Object | Attribute | Predicate |
| ------------ | ------ | --------- | --------- |
| \#Categories | 2500   | ~600      | ~400      |

This data loader further increase the number of categories for training more fine-grained visual representations.

### Compilation

Compile the cuda dependencies using the following commands:

```
cd lib/scene_parser/rcnn
python setup.py build develop
```

After that, you should see all the necessary components, including nms, roi_pool, roi_align are compiled successfully.

## Train

### Train object detection model:

- Faster r-cnn model with resnet-101 as backbone:

```
python main.py --config-file configs/faster_rcnn_res101.yaml
```

Multi-GPU training:

```
python -m torch.distributed.launch --nproc_per_node=$NGPUS main.py --config-file configs/faster_rcnn_res101.yaml
```

where NGPUS is the number of gpus available.

### Train scene graph generation model jointly (train detector and sgg as a whole):

- Vanilla scene graph generation model with resnet-101 as backbone:

```
python main.py --config-file configs/sgg_res101_joint.yaml --algorithm $ALGORITHM
```

Multi-GPU training:

```
python -m torch.distributed.launch --nproc_per_node=$NGPUS main.py --config-file configs/sgg_res101_joint.yaml --algorithm $ALGORITHM
```

where NGPUS is the number of gpus available. ALGORIHM is the scene graph generation model name.

### Train scene graph generation model stepwise (train detector first, and then sgg):

- Vanilla scene graph generation model with resnet-101 as backbone:

```
python main.py --config-file configs/sgg_res101_step.yaml --algorithm $ALGORITHM
```

Multi-GPU training:

```
python -m torch.distributed.launch --nproc_per_node=$NGPUS main.py --config-file configs/sgg_res101_step.yaml --algorithm $ALGORITHM
```

where NGPUS is the number of gpus available. ALGORIHM is the scene graph generation model name.

## Evaluate

### Evaluate object detection model:

- Faster r-cnn model with resnet-101 as backbone:

```
python main.py --config-file configs/faster_rcnn_res101.yaml --inference --resume $CHECKPOINT
```

where CHECKPOINT is the iteration number. By default it will evaluate the whole validation/test set. However, you can specify the number of inference images by appending the following argument:

```
--inference $YOUR_NUMBER
```

:warning: If you want to evaluate the model at your own path, just need to change the MODEL.WEIGHT_DET to your own path in faster_rcnn_res101.yaml.

### Evaluate scene graph frequency baseline model:

In this case, you do not need any sgg model checkpoints. To get the evaluation result, object detector is enough. Run the following command:

```
python main.py --config-file configs/sgg_res101_{joint/step}.yaml --inference --use_freq_prior
```

In the yaml file, please specify the path MODEL.WEIGHT_DET for your object detector.

### Evaluate scene graph generation model:

- Scene graph generation model with resnet-101 as backbone:

```
python main.py --config-file configs/sgg_res101_{joint/step}.yaml --inference --resume $CHECKPOINT --algorithm $ALGORITHM
```

- Scene graph generation model with resnet-101 as backbone and use frequency prior:

```
python main.py --config-file configs/sgg_res101_{joint/step}.yaml --inference --resume $CHECKPOINT --algorithm $ALGORITHM --use_freq_prior
```

Similarly you can also append the ''--inference $YOUR_NUMBER'' to perform partially evaluate.

:warning: If you want to evaluate the model at your own path, just need to change the MODEL.WEIGHT*SGG to your own path in sgg_res101*{joint/step}.yaml.

### Visualization

If you want to visualize some examples, you just simple append the command with:

```
--visualize
```

## Citation

    @inproceedings{yang2018graph,
        title={Graph r-cnn for scene graph generation},
        author={Yang, Jianwei and Lu, Jiasen and Lee, Stefan and Batra, Dhruv and Parikh, Devi},
        booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
        pages={670--685},
        year={2018}
    }

## Acknowledgement

We appreciate much the nicely organized code developed by [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). Our codebase is built mostly based on it.

# 2. Scene Graph Generation by Iterative Message Passing

![Scene Graph prediction samples](http://cs.stanford.edu/~danfei/scene-graph/preview.jpeg "Sample predictions")

## About this repository

This repository contains an implementation of the models introduced in the paper [Scene Graph Generation by Iterative Message Passing](https://arxiv.org/abs/1701.02426) by Xu et al. The model taks an image and a graph proposal as input and predicts the object and relationship categories in the graph. The network is implemented using [TensorFlow](https://www.tensorflow.org/) and the rest of the framework is in Python. Because the model is built directly on top of [Faster-RCNN by Ren et al](https://arxiv.org/abs/1506.01497), a substantial amount of data processing code is adapted from the [py-faster-rcnn repository](https://github.com/rbgirshick/py-faster-rcnn).

## Citing this work

If you find this work useful in your research, please consider citing:

```
@inproceedings{xu2017scenegraph,
  title={Scene Graph Generation by Iterative Message Passing},
  author={Xu, Danfei and Zhu, Yuke and Choy, Christopher and Fei-Fei, Li},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2017}
 }
```

## Project page

The project page is available at [http://cs.stanford.edu/~danfei/scene-graph/](http://cs.stanford.edu/~danfei/scene-graph/).

## Prerequisites

1. The framework does not include a regional proposal network implementation. A RoI proposal database pre-extracted using the py-faster-rcnn framework is available for download.
2. You need CUDA-compatible GPUs to run the framework. A CPU-compatible version will be released soon.
3. You need at least 320 GB of free space to store the processed VisualGenome image dataset. A training script that reads image files directly will be released in the future.
   However, if you just want to test/visualize some sample predictions, you may download a subset of the processed dataset (mini-vg) following the [instruction](data_tools/) or the "Quick Start" section. The subset takes ~4GB of space.

## Dependencies

To get started with the framework, install the following dependencies:

- Python 2.7
- [TensorFlow r0.11](https://www.tensorflow.org/versions/r0.11/get_started/os_setup#download-and-setup)
- [h5py](http://www.h5py.org/)
- [numpy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [scipy](https://www.scipy.org/)
- [pyyaml](https://pypi.python.org/pypi/PyYAML)
- [easydict](https://pypi.python.org/pypi/easydict/)
- [cython](http://cython.org/)
- [graphviz](https://pypi.python.org/pypi/graphviz) (optional, if you wish to visualize the graph structure)
- CUDA 7.5/8.0 (required, CPU-only version will be released soon)

1. It is recommended that you install everything in an [Anaconda](https://www.continuum.io/downloads) environment, i.e., install Anaconda and run

   ```
   conda create -n sence-graph python=2.7
   source activate scene-graph
   ```

2. Run `pip install -r requirements.txt` to install all the requirement except TensorFlow and CUDA. Follow the provided [URL](https://www.tensorflow.org/versions/r0.11/get_started/os_setup#download-and-setup) to install TensorFlow r0.11 (0.10 and 0.12 also works).

The code has not been tested on TensorFlow 1.0 and above, but may potentially work once you convert all TF-related code using the offical transition [script](https://www.tensorflow.org/install/migration).

## Compile libraries

1. After you have installed all the dependencies, run the following command to compile nms and bbox libraries.

```
cd lib
make
```

2. Follow this [this instruction](lib/roi_pooling_layer/) to see if you can use the pre-compiled roi-pooling custom op or have to compile the op by yourself.

## Quick Start

1. Make sure you have installed all the dependencies and compiled the libraries.
2. Run the `download.sh` script to download the mini-vg dataset and a model checkpoint.

   `./download.sh`

3. Run the following command to visualize a predicted scene graph. Set `GPU_ID` to the ID of the GPU you want to use, e.g. `0`.

   ```
   ./experiments/scripts/test.sh mini-vg -1 \
                                 dual_graph_vrd_final 2 \
                                 checkpoints/dual_graph_vrd_final_iter2.ckpt \
                                 viz_cls \
                                 GPU_ID
   ```

## Dataset

The scene graph dataset used in the paper is the [VisualGenome dataset](https://visualgenome.org), although the framework can work with any scene graph dataset if converted to the desired format. Please refer to the [dataset README](data_tools/) for further instructions on converting the VG dataset into the desired format or downloading pre-processed datasets.

## Train a model

Follow the following steps to train a model:

1. Prepare or download the [full dataset](data_tools/).
2. Download a [Faster-RCNN model](http://cvgl.stanford.edu/scene-graph/dataset/coco_vgg16_faster_rcnn_final.npy) pretrained on the MS-COCO dataset and save the model to `data/pretrained/`.
3. Edit the training script `experiments/scripts/train.sh` such that all paths agree with the files on your file system.
4. To train the final model with inference iterations = 2, run:

   `./experiments/scripts/train.sh dual_graph_vrd_final 2 CHECKPOINT_DIRECTORY GPU_ID`

The program saves a checkpoint to `checkpoints/CHECKPOINT_DIRECTORY/` every 50000 iterations. Training a full model on a desktop with Intel i7 CPU, 64GB memory, and a TitanX graphics card takes around 20 hours. You may use tensorboard to visualize the training process. By default, the tf log directory is set to `checkpoints/CHECKPOINT_DIRECTORY/tf_logs/`.

## Evaluate a model

Follow the following steps to evaluate a model:

1. Prepare or download [the full dataset or the mini-vg dataset](data_tools/).
2. If you wish to evaluate a pre-trained model, first download a checkpoint in the "Checkpoints" section.
3. Edit the evaluation script `experiments/scripts/test.sh` such that all paths agree with the files on your file system.
4. To evaluate the final model with inference iterations = 2 using 100 images in the test set of the full VG dataset (use mini-vg for the [mini VG dataset](data_tools/)), run

   `./experiments/scripts/test.sh vg 100 dual_graph_vrd_final 2 CHECKPOINT_PATH(.ckpt) all GPU_ID`

Note that to reproduce the results as presented the paper, you have to evaluate the entire test set by setting the number of images to -1.
The evaluation process takes around 10 hours. Setting the evaluation mode to `all` is to evaluate the models on all three tasks, i.e., `pred_cls, sg_cls, sg_det`.
You can also set the evaluation mode to individual tasks.

## Visualize a predicted scene graph

Follow the following steps to visualize a scene graph predicted by the model:

1. Prepare or download [the full dataset or the mini-vg dataset](data_tools/).
2. If you wish to evaluate a pre-trained model, first download a checkpoint in the "Checkpoints" section.
3. Edit the evaluation script `experiments/scripts/test.sh` such that all paths agree with the files on your file system.
4. To visualize the predicted graph of the first 100 images in the test set of the full VG dataset (use mini-vg for the [mini VG dataset](data_tools/)), run

   `./experiments/scripts/test.sh vg 100 dual_graph_vrd_final 2 CHECKPOINT_PATH(.ckpt) viz_cls GPU_ID`

The `viz_cls` mode assumes ground truth bounding boxes and predicts the predicted object and relationship labels, which is of the same setting as the `sg_cls` task.
`viz_det` mode uses the proposed bounding box from the regional proposal network as the object proposals, which is of the same setting as the `sg_det` task.

## Checkpoints

A TensorFlow checkpoint of the final model trained with 2 inference iterations:

[dual_graph_vrd_final_iter2_checkpoint.zip](http://cvgl.stanford.edu/scene-graph/checkpoints/dual_graph_vrd_final_iter2_checkpoint.zip)

## License

MIT License
