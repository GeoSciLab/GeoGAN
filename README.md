# GeoGAN
PyTorch implementation of TemDeep: Adversarial Augmented Dataset for Refined Daily Prediction of Sea Surface Temperature Fields

### Training GeoGAN:
Simply run the following to train an encoder-decoder network using GeoGAN on the your dataset:
```
python main.py 
```

### Distributed Training
With distributed data parallel (DDP) training:
```
CUDA_VISIBLE_DEVICES=0 python main.py --nodes 2 --nr 0
CUDA_VISIBLE_DEVICES=1 python main.py --nodes 2 --nr 1
CUDA_VISIBLE_DEVICES=2 python main.py --nodes 2 --nr 2
CUDA_VISIBLE_DEVICES=N python main.py --nodes 2 --nr 3
```

## What is GeoGAN?
In the marine sciences, the precision of oceanographic data analysis directly impacts our understanding and prediction of marine ecosystems, climate change effects, and weather phenomena. A significant challenge in this domain is the limited availability and uneven distribution of sea surface temperature measurements, which are crucial for accurate climate modeling. This scarcity necessitates effective data augmentation techniques to enhance dataset quality and coverage. Current methods for augmenting geophysical data, particularly for marine applications, are notably lacking, leaving a gap in our ability to refine predictions and analyses. To address this, our study introduces a novel approach using a Generative Adversarial Network (GAN), where the generator employs depthwise separable convolutions alongside U-Net architecture for efficient data generation, and the discriminator is enhanced with residual attention mechanisms for precise validation. Incorporating Mean Absolute Error (MAE) as a regularization term alongside the traditional GAN loss ensures the generated sea surface temperature (*sst*) fields are distinct yet realistic compared to the original ERA5 daily *sst* data. Evaluating our method through predictive performance on augmented datasets, we observed a significant improvement, with a 34.48% reduction in MAE.

<p align="center">
  <img src="./pics/framework.jpg" width="1000"/>
</p>



## Usage
Simply run for single GPU or CPU training:
```
python main.py
```

For distributed training (DDP), use for every process in nodes, in which N is the GPU number you would like to dedicate the process to:
```
CUDA_VISIBLE_DEVICES=0 python main.py --nodes 2 --nr 0
CUDA_VISIBLE_DEVICES=1 python main.py --nodes 2 --nr 1
CUDA_VISIBLE_DEVICES=2 python main.py --nodes 2 --nr 2
CUDA_VISIBLE_DEVICES=N python main.py --nodes 2 --nr 3
```

`--nr` corresponds to the process number of the N nodes we make available for training.

### Testing
To test a trained model, make sure to set the `model_path` variable in the `config/config.yaml` to the log ID of the training (e.g. `logs/0`).
Set the `epoch_num` to the epoch number you want to load the checkpoints from (e.g. `4000`).

```
python eval.py
```

or in place:
```
python eval.py --model_path=./save --epoch_num=4000
```

## Logging and TensorBoard
To view results in TensorBoard, run:
```
tensorboard --logdir runs
```

## Environment

  - Python >= 3.6
  - PyTorch, tested on 1.9, but should be fine when >=1.6

## Citation

If you find our code or datasets helpful, please consider citing our related works.

## Contact

If you have questions or suggestions, please open an issue here or send an email to xmcao508@126.com.
