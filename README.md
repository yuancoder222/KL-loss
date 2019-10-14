# kl-loss

This is only the caffe implementation for the KL loss in the paper:

Bounding Box Regression with Uncertainty for Accurate Object Detection

@inproceedings{klloss,
  title={Bounding Box Regression with Uncertainty for Accurate Object Detection},
  author={He, Yihui and Zhu, Chenchen and Wang, Jianren and Savvides, Marios and Zhang, Xiangyu },
  booktitle={2019 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019},
  organization={IEEE}
}

### How to write the prototxt?

if you have three inputs (not include weights)

```
layer {
  name: "loss_kl"
  type: "KlLoss"
  bottom: "coor"
  bottom: "theta"
  bottom: "label"
  top: "kl_loss"
  propagate_down: 1
  propagate_down: 1
  propagate_down: 0
  loss_weight: 1
}
```


if you have four inputs (include weights)

```
layer {
  name: "loss_kl"
  type: "KlLoss"
  bottom: "coor"
  bottom: "theta"
  bottom: "label"
  bottom: "weight"
  top: "kl_loss"
  propagate_down: true
  propagate_down: true
  propagate_down: false
  propagate_down: false
  include {
    phase: TRAIN
  }
}

```
