# MindSpore Temporal Relation Networks

## Introduction

This work is used for reproduce TRN based on NPU(Ascend 910)

**Temporal Relation Networks** is introduced in [arxiv](https://arxiv.org/pdf/1711.08496.pdf)


## Training

```
mpirun -n 8 python train.py > train.log 2>&1 &
```

## Evaluation 


```
python eval.py --config <config path>
```


## Acknowledgement

We heavily borrow the code from [ TRN-pytorch](https://github.com/zhoubolei/TRN-pytorch)
We thank the authors for the nicely organized code!
