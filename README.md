# PyTorch-Deformable-Convolution-v2
Don't feel pain to use Deformable Convolution v2(DCNv2)

![](offset_visualization.gif)

If you are curious about how to visualize offset(red point), refer to [offset_visualization.py](./offset_visualization.py)

# Usage

```python

from dcn import DeformableConv2d

class Model(nn.Module):
    ...
    self.conv = DeformableConv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
    ...

```

# Experiment

You can simply reproduce the results of my experiment on Google Colab.

Refer to experiment.ipynb!

## Task

**Scaled-MNIST** Handwritten Digit Classification

## Model

Simple CNN Model including 5 conv layers

```python
class MNISTClassifier(nn.Module):
    def __init__(self,
                 deformable=False):

        super(MNISTClassifier, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)   
        conv = nn.Conv2d if deformable==False else DeformableConv2d
        self.conv4 = conv(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = conv(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x) # [14, 14]
        x = torch.relu(self.conv2(x))
        x = self.pool(x) # [7, 7]
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = self.gap(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x
```

## Training

- Optimizer: Adam
- Learning Rate: 1e-3
- Learning Rate Scheduler: StepLR(step_size=1, gamma=0.7)
- Batch Size: 64
- Epochs: 14
- Augmentation: **NONE**

## Test

In the [paper](https://arxiv.org/abs/1811.11168), authors mentioned that the network's ability to model geometric transformation with DCNv2 is considerably enhanced.

I verified it with scale augmentation.

All images in the test set of MNIST dataset are augmented by scale augmentation(x0.5, x0.6, ..., x1.4, x1.5).


### Results

|Model|Top-1 Accuracy(%)|
|---|---|
|w/o DCNv2|90.03%|
|**w/ DCNv2**|**92.90%**|

## References

[mxnet implementation](https://github.com/apache/incubator-mxnet/blob/5722f8b38af58c5a296e46ca695bfaf7cff85040/python/mxnet/gluon/nn/conv_layers.py#L1447-L1631
)

## To Do Lists

- [ ] Support Onnx Conversion
