# Simple-PyTorch-Deformable-Convolution-v2
Don't feel fain to use Deformable Convolution v2(DCNv2)

# Usage

```python

from dcn import DeformableConv2d

class Model(nn.Module):
    ...
    self.conv = DeformableConv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
    ...

```

# Experiment

You can simply reproduce the results of the my experiment on Google Colab.

Refer to .ipynb file!

## Task

**Scaled-MNIST** Handwritten Digit Classification

## Model

Simple CNN Model that the number of conv layers is 5.

```python
class MNISTClassifier(nn.Module):
    def __init__(self,
                 deformable=False):

        super(MNISTClassifier, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True)   
        conv = nn.Conv2d if deformable==False else DeformableConv2d
        self.conv4 = conv(16, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = conv(16, 16, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 10)
        
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
- Augmentation: **X**

## Test

All images in the test set of MNIST dataset are augmented by scale augmentation(x0.5, x0.6, ..., x1.4, x1.5).

The scale augmentation is used to verify that DCNv2 is robust to scale variation.

### Results










