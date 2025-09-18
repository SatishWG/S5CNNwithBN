# MNIST Classification with CNN and Batch Normalization

This project implements a Convolutional Neural Network (CNN) for MNIST digit classification using PyTorch. The model incorporates batch normalization and dropout for better training stability and generalization.

## Model Architecture

The network consists of three main convolution blocks with the following layer-wise details:

### Layer Details

| Layer | Operation | Input Shape | Output Shape | RF |
|-------|-----------|-------------|--------------|-----|
| **First Convolution Block** |
| Conv1 | Conv2d(1, 8, 3) | 28x28x1 | 26x26x8 | 3x3 |
| Conv2 | Conv2d(8, 16, 3) | 26x26x8 | 24x24x16 | 5x5 |
| Pool1 | MaxPool2d(2, 2) | 24x24x16 | 12x12x16 | 6x6 |
| **Second Convolution Block** |
| Conv3 | Conv2d(16, 32, 1) | 12x12x16 | 12x12x32 | 6x6 |
| Conv4 | Conv2d(32, 16, 3) | 12x12x32 | 10x10x16 | 10x10 |
| Conv5 | Conv2d(16, 8, 3) | 10x10x16 | 8x8x8 | 14x14 |
| **Third Convolution Block** |
| Conv6 | Conv2d(8, 16, 1) | 8x8x8 | 8x8x16 | 14x14 |
| Conv7 | Conv2d(16, 8, 3) | 8x8x16 | 6x6x8 | 18x18 |
| **Output** |
| FC | Linear | 8*6*6 | 10 | - |

### Architecture Highlights:
1. **First Block**: Initial feature extraction and dimensionality reduction
   - Uses two 3x3 convolutions followed by maxpooling
   - Batch normalization after each convolution
   - Dropout(0.25) for regularization

2. **Second Block**: Feature transformation
   - Uses 1x1 convolution for channel manipulation
   - Two 3x3 convolutions for feature extraction
   - Each conv layer followed by BatchNorm

3. **Third Block**: Final feature processing
   - 1x1 convolution for channel adjustment
   - Final 3x3 convolution before classification
   - Maintains consistent dropout pattern

## Training Details

- **Optimizer**: SGD with momentum (lr=0.001, momentum=0.9)
- **Loss Function**: Negative Log Likelihood Loss
- **Epochs**: 20
- **Batch Size**: 64
- **Data Augmentation**:
  - Random Center Crop (22x22)
  - Resize to 28x28
  - Random Rotation (-15° to 15°)
  - Normalization (mean=0.1307, std=0.3081)

## Results

The model achieves:
- Training Accuracy: ~98%
- Test Accuracy: ~99%

### Performance Plots

![Training and Testing Metrics](https://raw.githubusercontent.com/yourusername/ERA_V4/main/S5CNN_Backprop/Assignment5/results.png)

The plots demonstrate:
1. **Training Loss (Top Left)**: Shows consistent decrease indicating good optimization
2. **Test Loss (Top Right)**: Follows training loss closely, suggesting good generalization
3. **Training Accuracy (Bottom Left)**: Steady improvement reaching ~98%
4. **Test Accuracy (Bottom Right)**: Stabilizes around 99%, indicating good model performance

## Key Features

1. Batch Normalization after each convolution layer
2. Strategic use of 1x1 convolutions for channel dimension management
3. Dropout layers (0.25) for regularization
4. Effective data augmentation pipeline
5. Balanced architecture with gradual reduction in spatial dimensions

## Dependencies

- PyTorch
- torchvision
- tqdm
- matplotlib
- torchsummary

## Usage

```bash
python ERAV4S5CNNwithBN.py
```

## Model Summary
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 8, 26, 26]              72
              ReLU-2           [-1, 8, 26, 26]               0
       BatchNorm2d-3           [-1, 8, 26, 26]              16
            Conv2d-4          [-1, 16, 24, 24]           1,152
              ReLU-5          [-1, 16, 24, 24]               0
       BatchNorm2d-6          [-1, 16, 24, 24]              32
         MaxPool2d-7          [-1, 16, 12, 12]               0
           Dropout-8          [-1, 16, 12, 12]               0
            Conv2d-9          [-1, 32, 12, 12]             512
             ReLU-10          [-1, 32, 12, 12]               0
      BatchNorm2d-11          [-1, 32, 12, 12]              64
           Conv2d-12          [-1, 16, 10, 10]           4,608
             ReLU-13          [-1, 16, 10, 10]               0
      BatchNorm2d-14          [-1, 16, 10, 10]              32
           Conv2d-15            [-1, 8, 8, 8]           1,152
             ReLU-16            [-1, 8, 8, 8]               0
      BatchNorm2d-17            [-1, 8, 8, 8]              16
          Dropout-18            [-1, 8, 8, 8]               0
           Conv2d-19           [-1, 16, 8, 8]             128
             ReLU-20           [-1, 16, 8, 8]               0
      BatchNorm2d-21           [-1, 16, 8, 8]              32
          Dropout-22           [-1, 16, 8, 8]               0
           Conv2d-23            [-1, 8, 6, 6]           1,152
             ReLU-24            [-1, 8, 6, 6]               0
      BatchNorm2d-25            [-1, 8, 6, 6]              16
          Dropout-26            [-1, 8, 6, 6]               0
           Linear-27                   [-1, 10]           2,890
================================================================
Total params: 11,874
Trainable params: 11,874
Non-trainable params: 0
----------------------------------------------------------------
```