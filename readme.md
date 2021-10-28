# Semantic Segmentation using UNET
This is the implementation of UNET on <a href= "https://www.kaggle.com/c/carvana-image-masking-challenge" > Carvana Image Masking Kaggle Challenge  </a>

## About the Dataset
This dataset contains a large number of car images (as .jpg files). Each car has exactly 16 images, each one taken at different angles.

For the training set, you are provided a .gif file that contains the manually cutout mask for each image. 
Link to download the dataset: <a href="https://www.kaggle.com/c/carvana-image-masking-challenge"> Here </a>


## UNET Architecture
![alt text](https://github.com/anushka17agarwal/Segmentation_UNET/blob/main/images/unet.png)


The UNET CNN architecture may be divided into the *Encoder*, *Bottleneck* and *Decoder* blocks, followed by a final segmentation output layer. 

- Encoder: There are 4 Encoder blocks, each consisting of a convolutional block followed by a Spatial Max Pooling layer. 
- Bottleneck: The Bottleneck consists of a single convolutional block.
- Decoder: There are 4 Decoder blocks, each consisting of a deconvolution operation, followed by a convolutional block, along with skip connections.

**Note**: The *convolutional block* consists of 2 conv2d operations each followed by a BatchNorm2d, finally followed by a ReLU activation.


## Implementation Details
- Image preprocessing included augmentations like HorizontalFlip, VerticalFlip, Rotate.
- Dataloader object was created for both training and test data
- Training process was carried out for 10 epochs, using the Adam Optimizer with a Learning Rate 1e-4.
- Validation was carried out using Dice Loss and Intersection over Union Loss.


## Installation and Quick Start
To use the repo and run inferences, please follow the guidelines below

- Cloning the Repository: 

        $ git clone https://github.com/anushka17agarwal/Segmentation_UNET
        
- Entering the directory: 

        $ cd unet/
        
- Setting up the Python Environment with dependencies:

        $ pip install -r requirements.txt

- Running the file for inference:

        $ python3 train.py



## Accuracy Scores
- Got 38967548.0/39091200 with acc 99.68Dice score: 0.9924604296684265