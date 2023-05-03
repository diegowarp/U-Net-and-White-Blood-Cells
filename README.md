# U-Net and White Blood Cells
Application of the U-Net convolutional neural network in the segmentation and classification of regions in white blood cells using the Raabin WBC Data dataset.

Using the PyTorch language, the architecture of the original U-Net was developed based on the article "U-Net: Convolutional Networks for Biomedical Image Segmentation" by Olaf Ronneberger et al (2015), according to Figure 1 presented in this article.

<img src="https://user-images.githubusercontent.com/60277333/235810738-ae43a82d-5e79-49b4-8ae5-ba8d52e7a258.png" width="600" height="500">

<b>Composition of the dataset used:</b>

- Total of 1145 images;
- 218 images of basophil cells;
- 201 images of eosinophil cells;
- 242 images of lymphocyte cells;
- 242 images of monocyte cells;
- 242 images of neutrophil cells.

Dataset split: 85% for training and 15% for validation (there was no test).

The training environment was on Google Colab, using the free resources available and the possibility of allocation in the GPU.

<b>Training parameters:</b>

- 30 epochs;
- Batch size: 5;
- Learning Rate: 0.0001;
- Optimizer: optm.Adam, in PyTorch;
- Loss function: nn.BCEWithLogitsLoss, in PyTorch;
- Data Augmentation: Rotation, Horizontal Flip, Vertical Flip, and Perspective.

The result of the training with 30 epochs can be observed in the following graph:

<img src="https://user-images.githubusercontent.com/60277333/235813197-c635b9e1-941f-41ae-9e79-82f3e4814757.png" width="600" height="350">

As metrics to evaluate the performance of the network, Accuracy and Dice Score were used, <b>reaching 99.34% of Accuracy and 97.16% of Dice Score in the end.</b>

All the resulting images from the training can be accessed in the "results" folder, but here are the results for eosinophil cells:

<img src="https://user-images.githubusercontent.com/60277333/235814793-e1708cc4-34ec-42cd-88c2-842a6973bc90.jpg" width="600" height="1000">

Hope you enjoy it! 

Diego Oliveira
