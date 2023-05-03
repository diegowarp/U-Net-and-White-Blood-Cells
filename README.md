# U-Net and White Blood Cells
Application of the U-Net convolutional neural network in the segmentation and classification of regions in white blood cells using the Raabin WBC Data dataset.

Using the PyTorch language, the architecture of the original U-Net was developed based on the article "U-Net: Convolutional Networks for Biomedical Image Segmentation" by Olaf Ronneberger et al (2015), according to Figure 1 presented in this article.

<img src="https://user-images.githubusercontent.com/60277333/235810738-ae43a82d-5e79-49b4-8ae5-ba8d52e7a258.png" width="600" height="500">

Composition of the dataset used:

- Total of 1145 images;
- 218 images of basophil cells;
- 201 images of eosinophil cells;
- 242 images of lymphocyte cells;
- 242 images of monocyte cells;
- 242 images of neutrophil cells.

Dataset split: 85% for training and 15% for validation (there was no test).

