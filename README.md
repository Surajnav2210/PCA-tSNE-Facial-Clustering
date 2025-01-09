# PCA and t-SNE Based Facial Analysis and Clustering

### Overview
This project focuses on analyzing and clustering facial images using **PCA** for dimensionality reduction and **t-SNE** for visualization. The goal is to compress high-dimensional data while retaining key information and uncover patterns in the reduced data.

### Features
- Reduced 1024-dimensional facial data (2414 images) with **PCA**, retaining **90% variance**.
- Achieved reconstruction quality with:
  - **PSNR**: 20.01 dB
  - **SSIM**: 0.76
- Visualized facial patterns using **t-SNE** and grouped data into clusters with **K-Means**.

### Tools Used
- **Languages**: Python
- **Libraries**: NumPy, SciPy, Matplotlib, scikit-learn, scikit-image

### Results
- Dimensionality reduced by **80%** while retaining facial features.
- Patterns visualized with **t-SNE**; moderate clustering achieved with a silhouette score of **0.26**.

### Dataset
- **YaleB Dataset**: 2414 grayscale facial images, each 32x32 pixels.
