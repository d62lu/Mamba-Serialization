# Mamba-Serialization
Exploring Token Serialization for Mamba-based LiDAR Point Cloud Segmentation

## Abtract

LiDAR point cloud segmentation has increasingly benefited from the application of Mamba-based models, offering efficient long-range dependency modeling with linear complexity. However, the unordered and irregular nature of point clouds
necessitates serialization, which significantly impacts the performance of Mamba-based methods. This paper explores the critical role of token serialization in Mamba-based LiDAR point cloud processing, using the pure Mamba network, PointMamba, as the baseline. We systematically investigated and analyzed existing point cloud serialization methods, evaluating their performance on two challenging LiDAR datasets: the airborne MultiSpectral LiDAR (MS-LiDAR) dataset and the aerial DALES dataset. To explore the inherent factors of serialization contributing to Mamba’s performance, we design novel indicators for serialization quality, focusing on spatial and semantic proximity. These indicators are validated across all datasets, offering a valuable reference and guidance for advancing token serialization in Mamba-based point cloud processing. Guided by these indicators, we proposed a new point cloud serialization method that integrates spatial and semantic features through a weighted comprehensive distance matrix. The proposed method achieves superior results on all LiDAR datasets, surpassing existing approaches, and establishes a strong foundation for advancing Mamba-based point cloud processing.


## Token Serialization

<img width="715" alt="1743347976442" src="https://github.com/user-attachments/assets/aaba247a-c3df-442f-93ac-c7addf3f93b7" />


## Install
The latest codes are tested on CUDA11.3 and above, PyTorch 1.10.1 and Python 3.9.
For mamba installation, please refer to PointMamba (https://github.com/LMD0311/PointMamba)


## Acknowledgement

We would like to express our sincere gratitude to PointMamba (https://github.com/LMD0311/PointMamba) for their valuable work on, which has significantly contributed to the development of this project.
