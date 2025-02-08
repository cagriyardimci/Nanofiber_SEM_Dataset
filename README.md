Nanofiber Classifier: Automated Defect Detection in Electrospun PAN Nanofibers
Authors: Cagri YARDIMCI - Mevlut ERSOY
Year: 2025
Journal: The Visual Computer

Project Overview
This repository contains the source code and dataset necessary to replicate the results of the research paper:
"Automated Defect Classification of Electrospun Polyacrylonitrile Nanofibers via Deep Learning of SEM Images"

Abstract
This study presents a deep learning-based framework for automated defect detection in electrospun Polyacrylonitrile (PAN) nanofibers using Scanning Electron Microscope (SEM) images. The proposed Nanofiber Classifier model is trained on a dataset consisting of three categories:

Slightly Defective
Defective
Non-Defective

Dataset
The dataset used in this study is publicly available:
📂 Nanofiber SEM Dataset: Figshare Repository

The dataset consists of 2 main folders and 3 subfolders: Training_Dataset and Test_Dataset with 3 categories mentioned above.

The images were preprocessed using a bilateral filter, followed by data augmentation techniques.

Requirements
To replicate the study, install the required dependencies:

Matlab

Matlab - Deep Network Designer

Usage
Training the Model

Run the following code in MATLAB;
Nanofiber_Classifier.m

Testing the Model

Run the following code in MATLAB;
Test_Classifier.m

Performance Metrics;

sensitivity,
specificity,
accuracy,
precision,
f_measure,

Citation
If you use this repository in your research, please cite the original article:

Cagri Yardimci et al. (2025), Automated Defect Classification of Electrospun Polyacrylonitrile Nanofibers via Deep Learning of SEM Images, The Visual Computer.

Data citations should include a persistent identifier (such as a DOI), should be included in the reference list using the minimum information recommended by DataCite (Dataset Creator, Dataset Title, Publisher [repository], Publication Year, Identifier [e.g. DOI, Handle, Accession or ARK]) and follow journal style.
