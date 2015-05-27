PyReID: Re-identification Framework in Python
=====================================================
Flexible and extensible Framework for [Person Re-identification problem](http://www.sciencedirect.com/science/article/pii/S0262885614000262)].

PyReID allows configuring multiple **preprocessing** as a pipeline, followed by a **Feature Extraction** process and finally **Feature Matching**. It can calculate statistics like CMC or AUC. Not only that, but it also allows **PostRanking** optimization, with a functional GUI made with QT, and some embedded methods.

Main Features
-------------
  - Allows multiple preprocessing methods, including BTF, Illumination Normalization, Foreground/Background Segmentation using GrabCut, Symmetry-based Silhouette Partition, Static Vertical Partition and Weight maps for Weighted Histograms using Gaussian Kernels or Gaussian Mixture Models.
  
  - Feature Matching based on Histograms calculations. Admit 1D and 3D histograms, independent bin size for each channel, admits Regions for calculating independent Histograms per Region and Weight maps for Weighted Histograms.
  
  - Feature Matching admits Histograms comparison methods: Correlation, Chi-Square, Intersection, Bhattacharyya distance and Euclidean distance.
  
  - Automatically creates Ranking Matrix. For each element of the Probe obtains all Gallery elements sorted.
  
  - Statistics Module. With the Ranking Matrix and the Dataset, obtain stats as CMC, AUC, range-X and mean value.
  
  - Save stats in a complete excel for further review or plots creation. 
  
  - Save all your executions statistics in a file for later use (using python shelves).
  
Installation
------------
Right now the project is not prepared for a direct installation as a library. In the meantime, you can check the latest sources with the command:


    git clone https://github.com/Luigolas/PyReid.git

Dependencies
------------
PyReID is tested to work with Python 2.7+ and the next dependencies are needed to make it work:

  - **OpenCV 3.0**: This code makes use of OpenCV 3.0. It won't work with previous versions. This is due to incompatibility with 3D histograms and predefined values name.
  - **Numpy**: Tested with version *1.9.2*
  - **matplotlib**: Tested with version *1.4.2*
  - **pandas**: Tested with version *0.15.0*
  - **scikit-learn**: Tested with version *0.16.1*
  - **scipy**: Tested with version *0.14.1*
  - **xlwt**: Tested with version *0.7.5*

*Under Construction*...


Authors
-------
This project is initiated as a Final Project in the University of Las Palmas by **Luis González Medina**. 

Contact: luigolas@gmail.com

License
-------
The MIT License (MIT)

Copyright (c) 2015 Luis María González Medina

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
