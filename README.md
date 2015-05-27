PyReID: Re-identification Framework in Python
=====================================================
Flexible and extensible Framework for [Person Re-identification problem](http://www.sciencedirect.com/science/article/pii/S0262885614000262).

PyReID allows configuring multiple **preprocessing** as a pipeline, followed by a **Feature Extraction** process and finally **Feature Matching**. It can calculate statistics like CMC or AUC. Not only that, but it also allows **PostRanking** optimization, with a functional GUI made with QT, and some embedded methods.

Main Features
-------------
  - Allows multiple **preprocessing** methods, including BTF, Illumination Normalization, Foreground/Background Segmentation using GrabCut, Symmetry-based Silhouette Partition, Static Vertical Partition and Weight maps for Weighted Histograms using Gaussian Kernels or Gaussian Mixture Models.
  
  - **Feature Extraction** based on Histograms calculations. Admit 1D and 3D histograms, independent bin size for each channel, admits Regions for calculating independent Histograms per Region and Weight maps for Weighted Histograms.
  
  - **Feature Matching** admits Histograms comparison methods: Correlation, Chi-Square, Intersection, Bhattacharyya distance and Euclidean distance.
  
  - Automatically creates Ranking Matrix. For each element of the Probe obtains all Gallery elements sorted.
  
  - **Statistics Module**. With the Ranking Matrix and the Dataset, obtain stats as CMC, AUC, range-X and mean value.
  
  - **Use of multiprocessing** to improve speed. Preprocessing, Feature Extraction and Feature Matching are designed using multiprocessing to dramatically improve speed in multicore processors.
  
  - Save stats in a complete excel for further review or plots creation. Save all your executions statistics in a file for later use (using *Python Shelves*).
  
Installation
------------
Right now the project is not prepared for a direct installation as a library. In the meantime, you can check the latest sources with the command:


    git clone https://github.com/Luigolas/PyReid.git

Dependencies
------------
PyReID is tested to work with Python 2.7+ and the next dependencies are needed to make it work:

  - **OpenCV 3.0**: This code makes use of OpenCV 3.0. It won't work with previous versions. This is due to incompatibility with 3D histograms and predefined values names.
  - **Numpy**: Tested with version *1.9.2*
  - **matplotlib**: Tested with version *1.4.2*
  - **pandas**: Tested with version *0.15.0*
  - **scikit-learn**: Tested with version *0.16.1*
  - **scipy**: Tested with version *0.14.1*
  - **xlwt**: Tested with version *0.7.5*

*Under Construction*...

How to Use
----------
*This section is under construction. It is planned to have a dedicated page to further explain is usage and full potential.* 

Next code shows how to prepare a simple execution:

```python
from package.dataset import Dataset
import package.preprocessing as preprocessing
from package.image import CS_HSV, CS_YCrCb
import package.feature_extractor as feature_extractor
import package.feature_matcher as feature_matcher
from package.execution import Execution
from package.statistics import Statistics

# Preprocessing
IluNormY = preprocessing.Illumination_Normalization(color_space=CS_YCrCb)
mask_source = "../resources/masks/ViperOptimalMask-4.txt"
grabcut = preprocessing.Grabcut(mask_source)  # Segmenter
preproc = [IluNormY, grabcut]  # Executes on this order

fe = feature_extractor.Histogram(CS_HSV, [16, 16, 4], "1D")
f_match = feature_matcher.HistogramsCompare(feature_matcher.HISTCMP_BHATTACHARYYA)

probe = "../datasets/viper/cam_a"
gallery = "../datasets/viper/cam_b"
ex = Execution(Dataset(probe, gallery), preproc, fe, f_match)
ranking_matrix = ex.run()

statistic = Statistics()
statistic.run(ex.dataset, ranking_matrix)

print "Range 20: %f" % statistic.CMC[19]
print "AUC: %f" % statistic.AUC
```

To try the GUI for PostRanking just run *main.py*.

*Note: You will need to have a Person Re-identification Dataset to play with. You can download Viper dataset from [here](https://vision.soe.ucsc.edu/node/178). Example mask seeds for GrabCut are available at resource folder.*

Author
------
This project is initiated as a Final Project in the University of Las Palmas by [**Luis González Medina**](www.luigolas.com). 

Contact: luigolas@gmail.com

License
-------
Copyright (c) 2015 Luis María González Medina.

**PyReID** is free software made available under the **MIT License**. For details see the LICENSE file.