# OpenImages Visual Relationship Detection Challenge

## Dataset

Download the dataset from [here](https://storage.googleapis.com/openimages/web/challenge.html).  
The dataset is converted to Pascal VOC format before being used to train the models.  

Steps for preparing the data:-
1. Convert images to RGB format.
    ```Shell
        python scripts/convert_to_rgb.py <path to image dir>
        # This script validates all images in rgb format and converts them if not.
        # This is neccesary since many images are single channel which leads to problems in training
    ```
2. Generate xml files in VOC format.
    ``` Shell
        # Module 1
        python scripts/convert_data_attr_is.py <challenge-2018-train-vrd.csv> <image dir> < anno folder path> <log file>
        # Module 2
        python scripts/convert_data_attr_rest.py <challenge-2018-train-vrd.csv> <image dir> < anno folder path> <log file>
    ```
3. Generate labelmap files for SSD code.
    ``` Shell
        python scripts/gen_labelmap.py <challenge-2018-train-vrd.csv> <module> <output file>
        # module is one of is, crop and region. 
    ```
4. Generate trainval and test splits.  
The dataset is divided into 90:10 split for trainval:test.
    ``` Shell
        # For "is" and "region" module 
        python scripts/gen_trainval.py <challenge-2018-train-vrd.csv> <module> <output dir>
        # The scripts saves two files: trainval.txt and test.txt in output_dir
        # For "crop" module
        python scripts/gen_trainval.py <challenge-2018-train-vrd.csv> <output_dir>
    ```
Organise the images, annotations in xml format, trainval and test splits as in Pascal VOC data.   
For training the models, follow the instructions in the caffe-ssd submodule.