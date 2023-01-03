# TiNet-iScience2023
Implementation code of TiNet: Human-understandable thyroid ultrasound imaging AI report system — A bridge between AI and clinicians

## Usage
1. For the input image, first use the ForOptFPN method to segment, you can use segmentation/scripts/train_ForOptFPN.sh for training, and use segmentation/scripts/eval_ForOptFPN.sh for testing.

2. The model folder contains methods for extracting five features of images.

3. After the previous segmentation and independent feature extraction, we have been able to obtain the results of the five features of the image. Use main.R to establish generalized linear equations for the calculation of comprehensive forecasts.

## Input Data
The input is an ultrasound image of the thyroid containing a nodule. The input is an ultrasound image of the thyroid gland containing a nodule.

## Metrics
Models were evaluated using accuracy, precision, sensitivity and AUC. For the segmentation model, IOU is also used for evaluation.
