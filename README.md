## FIRST Classifier: Compact and Extended Radio Galaxies Classification using Deep Convolutional Neural Networks
- ### Wathela Alhassan et al.
### Paper
`Coming soon` ...

### [FIRST Classifier](FIRST_CLASSIFIER.py)
We present the FIRST Classifier, an on-line system for automated classification of Compact and Extended radio sources. We developed the FIRST Clas- sifier based on a trained Deep Convolutional Neural Network Model to automate the morphological classification of compact and extended radio sources observed in the FIRST radio survey. Our model achieved an overall accuracy of 97% and a recall of 98%, 100%, 98% and 93% for Compact, BENT, FRI and FRII galaxies respectively. The current version of the FIRST classifier is able to predict the morphological class for a single source or for a list of sources as Compact or Extended (FRI, FRII and BENT).

### [Single Source classification](single_source_classification.py):
Classify only one single source.
#### Input: 
Coordinates of single radio source (Right Ascention and Declination in degree).
#### Output: 
Predicted morphology type(corresponding to the highest probability), probabilities plot of the classification and a direct link to download the FITS file cut out of the target.

How to run example:
python3 single_source_classification.py -Ra  124.39358 -Dec 44.98078

### [Multi Sources classification](multi_sources_classification.py):
Allow the classification of a list of sources (csv file).
#### Input: 
A csv file that has a list of coordinates of sources and index of the Right Ascention and Declination columns ( RA and DEC must be in degree).
#### Output: 
A csv file containing 4 columns: Coordinates (RA and DEC), Predicted class, Highest probability, Link to download the cut-out FITS file.

How to run example:
python3 multi_source_classification.py --data_dir wathela/test.csv --ra_col 0    --dec_col 1

#### Make sure all the [Requirement Packages](requirements.txt) are installed.
