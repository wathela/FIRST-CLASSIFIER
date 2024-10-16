## FIRST Classifier: Compact and Extended Radio Galaxies Classification using Deep Convolutional Neural Networks
- ### Wathela Alhassan et al. 2018
### Paper
MNRAS: https://academic.oup.com/mnras/advance-article/doi/10.1093/mnras/sty2038/5060783


Astro-ph: https://arxiv.org/abs/1807.10380


### [FIRST Classifier](FIRSTClassifier.py)
We present the FIRST Classifier, an online system for automated classification of compact and extended radio sources. The FIRST Classifier is based on a trained Deep Convolutional Neural Network (CNN) model to automate the morphological classification of compact and extended radio sources observed in the FIRST radio survey. Our model achieved an overall accuracy of 97%, with a recall of 98%, 100%, 98%, and 93% for Compact, BENT, FRI, and FRII galaxies, respectively. The current version of the FIRST classifier can predict the morphological class for a single source or a list of sources as Compact, FRI, FRII, or BENT.

<img src="https://github.com/wathela/FIRST-CLASSIFIER/blob/master/Diagram.png" width=493px>

### [Single Source Classification](FIRSTClassifier.py):
Classifies a single radio source.
- #### Input: 
  Coordinates of a single radio source (Right Ascension and Declination in degrees).
- #### Output: 
  Predicted morphology type (corresponding to the highest probability), a plot showing classification probabilities, and a direct link to download the FITS file cut-out for the target.

- How to run example:
  ```bash
  python FIRSTClassifier.py --ra 223.47337 --dec 26.80928
  
### [Multi Source classification](FIRSTClassifier.py):
Allow the classification of a list of sources (csv file).
- #### Input: 
  A CSV file containing the coordinates of multiple sources and the index of the Right Ascension and Declination columns (RA and DEC must be in degrees).
- #### Output: 
  A CSV file with four columns: Coordinates (RA and DEC), Predicted Class, Associated Probability, and a Link to download the FITS file cut-out.

- How to run example:
	```bash
  		python FIRSTClassifier.py --data_file path/to/file/test.csv --out_dir results.csv --ra_col 0 --dec_col 1
  
  
### Requirement:
- #### Python 3.x with the [Required Packages](requirements.txt) installed.

### How to cite:
@article{Alhassan2018,

author = {Alhassan, Wathela and Taylor, A R and Vaccari, Mattia},

doi = {10.1093/mnras/sty2038},

issn = {0035-8711},

journal = {Monthly Notices of the Royal Astronomical Society},

month = {jul},

title = {{The FIRST Classifier: Compact and Extended Radio Galaxy Classification using Deep Convolutional Neural Networks}},

url = {https://academic.oup.com/mnras/advance-article/doi/10.1093/mnras/sty2038/5060783},

year = {2018}

}
