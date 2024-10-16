#!/usr/bin/python
"""
    FIRST Classifier
    By: Wathela Alhassan
    University of Cape Town, Department of Astronomy.
    01/April/2018
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from keras.models import load_model
from keras.optimizers import RMSprop
# from tensorflow.keras.optimizers.legacy import RMSprop
from skimage.transform import resize
import warnings
import os
import pyvo as vo
from typing import Tuple



# Set display options and suppress warnings
pd.set_option('display.max_colwidth', None)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Global constants
IMG_SIZE: int = 150
MODEL = load_model("model.hdf5")


# Clipper function to remove values below threshold
def clip(data: np.ndarray, lim: float) -> np.ndarray:
    data[data < lim] = 0.0
    return data

# Function to clean and crop FITS image
def clean_crop_FITS(fname: str) -> np.ndarray:
    hdu_list = fits.getdata(fname)
    image = np.squeeze(hdu_list)
    image[np.isnan(image)] = 0
    sigma: float = 3.0

    _, _, std = sigma_clipped_stats(image, sigma=sigma, maxiters=10)
    img_clip = clip(image, std * sigma)
    img_clip = img_clip[75:225, 75:225]

    minval, maxval = img_clip.min(), img_clip.max()
    norm = img_clip - minval
    img = norm * (1. / (maxval - minval))

    return img

# Plotting function for images
def plot_img(img: np.ndarray, ra: float, dec: float) -> None:
    plt.figure()
    plt.title(f'Retrieved cut-out of {ra}_{dec}')
    plt.imshow(img)
    plt.colorbar()
    plt.grid(False)
    plt.savefig(f'{ra}_{dec}.png')
    plt.show()

# Classifier for image data
def get_class_name(classX: np.ndarray) -> str:
    class_names = {0: "BENT", 1: "C", 2: "I", 3: "II"}
    return class_names.get(classX[0], "Unknown")

# Classification function for the image
def Predict(image: np.ndarray) -> str:
    img = np.reshape(image, (IMG_SIZE, IMG_SIZE, 1))
    img = np.expand_dims(img, axis=0)

    optimizer = RMSprop(learning_rate=0.0001, rho=0.9, epsilon=1e-08, clipnorm=1.0)# decay=0.0,
    MODEL.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    classX = np.argmax(MODEL.predict(img),axis=1)
    prob = MODEL.predict(img)
    return get_class_name(classX), list(prob[0])

# Function to plot probability distribution
def plot_prob(prob: list, ra: float, dec: float) -> None:

    pos = [1, 2, 3, 4]
    plt.figure()
    plt.barh(pos, prob[:], color='b', align='center')
    plt.yticks(pos, ['BENT', 'COMP', 'FRI', 'FRII'])
    plt.xlabel('Probability')
    plt.title(f'Morphology prediction of: {ra}, {dec}')
    plt.grid(True)
    plt.savefig(f'{ra}_{dec}.png')
    plt.show()

# # Function to write data to file
def write_to_file(filename: str, content: bytes) -> None:
    with open(filename, "wb") as file:
        file.write(content) 

# Helper function for URL formatting
def make_clickable(val: str) -> str:
    return f'<a href="{val}">{val}</a>'

# Classifier for FIRST images
def Classifier(RA: float, DEC: float) -> Tuple[str, str, float, np.ndarray]:

    """
        fetches the radio source image for the target source from the FIRST survey database using Virtual Observatory tools 
        and perform a morphological classification using a trained deep neural network model.

        Input:
        - RA (float): Right Ascension of the source in degrees.
        - DEC (float): Declination of the source in degrees.

        Output:
        - A tuple containing:
            1. Predicted morphological class of the source (as a string).
            2. Direct URL to download the FITS file of the source's cut-out image.
            3. Probability of the predicted class (as a float).
            4. The cut-out image of the source as a NumPy array.

    """


    url: str = 'http://skyview.gsfc.nasa.gov/cgi-bin/vo/sia.pl?survey=first&'
    try:
        img = vo.imagesearch(url, pos=(RA, DEC), size=0.1, format="image/fits")
        data_url: str = img[0].getdataurl()

        response = requests.get(data_url)

        if response.status_code == 200:
            img_content = response.content
            write_to_file("fit.fits", img_content)

            data = clean_crop_FITS('fit.fits')
            predicted_cls, prob = Predict(data)
            
        else:
            data_url, predicted_cls, prob, data = 'nan', 'nan', 'nan', np.array([])

    except (ValueError, IndexError, requests.exceptions.RequestException):
        raise
    return data_url, predicted_cls, prob, data
