#!/usr/bin/python
"""
    FIRST Classifier
    By: Wathela Alhassan
    Uinversity of Cape Town, Department of Astronomy.
    01/April/2018
    """

#import sys
#sys.path.append('/Users/wathelaalhassan/anaconda3/lib/python3.6/site-packages')

from scipy.misc import imread, imsave
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import skimage.io
import urllib
import json
import pyvo as vo
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from skimage.transform import resize
from skimage.transform import rescale
import PIL.Image as Image
import requests
from keras.models import load_model
from keras.optimizers import SGD, RMSprop, Adam
pd.set_option('display.max_colwidth', -1)
# do *not* show python warnings
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'



global img_size
img_size=150

global model_load
model_load = load_model("model.hdf5")



def make_clickable(val):
    return '<a href="{}">{}</a>'.format(val, val)

def FIRST_classifier(RA,DEC):
    url = 'http://skyview.gsfc.nasa.gov/cgi-bin/vo/sia.pl?survey=first&'
    try:
        img=vo.imagesearch(url,pos=(RA, DEC), size=0.1, format="image/fits")#, timeout=500)
        dataurl= img[0].getdataurl()
        
        resp = requests.get(dataurl)#timeout = 120
        
        if resp.status_code == 200:
            with open("fit.fits", "wb") as f:
                img = resp.content
                f.write(img)
            
            data = clean_crop_FITS('fit.fits')
            CLASS = classification(data)
            prob = max(Prob(data)[0])
            f.close()
        else:
            dataurl, CLASS, prob='nan','nan','nan'
            print("No data found at "+ str(ra)+' , '+str(dec))

    except (ValueError, IndexError):
        raise
        pass
#        continue
    except urllib.error.URLError: # RuntimeError: #URLError:# , NameError
        raise
        print("Time out!, Re-run please.")
#        continue
    return dataurl,CLASS,prob,data

# Clipper function (Anyan el at.)
def clip(data,lim):
    data[data<lim] = 0.0
    return data


def clean_crop_FITS(fname):
    hdu_list = fits.getdata(fname)
    image = np.squeeze(hdu_list)
    nan = np.isnan(image)
    image[nan] = 0
    sigma = 3.0
    
    mean, median, std = sigma_clipped_stats(image, sigma=sigma, iters=10)
    # Clip off n sigma points
    img_clip = clip(image,std*sigma)
    img_clip = img_clip[ 75:225, 75:225]
    
    minval, maxval = img_clip.min(),img_clip.max()
    norm = img_clip - minval
    img = norm*(1./(maxval-minval))
    
    return img

def plot_img(img, ra,dec):
    plt.figure()
    plt.title(str(ra)+'_'+str(dec))
    plt.imshow(img)#,cmap="gist_heat")
    plt.colorbar()
    plt.grid(False)
    plt.savefig(str(ra)+'_'+str(dec)+'.png')
    plt.show()
    return None
#  {'BENT': 0, 'COMP': 1, 'FRI': 2, 'FRII': 3}
def classes(classX):
    if classX[0] == 0:
        prd = "BENT"
    
    elif classX[0] == 2:
        prd = "I"
    
    elif classX[0] == 3:
        prd = "II"
    #     print(classX[0])
    elif classX[0] == 1:
        prd = "C"
    return prd

def classification(image):
    x = np.array(image)
    
    img = x#*1./255.
    img = np.reshape(img,(img_size,img_size,1))
    img = np.expand_dims(img, axis=0)
    
    optimizer = RMSprop(lr=0.0001,rho=0.9, epsilon=1e-08, decay=0.0, clipnorm=1.)#lr=0.001,rho=0.9, epsilon=1e-08, decay=0.00001)
    model_load.compile(loss='categorical_crossentropy',
                       optimizer=optimizer,
                       metrics=['accuracy'])

    classX = model_load.predict_classes(img)
                       #     print(classX)
    return classes(classX)#print(classes(classX))

def plot_prob(image, ra,dec):
    x = np.array(image)
    
    img = x#*1./255.
    img = np.reshape(img,(img_size,img_size,1))
    img = np.expand_dims(img, axis=0)
    
    optimizer = RMSprop(lr=0.0001,rho=0.9, epsilon=1e-08, decay=0.0, clipnorm=1.)#lr=0.001,rho=0.9, epsilon=1e-08, decay=0.00001)
    model_load.compile(loss='categorical_crossentropy',
                       optimizer=optimizer,
                       metrics=['accuracy'])

    classX1 = model_load.predict(img)
   
    pos =[1,2,3,4]
    plt.figure()
    plt.barh(pos, classX1[0][:], color='b',align='center')
    plt.yticks(pos, ('BENT','COMP', 'FRI', 'FRII'))
    plt.xlabel('Probability')
    plt.title('Morphology prediction of: '+str(ra)+', '+str(dec))
    plt.grid(True)
    plt.savefig(str(ra)+'_'+str(dec)+'.png')
    plt.show()


def Prob(image):
    x = np.array(image)
    img = x#*1./255.
    img = np.reshape(img,(img_size,img_size,1))
    img = np.expand_dims(img, axis=0)
    
    optimizer = RMSprop(lr=0.0001,rho=0.9, epsilon=1e-08, decay=0.0, clipnorm=1.)#lr=0.0001,rho=0.9, epsilon=1e-08, decay=0.0, clipnorm=1.
    model_load.compile(loss='categorical_crossentropy',
                       optimizer=optimizer,
                       metrics=['accuracy'])
        
    classX1 = model_load.predict(img)
    return classX1

# def model():
#     model_load = load_model("300sof_4class_model_grey.hdf5")
#     print("Model Loaded!")
#     return model_load
