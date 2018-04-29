#!/usr/bin/python
"""
    Single Source Classification using FIRST Classifier
    By: Wathela Alhassan
    Uinversity of Cape Town, Department of Astronomy.
    01/April/2018
    """

from FIRST_CLASSIFIER import FIRST_classifier, plot_img, plot_prob

def single_source(ra, dec):
    print("Retriving image of ",ra, dec ,"....")
    URL, CLASS, prob,img = FIRST_classifier(ra, dec)
#    plot_img(img)
#    plot_prob(img,ra,dec)
    return print("Predicted Class: ", CLASS),plot_img(img, ra, dec), plot_prob(img,ra,dec), print("Image Download :", URL)


if __name__ == "__main__":

    ra = float(input("Input the Ra of the source in Deg: "))
    dec = float(input("Input the Dec of the source in Deg: "))
    single_source(ra,dec)
