#!/usr/bin/python
"""
Multi Source Classification using FIRST Classifier
By: Wathela Alhassan
Uinversity of Cape Town, Department of Astronomy.
01/April/2018
"""


from FIRST_CLASSIFIER import FIRST_classifier, plot_img, plot_prob
import pandas as pd
import argparse
import traceback

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='Path to csv file')
    parser.add_argument('--ra_col', type=int)
    parser.add_argument('--dec_col', type=int)#, help='will freeze the first N layers and unfreeze the rest')
    return parser.parse_args()

def csv_rw(file,ra_col,dec_col):
    
    gals_radio = pd.read_csv(file)
    
    t=0
    no_src=len(gals_radio[:])
    
    cols = ['RA', 'DEC', 'CLASS','PROB', 'URL']
    lst = []
    
    for i, cord in gals_radio.iterrows():
        t+=1
        try:
            RA, DEC = cord[ra_col], cord[dec_col]
            URL, CLASS, prob,img = FIRST_classifier(RA, DEC)
            print("Retriving image of ",RA, DEC,"||", t,"out of", no_src, '...')
            #             plot_img(img)
            
            lst.append([RA, DEC, CLASS, prob, URL])
        except:
#            raise
            continue

    df = pd.DataFrame(lst, columns=cols)
#    df.style.format({'URL': make_clickable})
    df.to_csv('class_result.csv', index=False, float_format='%g')
    return df


if __name__ == "__main__":
#    csv_rw("test/test.csv", 4, 5)
    try:
        args = parse_args()
        csv_rw(args.data_dir, args.ra_col, args.dec_col)
#        if args.data_dir:
#            data_dir = args.data_dir
#            ra =

#        user_input = input("CSV File name :")
#        ra_col = int(input("Input the index of Ra column: "))
#        dec_col = int(input("Input the index of Dec column: "))
#        csv_rw(user_input, ra_col, dec_col)
    except Exception as e:
        print(e)
        traceback.print_exc()

