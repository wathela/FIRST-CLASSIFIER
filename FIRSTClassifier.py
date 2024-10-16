#!/usr/bin/python
"""
FIRST Classifier
By: Wathela Alhassan
University of Cape Town, Department of Astronomy.
01/April/2018
"""

import pandas as pd
import argparse
import traceback
import csv
import logging
from typing import Optional
from utils import Classifier, plot_img, plot_prob


class FIRSTClassifier:
    def __init__(self, img_size: int = 150):
        """
        Initialize the FIRST Classifier with default image size.
        """
        self.img_size = img_size
    
    def write_to_file(self, output_file: str, row: list, header: Optional[list] = None) -> None:
        """
        Writes a single row of classification results to a CSV file.
        
        Args:
            output_file (str): Path to the output CSV file.
            row (list): Row of data to write to the file.
            header (list, optional): Header row for the CSV file, written only once if provided.
        """
        write_header = False
        # Check if the file exists
        try:
            with open(output_file, 'r'):
                pass
        except FileNotFoundError:
            write_header = True  # If file doesn't exist, write the header

        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header and header:
                writer.writerow(header)
            writer.writerow(row)

    def multi_sources(self, file: str, ra_col: int, dec_col: int, output_file: str) -> None:
        """
        Classifies multiple sources from a CSV file based on RA and DEC values, writing results directly to a CSV file.
        
        Args:
            file (str): Path to the input CSV file.
            ra_col (int): Column index for RA values in the CSV file.
            dec_col (int): Column index for DEC values in the CSV file.
            output_file (str): Path to the output CSV file.
            
        Output:
            - A CSV file with 4 columns: RA, DEC, predicted class, highest probability, and a link to download the cut-out FITS file for each source.
        
        """
        gals_radio = pd.read_csv(file)
        num_sources = len(gals_radio)
        print(f"Processing {num_sources} sources from the file.")
        
        # Define the header for the output file
        header = ['RA', 'DEC', 'CLASS', 'PROB', 'URL']
        
        for idx, row in gals_radio.iterrows():
            try:
                RA, DEC = row[ra_col], row[dec_col]
                print(f"Retrieving image for RA: {RA}, DEC: {DEC} || {idx + 1} out of {num_sources}...")
                
                # Classification process
                url, class_label, probability, img = Classifier(RA, DEC)
                # Write results to file
                result_row = [RA, DEC, class_label, max(probability), url]
                self.write_to_file(output_file, result_row, header if idx == 0 else None)

            except Exception as e:
                print(f"Error processing RA: {RA}, DEC: {DEC}: {e}")
                traceback.print_exc()
                continue

    def single_source(self, ra: float, dec: float) -> None:
        """
        Classifies a single source based on RA and DEC.
        
        Args:
            - ra (float): Right Ascension of the source in degrees.
            - dec (float): Declination of the source in degrees.
    
        Output:
            - Predicted morphology type (based on the highest probability).
            - A plot of classification probabilities.
            - A direct link to download the FITS file cut-out for the target.
        """

        try:
            print(f"Retrieving image for RA: {ra}, DEC: {dec} ....")
            url, class_label, probability, img = Classifier(ra, dec)
            plot_img(img, ra, dec)
            plot_prob(probability, ra, dec)
            print(f"Predicted Class: {class_label}, Probability: {max(probability)}")
            print(f"Image URL: {url}")
        except Exception as e:
            print(f"Error processing RA: {ra}, DEC: {dec}: {e}")
            traceback.print_exc()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Single and Multi Radio sources Classification from the FIRST survey using Convolutional Neural Networks.')
    parser.add_argument('--data_file', required=False, help='Path to the CSV file containing RA and DEC values')
    parser.add_argument('--out_dir', required=False, help='Path to the a directory on which the result file be saved')
    parser.add_argument('--ra_col', type=int, help='Column index for RA values')
    parser.add_argument('--dec_col', type=int, help='Column index for DEC values')
    parser.add_argument('--ra', type=float, help='RA for a single source classification')
    parser.add_argument('--dec', type=float, help='DEC for a single source classification')
    return parser.parse_args()



def main():

    print("*********************************************")
    print("* ^|^ ^|^ ^|^ ^|^ ^|^ ^|^ ^|^ ^|^ ^|^ ^|^ ^|^ *")
    print("* ^|^     Welcome to FIRST Classifier     ^|^ *")
    print("* ^|^ ^|^ ^|^ ^|^ ^|^ ^|^ ^|^ ^|^ ^|^ ^|^ ^|^ *")
    print("* Developed by Wathela Alhassan, et al. 2018. *")
    print("* Email: wathelahamed@gmail.com *")
    print("*********************************************")

    classifier = FIRSTClassifier() 
    
    try:
        args = parse_args()  
        
        if args.data_file is not None:  # If CSV file is provided, classify multiple sources
            if args.ra_col is not None and args.dec_col is not None:
                if args.out_dir is not None:
                    output_csv = args.out_dir
                else:
                    output_csv = 'classification_results.csv'
                    logging.info(f"Results will be stored in the current directory as {output_csv}.")
                    
                classifier.multi_sources(args.data_file, args.ra_col, args.dec_col, output_csv)
            else:
                raise ValueError("--data_dir, --ra_col and --dec_col have to be provided for multi source classification.")
        
        elif args.ra is not None and args.dec is not None:  # If RA and DEC are provided, classify a single source
            classifier.single_source(args.ra, args.dec)
        
        else:
            print()
            print()
            print("Please provide either a CSV file or RA/DEC for classification.")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":

    main()