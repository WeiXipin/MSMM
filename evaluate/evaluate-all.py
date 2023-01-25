import argparse
import logging
import pathlib
import pprint
import sys
from collections import defaultdict
import pandas as pd
import muspy
import numpy as np
import torch
import torch.utils.data
import tqdm
import os

import representation
import utils
from csv import reader

@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="show warnings only"
    )
    return parser.parse_args(args=args, namespace=namespace)

def csv_read_instruments(inputPath):
    csv_result = pd.read_csv(inputPath)
    row_list = csv_result.values.tolist()
    count =0
    # print(row_list)
    instruments =[]
    for r in row_list:
          if r[0]==3 and r[5] not in instruments:
                count+=1
                instruments.append(r[5])
    return count

def evaluate(data, encoding, filepath):
    """Evaluate the results."""

    path1 =filepath

    # Convert to a MusPy Music object
    music = representation.decode(data, encoding)
  
    if not music.tracks:
        return {
            "pitch_class_entropy": np.nan,
            "scale_consistency": np.nan,
            "groove_consistency": np.nan,
        }

    return {
        "pitch_class_entropy": muspy.pitch_class_entropy(music),
        "scale_consistency": muspy.scale_consistency(music),
        "groove_consistency": muspy.groove_consistency(
            music, 4 * music.resolution
        ),
    }

def pre_process(file):
    with open(file, 'r',encoding='latin-1') as f:
        data = list(reader(f))
    data = np.array(data[1:],dtype='int')
    
    instruments =[]
    for r in data:
        if r[1]<=64:
          if r[0]==3 and r[5] not in instruments:
                instruments.append(r[5])
    
    data_final=[]
    for r in data:
        if r[0]==1 and r[5] in instruments:
            data_final.append(r)
        if  r[0]!=1 and r[0]!=3:
            data_final.append(r)  
        if r[0]==3 and r[1]<=64:
            data_final.append(r)
    return data_final

def main():
    InputPath = "/work100/weixp/mtmt3-paper-2/mtmt-baseline-220/exp/sod/ape/eval-3"
    print(InputPath)
    InputPath_file=pathlib.Path(InputPath)
    encoding = representation.load_encoding("./encoding.json")
    # Parse the command-line arguments
    args = parse_args()
    # Set up the logger
    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(InputPath_file / "evaluate.log", "w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    results = defaultdict(list)

    InputPath_truth = InputPath+'/'+"truth"+'/csv'
    datanames1 = os.listdir(InputPath_truth)
    truth_number= 0
    for i1 in datanames1:
        if i1[-3:]=="csv":
            path1 = InputPath_truth+'/'+i1
            data_csv0 = pre_process(path1)
            result = evaluate(
                data_csv0, encoding,path1
            )
            results["truth"].append(result)
            truth_number+=1
    
    unconditioned_number= 0
    InputPath_unconditioned = InputPath+'/unconditioned'+'/csv'
    datanames2 = os.listdir(InputPath_unconditioned)
    for i2 in datanames2:
        if i2[-3:]=="csv":
            path2 = InputPath_unconditioned+'/'+i2
            
            # ------------------------
            # Unconditioned generation
            # ------------------------
            # Evaluate the results
            data_csv2 = pre_process(path2)
            result = evaluate(
                data_csv2,
                encoding,
                path2
            )
            results["unconditioned"].append(result)
            unconditioned_number+=1
            
    
    for exp, result in results.items():
        logging.info(exp)
        for key in result[0]:
            logging.info(
                f"{key}: mean={np.nanmean([r[key] for r in result]):.4f}, "
                f"steddev={np.nanstd([r[key]for r in result]):.4f}"
            )
    print("truth")
    print(truth_number)
    print("unconditioned")
    print(unconditioned_number)


if __name__ == "__main__":
    main()