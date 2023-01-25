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

import representation
import utils
from csv import reader
import os
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


def evaluate_bar(data,encoding,barnum):
    instruments =[]
    for r in data:
    #   if r[0]==3 and r[1]<=64:
          
        if r[0]==3 and (r[1]-1)//4==barnum:
          if r[5] not in instruments:
                instruments.append(r[5]);
    instruments_number0=len(instruments)
   

    return {
        "instruments_number":instruments_number0,

    }
    
def get_all_instruments(data):
    instruments =[]
    for r in data:
        if r[0]==3 and r[5] not in instruments:
                instruments.append(r[5]);
    
    return instruments

   
def evaluate(data, encoding, filepath):
    """Evaluate the results."""
    path1 =(filepath)

    flag= True
    
    set_all={

        "instruments_number":0,
    }
    
    max_beat = 0
    for r in data:
        if r[1]>max_beat:
            max_beat=r[1]
    max_bar = (max_beat-1)//4

    max_bar_not_empty=0

    
    for i in range(max_bar+1):
        
        set_this=evaluate_bar(data,encoding,i)
        if set_this['instruments_number']==0:
            continue
        else:
            max_bar_not_empty+=1
        set_all['instruments_number']+=set_this['instruments_number']
    
    if max_bar_not_empty==0:
        set_all['instruments_number']=0
    else:
        set_all['instruments_number']=set_all['instruments_number']/(max_bar_not_empty)
    

    return set_all

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
    InputPath = "/work100/weixp/mtmt3-paper-2/mtmt-baseline-220/exp/lmd-bar/ape/samples-0/csv"
    
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

    InputPath_truth = InputPath+'/'+"truth"
    datanames1 = os.listdir(InputPath_truth)
    for i1 in datanames1:
        if i1[-3:]=="csv":
            print(i1)
            path1 = InputPath_truth+'/'+i1
            data_csv0 = pre_process(path1)
            instru_number1 = get_all_instruments(data_csv0)
            if len(instru_number1)<=1:
                continue
            result = evaluate(
                data_csv0, encoding,path1
            )
            results["truth"].append(result)
    
    
    
    InputPath_unconditioned = InputPath+'/'+"unconditioned"
    datanames2 = os.listdir(InputPath_unconditioned)
    for i2 in datanames2:
        if i2[-3:]=="csv":
            print(i2)
            path2 = InputPath_unconditioned+'/'+i2
            
            # ------------------------
            # Unconditioned generation
            # ------------------------
            # Evaluate the results
            data_csv2 = pre_process(path2)
            instru_number2 = get_all_instruments(data_csv2)
            if len(instru_number1)<=1:
                continue
            result = evaluate(
                data_csv2,
                encoding,
                path2
            )
            results["unconditioned"].append(result)
            
    
    for exp, result in results.items():
        logging.info(exp)
        for key in result[0]:
            logging.info(
                f"{key}: mean={np.nanmean([r[key] for r in result]):.4f}, "
                f"steddev={np.nanstd([r[key]for r in result]):.4f}"
            )


if __name__ == "__main__":
    main()