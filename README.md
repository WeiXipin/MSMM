# MSMM

## Preprocessing

The relevant code for preprocessing is in pre-dataset/process. Go to that folder under.

### Prepare the name list

Get a list of filenames for each dataset.

```sh
find data/sod/SOD -type f -name *.mid -o -name *.xml | cut -c 14- > data/sod/original-names.txt
```

> Note: Change the number in the cut command for different datasets.

### Convert the data

Convert the MIDI and MusicXML files into MusPy files for processing.

```sh
python convert_sod.py
```

> Note: You may enable multiprocessing via the `-j {JOBS}` option. For example, `python convert_sod.py -j 10` will run the script with 10 jobs.

### Extract the note list

Extract a list of notes from the MusPy JSON files.

```sh
python extract.py -d sod
```

### Split training/validation/test sets

Split the processed data into training, validation and test sets.

```sh
python split.py -d sod
```

## Training
Training MSMM in MSMM/.Select the corresponding training model and go to the corresponding folder
MSMM-C-L in MSMM/tree/main/MSMM/MSMM-final-local_attetion/msmm
MSMM-C-G in MSMM/tree/main/MSMM/MSMM-final-global_attetion/msmm
MSMM-J-L in MSMM/MSMM-joint-local_attetion/msmm
MSMM-J-G in MSMM/MSMM-joint-global_attetion/msmm

### MSMM-C-L or MSMM-C-G
First train both left and right encoders
  `python train_v1.py -d sod -o exp/sod/ape -g 0`
  `python train_v3.py -d sod -o exp/sod/ape -g 0`  

Please fill in the relevant parameters into train123.py, train encoder in the middle.
  `python train123.py -d sod -o exp/sod/ape -g 0`

### MSMM-J-L or MSMM-J-G 
  `python train-sod.py -d sod -o exp/sod/ape -g 0`
  
## Generation
Generate new samples using a trained model.

### MSMM-C-L or MSMM-C-G
```sh
python mtmt/generate.py -d sod -o exp/sod/ape -g 0
```
### MSMM-J-L or MSMM-J-G 
```sh
python mtmt/generate-sod.py -d sod -o exp/sod/ape -g 0
```

## Evaluation
The relevant codes for the evaluation are in MSMM/tree/main/evaluate.

Evaluate the trained model.

```sh
python evaluate-all.py -d sod -o exp/sod/ape -g 0
```
```sh
python evaluate-track-horizontal.py -d sod -o exp/sod/ape -g 0
```
```sh
python evaluate-instruments.py -d sod -o exp/sod/ape -g 0
```