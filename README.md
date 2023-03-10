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
python mtmt/convert_sod.py
```

### Extract the note-level representation

Extract a list of notes from the MusPy JSON files.

```sh
python mtmt/extract.py -d sod
```

### Cut the seq_len to 1024
Enter the file address in the corresponding place of the code.

```sh
python cuthang_1024.py
```

### Extract the bar-level representation/track-level representation

sort the note-level representation to bar-level and track-level representation

```sh
python representation-bar.py -d sod
python representation-track.py -d sod
```


### Split training/validation/test sets

Split the processed data into training, validation and test sets.

```sh
python mtmt/split.py -d sod
```

## Training
Select the corresponding training model and go to the corresponding folder.

MSMM-LA in MSMM/MSMM-final-local_attetion/msmm

MSMM-GA in MSMM/MSMM-final-global_attetion/msmm


### MSMM-LA or MSMM-GA

First train both left and right encoders

  `python train_v1.py -d sod-note -o exp/sod-note/ape -g 0`
  
  `python train_v3.py -d sod-track -o exp/sod-track/ape -g 0`  

Please fill in the relevant parameters into train123.py, train encoder in the middle.

  `python -m torch.distributed.launch --nproc_per_node=3 train123.py -o exp/sod-bar/ape`

  
## Generation
Generate new samples using a trained model.

### MSMM-LA or MSMM-GA
```sh
python mtmt/generate.py -d sod-bar -o exp/sod-bar/ape -g 0
```

## Evaluation
The relevant codes for the evaluation are in MSMM/evaluate.

Evaluate the trained model.

Modify the location of the test folder specified in the py file and run the following code:
```sh
python evaluate-all.py
```
```sh
python evaluate-track-horizontal.py
```
```sh
python evaluate-instruments.py
```
