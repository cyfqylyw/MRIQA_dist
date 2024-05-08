# MRIQA_dist

Code implementation for

>  



## Dataset Preparation

### Change directory to `datasets` folder

```
cd datasets
```


### Download [IXI](https://brain-development.org/ixi-dataset/) dataset

```
wget http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar
wget http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T2.tar

mkdir IXI-T1
tar -xvf IXI-T1.tar -C IXI-T1

mkdir IXI-T2
tar -xvf IXI-T1.tar -C IXI-T2
```


### Download [OASIS](https://sites.wustl.edu/oasisbrains/home/oasis-1/) dataset

```
cd OASIS
chmod +x download_oasis.sh
./download_oasis.sh
```


### Download [NFBS](http://preprocessed-connectomes-project.org/NFB_skullstripped/index.html) dataset

```
mkdir NFBS
cd NFBS
wget https://fcp-indi.s3.amazonaws.com/data/Projects/RocklandSample/NFBS_BEaST_Library.tar
tar -xvf NFBS_BEaST_Library.tar
```


### Download [preprocessed ABIDE](http://preprocessed-connectomes-project.org/abide/) dataset

```
curl -O -L https://raw.githubusercontent.com/preprocessed-connectomes-project/abide/master/download_abide_preproc.py

mkdir ABIDE

python download_abide_preproc.py -d reho -p ccs -s nofilt_noglobal -o ./ABIDE
python download_abide_preproc.py -d reho -p cpac -s nofilt_noglobal -o ./ABIDE
python download_abide_preproc.py -d reho -p dparsf -s nofilt_noglobal -o ./ABIDE
python download_abide_preproc.py -d reho -p niak -s nofilt_noglobal -o ./ABIDE
```


## Overview of datasets

| Dataset | Year | # Subject | # Image | format | shape |
| --- | --- | --- | --- | --- | --- |
| IXI (T1w) | 2008 | 581 | 581 | .nii.gz | (256, 256, 130-150) |
| IXI (T2w) | 2008 | 581 | 581 | .nii.gz | (256, 256, 140-150) |
| OASIS-1 | 2007 | 416 | 1688 | .img | (256, 256, 128, 1) |
| NFBS | 2011 | 125 | 125 | .mnc |  (193, 229, 193)  |
| preprocessed ABIDE (ccs) | 2013 | 884 | 884 | .nii.gz | (61, 73, 61) |
| preprocessed ABIDE (cpac) | 2013 | 884 | 884 | .nii.gz | (61, 73, 61) |
| preprocessed ABIDE (dparsf) | 2013 | 884 | 884 | .nii.gz | (61, 73, 61) |
| preprocessed ABIDE (niak) | 2013 | 884 | 884 | .nii.gz | (61, 73, 61) |



## Usage





## Citation

TBD




