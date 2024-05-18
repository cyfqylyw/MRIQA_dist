# MRIQA_dist

Code implementation for

>  



## Datasets Preparation

### Change directory to `datasets` folder

```
cd datasets
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


### Download [IXI](https://brain-development.org/ixi-dataset/) dataset

```
wget http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar
wget http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T2.tar

mkdir IXI-T1
tar -xvf IXI-T1.tar -C IXI-T1

mkdir IXI-T2
tar -xvf IXI-T2.tar -C IXI-T2
```


### Download [QTAB](https://openneuro.org/datasets/ds004146/versions/1.0.4) dataset using [AWS CLI](https://aws.amazon.com/cli/)

```
mkdir QTAB
cd QTAB
aws s3 sync --no-sign-request s3://openneuro.org/ds004146 ds004146-download/
```


### Download [ARC](https://openneuro.org/datasets/ds004884/versions/1.0.1) dataset using [AWS CLI](https://aws.amazon.com/cli/)

```
mkdir ARC
cd ARC
aws s3 sync --no-sign-request s3://openneuro.org/ds004884 ds004884-download/
```


<!-- 
### Download [preprocessed ABIDE](http://preprocessed-connectomes-project.org/abide/) dataset

```
curl -O -L https://raw.githubusercontent.com/preprocessed-connectomes-project/abide/master/download_abide_preproc.py

mkdir ABIDE

python download_abide_preproc.py -d reho -p ccs -s nofilt_noglobal -o ./ABIDE
python download_abide_preproc.py -d reho -p cpac -s nofilt_noglobal -o ./ABIDE
python download_abide_preproc.py -d reho -p dparsf -s nofilt_noglobal -o ./ABIDE
python download_abide_preproc.py -d reho -p niak -s nofilt_noglobal -o ./ABIDE
``` 
-->


## Distortion generation

Generate five types of distortion and four levels for each with [TorchIO](https://torchio.readthedocs.io) library.

```
python transform.py

python distortion.py --dist motion
python distortion.py --dist ghosting
python distortion.py --dist spike
python distortion.py --dist noise
python distortion.py --dist blur
```



## Overview of the datasets

| Dataset | Year | # Subjects | # Images | # Samples | Format | Shape |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| OASIS | 2007 | 416 | 1688 | 50640 | .img | (256, 256, 128, 1) |
| NFBS | 2011 | 125 | 125 | 3750 | .mnc |  (193, 229, 193)  |
| IXI (T1w) | 2008 | 581 | 581 | 14525 | .nii.gz | (256, 256, 130-150) |
| IXI (T2w) | 2008 | 578 | 578 | 14375 | .nii.gz | (256, 256, 120-150) |
| QTAB (T1w) | 2022 | 417 | 1441 | 36025 | .nii.gz | (176-208, 300, 320) |
| QTAB (T2w) | 2022 | 417 | 1821 | 45525 | .nii.gz | (768, 768, 50-60) |
| ARC (T1w) | 2023 | 230 | 447 | 11100 | .nii.gz | (160-192, 256, 256) |
| ARC (T2w) | 2023 | 230 | 441 | 10925 | .nii.gz | (160-192, 256, 256) |
<!-- | ABIDE (ccs) | 2013 | 884 | 884 | 22100 | .nii.gz | (61, 73, 61) |
| ABIDE (cpac) | 2013 | 884 | 884 | 22100 | .nii.gz | (61, 73, 61) |
| ABIDE (dparsf) | 2013 | 884 | 884 | 22100 | .nii.gz | (61, 73, 61) |
| ABIDE (niak) | 2013 | 884 | 884 | 22100 | .nii.gz | (61, 73, 61) | -->


*Notes:*
- \# Subjects: Number of research subjects or participants.
- \# Images: Number of MR images obtained for each study.
- \# Samples: Number of MR images after distortion processing


## Usage

Initialize the python environment.

```
git clone https://github.com/cyfqylyw/MRIQA_dist.git
cd MRIQA_dist
conda create -n mriqa python=3.9
conda activate mriqa
pip install -r requirements.txt
```



## Citation

TBD




