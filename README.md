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

### Download [ABIDE](http://preprocessed-connectomes-project.org/abide/) dataset

```
curl -O -L https://raw.githubusercontent.com/preprocessed-connectomes-project/abide/master/download_abide_preproc.py

mkdir ABIDE

python download_abide_preproc.py -d reho -p ccs -s nofilt_noglobal -o ./ABIDE
python download_abide_preproc.py -d reho -p cpac -s nofilt_noglobal -o ./ABIDE
python download_abide_preproc.py -d reho -p dparsf -s nofilt_noglobal -o ./ABIDE
python download_abide_preproc.py -d reho -p niak -s nofilt_noglobal -o ./ABIDE
```







## Usage





## Citation






