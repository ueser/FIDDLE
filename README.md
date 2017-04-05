# FIDDLE

An integrative deep learning framework for functional genomic data inference.

Based on: [http://biorxiv.org/content/early/2016/10/17/081380.full.pdf]

Thanks to @DylanM-Marshall for documentation and organization

<img src="https://cloud.githubusercontent.com/assets/25555398/22895505/c1247cc4-f1ea-11e6-85ef-0e81183a636d.png" title="Architecture" />

![alt text](https://cloud.githubusercontent.com/assets/1741502/24565878/28229be6-1625-11e7-88e5-555508e3e25c.gif)

## Installation and Quick Start

Docker image to be made in the near future. For now ... 

1) copy & paste the below:

_note: requires python 2.7!_

```markdown
git clone https://github.com/ueser/FIDDLE.git 
pip install virtualenv
virtualenv ../venv/
source ../venv/bin/activate
pip install -r requirements.txt
mkdir data/hdf5datasets/
```

2) download training, validation and test hdf5 datasets and place into 'data/hdf5datasets/':

_warning: several gb of data_

[training.h5](https://drive.google.com/file/d/0B9aDFb1Ds4IzWWZ5aWhtTkVUWE0/view?usp=sharing)
[validation.h5](https://drive.google.com/file/d/0B9aDFb1Ds4IzZ3JrLXp3SEY5aGs/view?usp=sharing)
[test.h5](https://drive.google.com/file/d/0B9aDFb1Ds4IzT05wTTZVQmFvcG8/view?usp=sharing)

3) Run it:

Change directories to FIDDLE/fiddle/

```markdown
python main.py --visualizePrediction online
```

4) Check out predictions:

Change directories to FIDDLE/results/experiment/

