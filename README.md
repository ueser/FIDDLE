# FIDDLE

An integrative deep learning framework for functional genomic data inference.

Based on: [http://biorxiv.org/content/early/2016/10/17/081380.full.pdf]

Thanks to [Dylan Marshall](https://github.com/DylanM-Marshall) for documentation & organization.

On this page:
---------------
1. Installation and Quick Start
2. Further Details 
3. HMS Orchestra HPC Instructions

<img src="https://preview.ibb.co/iDo3v5/FIDDLE_001.jpg" title="Architecture" />
<img src="https://preview.ibb.co/eSebF5/FIDDLE_002.jpg" title="case study" />

![alt text](https://cloud.githubusercontent.com/assets/1741502/24565878/28229be6-1625-11e7-88e5-555508e3e25c.gif)

<img src="https://preview.ibb.co/mwc2oQ/FIDDLE_003.jpg" title="interpretation" />

Installation and Quick Start (can be done on a local machine)
-------------------------------------------------------------

##### 1. Set up FIDDLE environment:

_NOTE: Requires python 2.7 and pip. Anaconda can be a nuisance, make sure to comment out any "export PATH"s to Anaconda in your ~/.bash_profile or ~/.bashrc and then re-source it (or even restart current terminal session):_

###### a) Install Python package manager pip:

```markdown 
$ sudo easy_install pip 
```

###### b) Install isolated Python environments:

```markdown
$ sudo pip install virtualenv
```

###### c) Clone this repository to an appropriate location (for instance ~/Desktop):

```markdown 
$ git clone https://github.com/ueser/FIDDLE.git 
```

###### d) Instantiate FIDDLE virtual environment, source it:

```markdown
$ sudo virtualenv venvFIDDLE
$ source venvFIDDLE/bin/activate
```

###### e) Install necessary Python packages to FIDDLE virtual environment:

```markdown
$ pip install -r requirements.txt
```

##### 2. Download training/validation/test datasets:

###### a) Create data directory:

```markdown
$ cd FIDDLE/
$ mkdir -p data/hdf5datasets/
```

###### b) Download quickstart datasets: 

Place the following datasets in /FIDDLE/data/hdf5datasets/

_WARNING: several gb of data_

[training.h5](https://drive.google.com/file/d/0B9aDFb1Ds4IzWWZ5aWhtTkVUWE0/view?usp=sharing)

[validation.h5](https://drive.google.com/file/d/0B9aDFb1Ds4IzZ3JrLXp3SEY5aGs/view?usp=sharing)

[test.h5](https://drive.google.com/file/d/0B9aDFb1Ds4IzT05wTTZVQmFvcG8/view?usp=sharing)

##### 3) Run FIDDLE

```markdown
$ cd fiddle
```
___
**Documentation Interlude**

There are two (of many) methods to examine FIDDLE's internal documentation and docstrings:

###### a) Instantiating a Python session and using the help() function:

```markdown
$ python
>>> import main # or any other FIDDLE Python script
>>> help(main)
```

###### b) Employing the --help (or -h) flag (only shows information about flags):

```markdown
$ python main.py --help
```
___

```markdown
$ python main.py
```

##### 4) Create visualization of training:

```markdown
$ python visualization.py
```

##### 5) Create representations and predictions datasets:

```markdown
$ python analysis.py
```

##### 6) Examine training trajectory:

Change directories to FIDDLE/results/ < --runName (default = experiment) > /. The training trajectory visualization files (.png and .gif) are found in this directory. The representations and predictions created in step 5 are found in the hdf5 files "representations.h5" and "predictions.h5".

##### 7) Plot results:

Change directories to FIDDLE/fiddle and instantiate a jupter notebook session, start up the 'predictions_visualization.ipynb' and follow the instructions outlined in the Markdown cells.

To download Jupyter Notebook, start here: http://jupyter.readthedocs.io/en/latest/install.html.

```markdown
$ jupyter notebook
```

Further Details:
---------------------
For more complete instructions on file types and FIDDLE's work flow, open up the 'guide.ipynb' jupyter notebook. 

```markdown
cd FIDDLE/fiddle
$ jupyter notebook
```

HMS Orchestra HPC Instructions:
-------------------------------

##### 1) Start interactive session, enter FIDDLE directory:

```markdown
$ bsub -Is -q interactive bash
$ cd FIDDLE/
```

##### 2) Load correct Tensorflow module

```markdown
$ module load dev/tensorflow/1.0-GPU
```

##### 3) Set up virtual environment

###### a) Orchestra's Tensorflow module does not play nice with virtual environments, the module above _must_ be loaded before instantiating and then sourcing a virtual environment. More here: https://wiki.med.harvard.edu/Orchestra/PersonalPythonPackages

```markdown
$ virtualenv venvFIDDLE --system-site-packages
$ source venvFIDDLE/bin/activate
```

###### b) Comment out the 'tensorflow==1.0.1' line in the requirements.txt file:

```markdown
$ vim requirements.txt
tensorflow==1.0.1 --> #tensorflow==1.0.1
```

###### c) pip install remaining requirements:

```markdown
$ pip install -r requirements.txt
```

##### 4) Put those dwindling GPUs on blast... A template for submission lies in FIDDLE/fiddle/, modify accordingly. More on GPU usage here: https://wiki.med.harvard.edu/Orchestra/OrchestraNvidiaGPUs

```markdown
$ vim orchestra_job_submit.sh
$ bash orchestra_job_submit.sh
```
