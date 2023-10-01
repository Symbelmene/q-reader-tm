# Q-Reader_TM

## Description
This code reads in a questionnaire with marked black bars and uses machine vision to locate marks
on the bars and score from 0-100 according to their distance along the bar. 

Place image files (.jpg, .png, .bmp) in the `images` folder and run `main.py` to score them. Results
will be dumped to the output folder in the form of annotated images, and CSV & JSON files with the scores.

## Installation
This code requires Python 3.8 or later. It is recommended to use a virtual environment to install the
dependencies. You can do this using either conda, or pip venv. I would recommend using conda as it is
easier to install the dependencies.

### Conda
Install Anaconda (https://www.anaconda.com/download) on your system, then run the following commands in the terminal:
```
conda create -n qreader python=3.8
conda activate qreader
pip install -r requirements.txt
```
This will create a virtual environment to install the dependencies in, and then install the dependencies. 
You will need to activate the environment each time you want to run the code by running:
```
conda activate qreader
```

Once inside the environment, you can run the code by placing the questionnaire files in the input directory (if it
does not exist then you may have to create it) and then running:
```
python main.py
```
