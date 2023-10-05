# Q-Reader_TM

## Description
This code reads in a questionnaire with marked black bars and uses machine vision to locate marks
on the bars and score from 0-100 according to their distance along the bar. 

Place image files (.jpg, .png, .bmp) in the `images` folder and run `main.py` to score them. Results
will be dumped to the output folder in the form of annotated images, and CSV & JSON files with the scores.

## Installation
This code requires Python 3.10 or later. It is recommended to use a virtual environment to install the
dependencies. You can do this using either conda, or pip venv. I would recommend using conda as it is
easier to install the dependencies.

### Conda
Install Anaconda (https://www.anaconda.com/download) on your system, then run the following commands in the terminal:
```
conda create -n qreader python=3.10
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

### Input Format
Input files should be placed in the 'input' folder, (when running for the first time this folder will need to be created. The expected format is ./input/<participant_id>/<filename>.jpg. There is
no limit to the number of participant directories that can be placed in the input folder. 

Files within the participant ID directory should be .jpg, .png, or .bmp files. The code will automatically detect the files and process them. The file name
order is important to match the template in the page_numbers_and_sequence.json, but providing the pages are scanned in a 
similar order to the examples in test_sample.zip, then the code should work.

(If the question numbers / orders are changed, then this can be manually updated in the page_numbers_and_sequence.json file)


### Output Format
The output will be placed in the 'output' folder, which will be automatically created if it does not exist. The output folder
will contain a directory for each participant ID, along with a CSV file containing combined scores for all participants.

Each output participant ID directory will contain annotated images of the scanned pages with the marks on the bars detected
and scored. It will also contain a CSV and a JSON file with the scores for each participant (N.B. this is just a duplicate of the
participant data in the combined CSV file).


### Reformatted Output
The output can be reformmated to a more readable format using the reformat_output.py script. This will create a new file
in the output directory called reformatted_output.csv. This file will contain the data from bar_results.csv, but pivoted such
that each row is a participant, and questions with marks (e.g. Q2.2_1) are the columns, with scores as the cells.



