import os
import shutil

import cv2
import json
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm

import utils


def findBarsInImage(im):
    # Find horizontal lines in image
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lineContours = []
    for cnt in contours:
        # Check if width is greater than height and that array is at least 50% of the image wide
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 * h and w > im.shape[1] / 2:
            lineContours.append(cnt)
            cv2.drawContours(im, [cnt], 0, (0, 255, 0), 3)
    return lineContours


def findMarks(im, barContours):
    imGrey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    resultsDict = {}
    for idx, cnt in enumerate(barContours):
        x, y, w, h = cv2.boundingRect(cnt)
        bar = imGrey[y:y + h, x:x + w]

        barInv = cv2.bitwise_not(bar)
        barInv = cv2.dilate(barInv, np.ones((5, 5), np.uint8), iterations=2)
        barInv = cv2.erode(barInv, np.ones((5, 5), np.uint8), iterations=2)
        barThresh = np.where(barInv > 100, 255, 0).astype(np.uint8)
        barLoc = (barThresh!=0).argmax(axis=0)

        # Take rolling average of barLoc
        barLoc = np.convolve(barLoc, np.ones(5)/5, mode='same').astype(int)

        maxShift = np.max(barLoc)

        barPad = imGrey[y-50:y+h+50, x:x+w]

        barPadInv = cv2.bitwise_not(barPad)
        barPadInv = np.where(barPadInv > 100, 255, 0).astype(np.uint8)
        barPadInv = cv2.dilate(barPadInv, np.ones((5, 5), np.uint8), iterations=2)
        barPadInv = cv2.erode(barPadInv, np.ones((5, 5), np.uint8), iterations=2)

        barPadInv = np.array([np.roll(col, -1*loc) for col, loc in zip(barPadInv.T, barLoc)]).T

        rowSums = np.sum(barPadInv, axis=1)
        barLoc = np.where(rowSums > 0.7*np.max(rowSums))[0]

        # Take sample from just above and just below bar
        topSampleLine = np.mean(barPadInv[np.max(barLoc)+10:np.max(barLoc)+25, :], axis=0)[10:-10]
        botSampleLine = np.mean(barPadInv[np.min(barLoc)-25:np.min(barLoc)-10, :], axis=0)[10:-10]

        marks = topSampleLine + botSampleLine

        # Find values greater than 0.2 of max that are at least 100px apart
        marks = np.where(marks > 100)[0]
        markList = []
        currGroup = []
        for mark in marks:
            if not currGroup:
                currGroup.append(mark)
            if mark - currGroup[-1] < 50:
                currGroup.append(mark)
            else:
                markList.append(int(sum(currGroup) / len(currGroup)))
                currGroup = []
        if currGroup:
            markList.append(int(sum(currGroup) / len(currGroup)))

        for mark in markList:
            # Draw on imGrey & write score
            score = 100 * mark / barPad.shape[1]
            cv2.line(im, (x + mark, y), (x + mark, y + h), (0, 255, 0), 10)
            cv2.putText(im, str(round(score)), (x + mark, y + h + 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10)

        resultsDict[idx] = [round(100 * mark / barPad.shape[1]) for mark in markList]
    return im, resultsDict


def findBarPositions(im):
    barContours = findBarsInImage(im)
    barContours.sort(key=lambda x: cv2.boundingRect(x)[1])
    showIm, results = findMarks(im, barContours)
    return showIm, results


def loadFileFromInputDir(path):
    fType = path.split('.')[-1]
    if fType == 'pdf':
        im = utils.loadPDFToImage(path)
    elif fType in ['jpg', 'png', 'bmp']:
        im = cv2.imread(path)
    else:
        raise Exception(f'File type {fType} not supported. Please ensure files are .pdf, .jpg, .png or .bmp')
    return im


def convertResultsToCSV(results):
    csv = 'Page No.,Question No.,Score(%)\n'
    for page, pageResults in results.items():
        for bar, barResults in pageResults.items():
            for result in barResults:
                csv += f'{page},{bar},{result}\n'
            if len(barResults) < 4:
                for _ in range(4 - len(barResults)):
                    csv += f'{page},{bar},\n'
    return csv


def readPageSequence():
    with open('page_numbers_and_sequence.json', 'r') as f:
        pageSequence = json.load(f)
    return pageSequence


def main():
    inputDir = 'input'

    outDir = 'output'
    if os.path.exists(outDir):
        shutil.rmtree(outDir)

    PAGE_DICT  = readPageSequence()
    PAGE_ORDER = PAGE_DICT['pageSequence']
    PAGE_QUESTIONS = PAGE_DICT['pageQuestions']

    for participantDir in os.listdir(inputDir):
        participantPath = f'{inputDir}/{participantDir}'
        outputPath = f'{outDir}/{participantDir}'

        if not os.path.exists(outputPath):
            os.makedirs(outputPath)

        pageResults = {}
        for idx, f in tqdm(enumerate(os.listdir(participantPath)), total=len(os.listdir(participantPath))):
            try:
                fPath = f'{participantPath}/{f}'
                im = loadFileFromInputDir(fPath)
                pageNumber = PAGE_ORDER[idx]
                questionNumbers = PAGE_QUESTIONS[str(pageNumber)]

                annotatedIm, results = findBarPositions(im)

                if len(questionNumbers) != len(results):
                    print(f'WARNING: Number of questions on page {pageNumber} does not match number of results found. \n'
                          f'Questions will not be marked!')
                    pageResults[pageNumber] = results
                else:
                    indexedResults = {questionNumbers[k]: v for k, v in results.items()}
                    pageResults[pageNumber] = indexedResults
                cv2.imwrite(f'{outDir}/{participantDir}/{f}', annotatedIm)
            except:
                print(f'Encountered error while processing {f}. The file will be skipped')
                print(traceback.format_exc())
                continue

        pageResults = {k: v for k, v in sorted(pageResults.items(), key=lambda item: int(item[0]))}
        with open(f'{outputPath}/bar_results.json', 'w') as f:
            json.dump(pageResults, f)

        with open(f'{outputPath}/bar_results.csv', 'w') as f:
            f.write(convertResultsToCSV(pageResults))

    # Merge participant output CSVs into single and add participant ID column
    dfList = []
    for participantDir in os.listdir(outDir):
        participantPath = f'{outDir}/{participantDir}'
        df = pd.read_csv(f'{participantPath}/bar_results.csv')
        df['Participant ID'] = participantDir
        dfList.append(df)
    df = pd.concat(dfList)
    df = df[['Participant ID', 'Page No.', 'Question No.', 'Score(%)']]
    df.to_csv(f'{outDir}/bar_results.csv', index=False)


if __name__ == '__main__':
    main()