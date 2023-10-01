import os
import cv2
import json
import traceback
import numpy as np
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
        barThresh = np.where(barInv > 200, 255, 0).astype(np.uint8)
        barLoc = (barThresh!=0).argmax(axis=0)

        # Take rolling average of barLoc
        barLoc = np.convolve(barLoc, np.ones(5)/5, mode='same').astype(int)

        # Shift each column in array by corresponding value in barLoc
        maxShift = np.max(barLoc)

        barPad = imGrey[y-maxShift:y+h+maxShift, x:x+w]
        barPadInv = cv2.bitwise_not(barPad)
        barPadInv = cv2.dilate(barPadInv, np.ones((5, 5), np.uint8), iterations=2)
        barPadInv = cv2.erode(barPadInv, np.ones((5, 5), np.uint8), iterations=2)

        barPadInv = np.array([np.roll(col, -1*loc) for col, loc in zip(barPadInv.T, barLoc)]).T

        rowSums = np.sum(barPadInv, axis=1)
        barLoc = np.where(rowSums > 0.7*np.max(rowSums))[0]

        # Take sample from just above and just below bar
        topSampleLine = np.mean(barPadInv[np.max(barLoc)+10:np.max(barLoc)+15, :], axis=0)[10:-10]
        botSampleLine = np.mean(barPadInv[np.min(barLoc)-15:np.min(barLoc)-10, :], axis=0)[10:-10]
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

        if len(markList) > 4:
            x = 1

        for mark in markList:
            # Draw on imGrey & write score
            score = 100 * mark / barPad.shape[1]
            cv2.line(im, (x + mark, y), (x + mark, y + h), (0, 255, 0), 10)
            cv2.putText(im, str(round(score, 1)), (x + mark, y + h + 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 10)

        resultsDict[idx] = [round(100 * mark / barPad.shape[1], 2) for mark in markList]
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
    csv = 'Filename,Bar,Score\n'
    for page, pageResults in results.items():
        for bar, barResults in pageResults.items():
            for result in barResults:
                csv += f'{page},{bar},{result}\n'
    return csv


def main():
    inputDir = 'input'

    outDir = 'output/bar_questions_marked_annotated'
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    pageResults = {}
    for f in tqdm(os.listdir(inputDir)):
        try:
            fPath = f'{inputDir}/{f}'
            im = loadFileFromInputDir(fPath)

            annotatedIm, results = findBarPositions(im)
            pageResults[f] = results
            cv2.imwrite(f'{outDir}/{f}', annotatedIm)
        except:
            print(f'Encountered error while processing {f}. The file will be skipped')
            print(traceback.format_exc())
            continue

    with open(f'{outDir}/bar_results.json', 'w') as f:
        json.dump(pageResults, f)

    with open(f'{outDir}/bar_results.csv', 'w') as f:
        f.write(convertResultsToCSV(pageResults))


if __name__ == '__main__':
    main()