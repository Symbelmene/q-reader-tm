import os
import cv2
import time
import numpy as np
from tqdm import tqdm


def rotateImageByDegrees(image, angleDegrees):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angleDegrees, 1.0)
    imRot = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return imRot


def templateMatchWithRotation(im, template, rotation=0):
    # Template match template against im and rotate the template by +/- rotation degrees
    # Return all matches above a threshold that are at least 100px apart
    downsampledIm = cv2.resize(im, (0, 0), fx=0.2, fy=0.2)
    downsampledTemplate = cv2.resize(template, (0, 0), fx=0.2, fy=0.2)
    threshIm = np.where(downsampledIm > 200, 255, 0).astype(np.uint8)
    threshTemplate = np.where(downsampledTemplate > 200, 255, 0).astype(np.uint8)
    threshTemplate = cv2.copyMakeBorder(threshTemplate, 200, 200, 0, 00, cv2.BORDER_CONSTANT, value=255)

    bestResult = None
    bestAngle = 0
    for angle in np.linspace(-rotation, rotation, 10):
        rotatedTemplate = rotateImageByDegrees(threshTemplate, angle)[190:-190, :]
        res = cv2.matchTemplate(threshIm, rotatedTemplate, cv2.TM_CCOEFF_NORMED)
        if bestResult is None:
            bestResult = res
        elif np.max(res) > np.max(bestResult):
            bestResult = res
            bestAngle = angle
        else:
            pass

    # Correct page rotation and do final match
    im = rotateImageByDegrees(im, -1*bestAngle)

    threshIm = np.where(im > 200, 255, 0).astype(np.uint8)
    threshTemplate = np.where(template > 200, 255, 0).astype(np.uint8)

    bestResult = cv2.matchTemplate(threshIm, threshTemplate, cv2.TM_CCOEFF_NORMED)

    # Find all matches above a threshold
    bestYMatches = np.max(bestResult, axis=1)

    # Find all matches above a threshold
    loc = np.where(bestYMatches >= np.max(bestYMatches) * 0.5)[0]

    # For matches that are less than 100px apart, take best score
    finalYPts = []
    groupList = []
    for i in range(loc.size):
        if not groupList:
            groupList.append(loc[i])
        elif loc[i] - groupList[-1] < 100:
            groupList.append(loc[i])
        else:
            # Get best score from group
            bestScoreIdx = np.argmax(bestYMatches[groupList])
            finalYPts.append(groupList[bestScoreIdx])
            groupList = []

        # Get last group scores
        if i == loc.size - 1:
            # Get best score from group
            bestScoreIdx = np.argmax(bestYMatches[groupList])
            finalYPts.append(groupList[bestScoreIdx])

    # Get associated x values
    finalXPts = [bestResult[i,:].argmax() for i in finalYPts]

    # Combine and convert to x,y,w,h
    loc = np.array([finalXPts, finalYPts]).T

    boxList = []
    for pt in loc:
        boxList.append((pt[0], pt[1], template.shape[1], template.shape[0]))

    return im, boxList


def findMarkPoints(bar, resolution):
    topBarLine = bar[0, :]
    topResults = groupPointsByLocation(topBarLine, resolution)

    bottomBarLine = bar[-1, :]
    bottomResults = groupPointsByLocation(bottomBarLine, resolution)

    # Test if top and bottom matches
    finalResults = []
    for top in topResults:
        for bottom in bottomResults:
            if abs(top - bottom) < bar.shape[1] / (resolution / 2):
                finalResults.append(int((top + bottom) / 2))
                break

    if len(finalResults) == 4:
        return finalResults

    # Test if top or bottom matches
    finalResults = []
    if len(finalResults) < 4:
        if len(topResults) == 4:
            finalResults = topResults
        elif len(bottomResults) == 4:
            finalResults = bottomResults

    if len(topResults + bottomResults) == 4:
        return topResults + bottomResults

    return topResults


def groupPointsByLocation(barLine, resolution):
    markPts = np.nonzero(barLine == 0)[0]
    resultPts = []
    groupPts = []
    for i in range(markPts.size):
        if not groupPts:
            groupPts.append(markPts[i])
        elif markPts[i] - groupPts[-1] < barLine.size / resolution:
            groupPts.append(markPts[i])
        else:
            resultPts.append(groupPts[int(len(groupPts) / 2)])
            groupPts = []

        # Get last group scores
        if i == markPts.size - 1:
            if not groupPts:
                resultPts.append(markPts[i])
            else:
                resultPts.append(groupPts[int(len(groupPts) / 2)])

    return resultPts


def findMarkScores(bar, marks):
    barPts = np.nonzero(bar[bar.shape[0] // 2, :] == 0)
    barLeft = barPts[0][0]
    barRight = barPts[0][-1]

    marksAsScores = []
    for mark in marks:
        marksAsScores.append(100*(mark - barLeft) / (barRight - barLeft))

    return [round(score, 1) for score in marksAsScores]


def findBarPositions(im, masterBar, resolution=200):
    im, results = templateMatchWithRotation(im, masterBar, rotation=2)
    showIm = findMarks(im, resolution, results)
    return showIm


def findMarks(im, resolution, results):
    showIm = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for idx, box in enumerate(results):
        bar = im[box[1]-5:box[1] + box[3]+5, box[0]:box[0] + box[2]]
        barBlur = cv2.GaussianBlur(bar, (5, 5), 0)
        barThresh = np.where(barBlur > 200, 255, 0).astype(np.uint8)

        marks = findMarkPoints(barThresh, resolution)
        scores = findMarkScores(barThresh, marks)

        # Draw marks and scores on original image, im
        topFlag = True
        for mark, score in zip(marks, scores):
            if len(scores) == 4:
                colour = (0, 255, 0)
            else:
                colour = (0, 0, 255)
            cv2.line(showIm, (box[0] + mark, box[1]), (box[0] + mark, box[1] + box[3]), colour, 10)
            if topFlag:
                cv2.putText(showIm, str(score), (box[0] + mark, box[1] + box[3] + 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            colour, 10)
                topFlag = False
            else:
                cv2.putText(showIm, str(score), (box[0] + mark, box[1] + box[3] - 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            colour, 10)
                topFlag = True
    return showIm


def main():
    masterBar = cv2.imread('master_bar.png', cv2.IMREAD_GRAYSCALE)
    outDir = 'bar_questions_marked_annotated'
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    for f in tqdm(os.listdir('bar_questions_marked')):
        st = time.time()
        fPath = f'bar_questions_marked/{f}'
        im = cv2.imread(fPath, cv2.IMREAD_GRAYSCALE)
        annotatedIm = findBarPositions(im, masterBar)
        cv2.imwrite(f'bar_questions_marked_annotated/{f}', annotatedIm)
        print(f'{f} took {time.time() - st} seconds')


if __name__ == '__main__':
    main()