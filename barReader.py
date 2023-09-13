import os
import cv2
import numpy as np


def getBoxes(barImage):
    if len(barImage.shape) == 3:
        barImage = cv2.cvtColor(barImage, cv2.COLOR_BGR2GRAY)

    # Draw boxes around found bar images
    contours, _ = cv2.findContours(barImage, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxList = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxList.append((x, y, w, h))

    return boxList


def measureEllipses(barImage, sampleBar):
    boxList = getBoxes(barImage)
    padX = (sampleBar.shape[1] // 2) + 10
    padY = (sampleBar.shape[0] // 2) + 10
    for idx, box in enumerate(boxList):
        x, y, w, h = box

        x0 = max(x - padX, 0)
        y0 = max(y - padY, 0)
        x1 = min(x + w + padX, barImage.shape[1])
        y1 = min(y + h + padY, barImage.shape[0])
        crop = barImage[y0:y1, x0:x1]

        # Template match sampleBar against crop and find best match
        res = cv2.matchTemplate(crop, sampleBar, cv2.TM_CCOEFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)

        templateCrop = crop[maxLoc[1]:maxLoc[1] + sampleBar.shape[0], maxLoc[0]:maxLoc[0] + sampleBar.shape[1], :]
        subtract = cv2.subtract(sampleBar, templateCrop)

        fullCrop = np.where(templateCrop, 0, 255).astype(np.uint8)[:, :, 0]

        # Get left and right most white pixels
        barPositions = np.nonzero(fullCrop)
        leftBar = np.min(barPositions[1])
        rightBar = np.max(barPositions[1])

        cv2.line(templateCrop, (leftBar, 0), (leftBar, templateCrop.shape[0]), (0, 0, 255), 2)
        cv2.line(templateCrop, (rightBar, 0), (rightBar, templateCrop.shape[0]), (0, 0, 255), 2)

        subCrop = subtract[:, :, 0]
        # Get left and right most points of ellipse
        try:
            markPositions = np.nonzero(subCrop)
            leftMark = np.min(markPositions[1])
            rightMark = np.max(markPositions[1])

            # Draw lines on image
            cv2.line(templateCrop, (leftMark, 0), (leftMark, templateCrop.shape[0]), (0, 255, 0), 2)
            cv2.line(templateCrop, (rightMark, 0), (rightMark, templateCrop.shape[0]), (0, 255, 0), 2)

            minPerc = int(100 * (leftMark - leftBar) / (rightBar - leftBar))
            maxPerc = int(100 * (rightMark - leftBar) / (rightBar - leftBar))
            meanPercent = int((minPerc + maxPerc) / 2)

            # Annotate barImage with values
            cv2.putText(barImage, f'{minPerc}%', (x0 + 100, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(barImage, f'{meanPercent}%', (x0 + 200, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(barImage, f'{maxPerc}%', (x0 + 300, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        except:
            # No mark on the line found
            pass

    cv2.imshow('barImage', cv2.resize(barImage, (1200, 500)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def measureLines(barImage, sampleBar):
    # Binarise both images
    barImageBin  = np.where(cv2.cvtColor(barImage, cv2.COLOR_BGR2GRAY) < 50, 255, 0).astype(np.uint8)
    sampleBarBin = np.where(cv2.cvtColor(sampleBar, cv2.COLOR_BGR2GRAY) < 50, 255, 0).astype(np.uint8)

    # Get boxes around bars
    boxList = getBoxes(barImageBin)

    padX = (sampleBar.shape[1] // 2) + 10
    padY = (sampleBar.shape[0] // 2) + 10
    for idx, box in enumerate(boxList):
        x, y, w, h = box

        x0 = max(x - padX, 0)
        y0 = max(y - padY, 0)
        x1 = min(x + w + padX, barImage.shape[1])
        y1 = min(y + h + padY, barImage.shape[0])
        crop = barImage[y0:y1, x0:x1]

        # Template match sampleBar against crop and find best match
        res = cv2.matchTemplate(crop, sampleBar, cv2.TM_CCOEFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)

        templateCrop = crop[maxLoc[1]:maxLoc[1] + sampleBar.shape[0], maxLoc[0]:maxLoc[0] + sampleBar.shape[1], :]
        subtract = cv2.subtract(sampleBar, templateCrop)

        fullCrop = np.where(templateCrop, 0, 255).astype(np.uint8)[:, :, 0]

        # Get left and right most white pixels
        barPositions = np.nonzero(fullCrop)
        leftBar = np.min(barPositions[1])
        rightBar = np.max(barPositions[1])

        cv2.line(templateCrop, (leftBar, 0), (leftBar, templateCrop.shape[0]), (0, 0, 255), 2)
        cv2.line(templateCrop, (rightBar, 0), (rightBar, templateCrop.shape[0]), (0, 0, 255), 2)

        lines = getBoxes(subtract[:, :, 0])
        lines.sort(key=lambda line: line[0])

        lineList = []
        for i in range(0, len(lines), 2):
            box1 = lines[i]
            box2 = lines[i + 1]
            centrePoint = (box1[0] + box2[0] + box2[2]) // 2
            lineList.append(centrePoint)

        for line in lineList:
            cv2.line(templateCrop, (line, 0), (line, templateCrop.shape[0]), (0, 255, 0), 2)

        cv2.imshow('templateCrop', templateCrop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    sampleBar = cv2.imread('sample_bar.png')
    barImage  = cv2.imread('line_bars.png')

    #measureEllipses(barImage, sampleBar)

    questionnaireDir = './bar_questions_marked'

    for f in os.listdir(questionnaireDir):
        measureLines(barImage, sampleBar)



if __name__ == '__main__':
    main()