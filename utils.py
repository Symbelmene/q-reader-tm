import os
import cv2
from pdf2image import convert_from_path
from PIL import Image


def loadPDFToImage(pdf_file):
    pages = convert_from_path(pdf_file, 500)
    for page in pages:
        page.save('temp.jpg', 'JPEG')
    im = cv2.imread('temp.jpg')
    os.remove('temp.jpg')
    return im