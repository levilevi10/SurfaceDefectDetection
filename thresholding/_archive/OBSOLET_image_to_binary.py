import cv2 as cv

def applyOtsuGaussThresholding(folder, filename, minvalue):
    '''
    INPUT
    folder: Folder with images
    filename: Name of the image
    minvalue: Minimum value for thresholding (between 0 and 255)
    OUTPUT
    writing 1 image in cwd

    THRESH_OTSU_GAUSS'
    '''
    #get path to image
    cwd = os.getcwd()
    img = cv.imread(cwd + folder + filename,0)
    maxvalue = 255
    ret,thresh6 = cv.threshold(img,minvalue,maxvalue,cv.THRESH_OTSU)
    blur = cv.GaussianBlur(img,(5,5),0)
    ret,thresh7 = cv.threshold(blur,0,maxvalue,cv.THRESH_OTSU)
    titles = ['THRESH_OTSU_GAUSS']
    images = [thresh7]

    #write files
    for name, img in zip(titles, images):
        cv.imwrite(filename + name + str(minvalue) + ".jpg", img)
        print('writing %s' %  name)

    return


def applyOtsuThresholding(folder, filename, minvalue):
    '''
    INPUT
    folder: Folder with images
    filename: Name of the image
    minvalue: Minimum value for thresholding (between 0 and 255)
    OUTPUT
    writing 1 images in cwd

    THRESH_OTSU
    '''
    #get path to image
    cwd = os.getcwd()
    base , current = os.path.split(cwd)
    img = cv.imread(folder + filename,0)
    if type(img) == None:
        print("Error: File was not read")
        return
    maxvalue = 255
    ret,thresh6 = cv.threshold(img,minvalue,maxvalue,cv.THRESH_OTSU)
    titles = ['THRESH_OTSU']
    images = [thresh6]

    #write files
    for name, img in zip(titles, images):
        cv.imwrite(filename + name + str(minvalue), img)
        print('writing %s' %  name)
    return







import os
from pathlib import Path
from thresholding._archive.OBSOLET_image_to_binary import applyOtsuThresholding

folder = Path(r'C:\Users\lvinzenz\Documents\Data\Image Recognition\SurfaceDefectDetection\LeoderBachelor\Example_Images_Surface_Defects')

folderdriver = folder.drive
liste = []
for image in folder.iterdir():
    if image.name.endswith('.jpg'):
        liste.append(image.name)

print(liste)
foldername = folder.name
basepath = folder.anchor
for testimage in liste:
    applyOtsuThresholding(str(folder)+ "/" ,testimage,120)
    print(str(folder),testimage)