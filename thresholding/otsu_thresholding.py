import cv2 as cv
import os
from pathlib import Path

#specify input and output folder

input_folder = Path(r'C:\Users\lvinzenz\Documents\Data\Image Recognition\SurfaceDefectDetection\LeoderBachelor\Images_to_use')
output_folder = "OtsuThresholding"

#get list of all images in folder
liste = []
for image in input_folder.iterdir():
    if image.name.endswith('.jpg'):
        liste.append(image.name)

#apply thresholding with otsu on each image in folder and safe output to thresholding_folder
for i in range(len(liste)):
    image_path = os.path.join(input_folder, liste[i])
    if image_path.endswith(".jpg"):
        img = cv.imread(image_path, 0)
        #use kernelsize for medianBlur [1, 3, 5]
        blur_kernel = 5
        median = cv.medianBlur(img, blur_kernel)
        ret2,th2 = cv.threshold(median,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

        os.chdir(os.path.join(str(input_folder)+"\\"+output_folder))
        cv.imwrite(liste[i][0:-4] + "_blur" + str(blur_kernel) + "_"+ "Otsu.jpg", th2)



