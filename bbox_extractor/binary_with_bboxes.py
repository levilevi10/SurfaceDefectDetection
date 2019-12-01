from create_bb import afirstmain as bboxdrawer
import cv2 as cv
import os
from pathlib import Path

input_folder = Path(r'C:\Users\lvinzenz\Documents\Data\Image Recognition\SurfaceDefectDetection\LeoderBachelor\Example_Images_Surface_Defects')
output_folder = "WithBoundingBox"

#get list of all images in folder
liste = []
for image in input_folder.iterdir():
    if image.name.endswith('.jpg'):
        liste.append(image.name)

#draw bboxes on each image in folder and safe output to output_folder
for i in range(len(liste)):
    image_path = os.path.join(input_folder, liste[i])
    if image_path.endswith(".jpg"):
        result_img, boundbox = bboxdrawer(image_path)
        os.chdir(os.path.join(str(input_folder) + "\\" + output_folder))
        cv.imwrite(liste[i][0:-4] + "_" + "withBbox.jpg", result_img)
