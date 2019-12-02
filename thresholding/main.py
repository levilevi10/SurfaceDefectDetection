from pathlib import Path
from thresholding._archive.OBSOLET_image_to_binary import applyOtsuThresholding

folder = Path(r'C:\Users\lvinzenz\Documents\Data\Image Recognition\SurfaceDefectDetection\LeoderBachelor\Example_Images_Surface_Defects')

liste = []
for image in folder.iterdir():
    if image.name.endswith('.jpg'):
        liste.append(image.name)

print(liste)

for testimage in liste:
    applyOtsuThresholding(folder.name,testimage,120)