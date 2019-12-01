from bbox_extractor.binary_to_bbox import extract_bboxes

from bbox_extractor.binary_to_bbox import write_coordinate_file

bboxes_of_file = extract_bboxes(r"C:\Users\lvinzenz\Documents\Data\Image Recognition\SurfaceDefectDetection\LeoderBachelor\Example_Images_Surface_Defects\OtsuThresholding\Material-1-1-3-4_blur5_Otsu.jpg")

print(bboxes_of_file)


write_coordinate_file("new_with_blur_5.txt", bboxes_of_file)