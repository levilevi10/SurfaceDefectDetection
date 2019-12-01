import cv2
import numpy as np


def find_if_close(cnt1, cnt2):
    LIMIT = 10
    row1, row2 = cnt1.shape[0], cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i] - cnt2[j])
            if abs(dist) < LIMIT:
                return True
            elif i == row1 - 1 and j == row2 - 1:
                return False


# Get BBox from array of coordinates
def allign_bbox(res_coords, bboxnumber):
    print("In allign_bbox", type(res_coords))
    left = res_coords[bboxnumber][1][0]
    top = res_coords[bboxnumber][1][1]
    right = res_coords[bboxnumber][3][0]
    bottom = res_coords[bboxnumber][3][1]
    damage = "Damage"
    return damage, left, top, right, bottom


# get all bboxes from coordinates in the picture
def create_all_bbox(res_coords):
    print("In create_all_bbox" , type(res_coords))
    bboxes = []
    for bbox in range(len(res_coords)):
        bboxes.append(allign_bbox(res_coords, bbox))
    return bboxes


def extract_bboxes(path):
    '''
    reshapes input image from path to 500x500
    returns bboxes in format:
    ('Damage', left, top, right, bottom)

    Example:

    ('Damage', 64, 277, 80, 435)

    '''
    # change of size and drawing fields
    img = cv2.imread(path)
    if img.size == 0:
        print('Error: Image was not read')
    img = cv2.resize(img, (500, 500))
    w, h, _ = img.shape
    cv2.rectangle(img, (0, 0), (w, h), (0, 0, 0), 0)
    img_area_limit = w * h * 0.2

    # getting and sorting contours
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_blur = cv2.bilateralFilter(img_rgb, d=7, sigmaSpace=75, sigmaColor=75)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
    a = img_gray.max()
    _, thresh = cv2.threshold(img_gray, a * 0.45, a, cv2.THRESH_BINARY_INV)
    image, contours_, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = []
    for cnt in contours_:
        area = cv2.contourArea(cnt)
        if area > img_area_limit:
            continue
        contours.append(cnt)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # grouping contours by placement
    LENGTH = len(contours)
    status = np.zeros((LENGTH, 1))
    for i, cnt1 in enumerate(contours):
        x = i
        if i != LENGTH - 1:
            for j, cnt2 in enumerate(contours[i + 1:]):
                x = x + 1
                dist = find_if_close(cnt1, cnt2)
                if dist:
                    val = min(status[i], status[x])
                    status[x] = status[i] = val
                else:
                    if status[x] == status[i]:
                        status[x] = i + 1
    unified = []
    maximum = int(status.max()) + 1
    for i in range(maximum):
        pos = np.where(status == i)[0]
        if pos.size != 0:
            cont = np.vstack(contours[i] for i in pos)
            hull = cv2.convexHull(cont)
            unified.append(hull)

    # processing and displaying the result
    res_coords = []
    for cnt in unified:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        res_coords.append(box)
    # refactor all bboxes for future IoU postprocessing
    final_bboxes = create_all_bbox(res_coords)
    print("In extract_bboxes", type(res_coords))
    return final_bboxes


def write_coordinate_file(filename, bboxes_of_image):
    """
    Write file with the name of the image input as .txt in the format:

    Damage left top right bottom

    Example:

    Damage 64 277 80 435

    """
    if len(bboxes_of_image) == 0:
        print("This file doesnt have a damage")

    f = open(filename, "w+")
    for bbox in bboxes_of_image:
        bbox_string = bbox[0] + " " + str(bbox[1]) + " " + str(bbox[2]) + " " + str(bbox[3]) + " " + str(bbox[4])
        f.write(bbox_string + "\n")
    f.close()


