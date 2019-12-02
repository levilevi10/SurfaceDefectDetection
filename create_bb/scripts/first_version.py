import time
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")


def find_if_close(cnt1, cnt2):
    LIMIT = 30
    row1, row2 = cnt1.shape[0], cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i] - cnt2[j])
            if abs(dist) < LIMIT:
                return True
            elif i == row1 - 1 and j == row2 - 1:
                return False


def main(path):
    # change of size and drawing fields
    img = cv2.imread(path)
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
        # print(box)
        res_coords.append(box)
        cv2.drawContours(img, [box], 0, (0, 0, 255), 1)
    return img, res_coords


if __name__ == '__main__':
    start_time = time.time()

    INPUT_PATH = '/home/sashatr/Desktop/projects/skripts/crop_obj/OtsuThresholding/Material-1-1-1-1_Otsu.jpg'
    OUTPUT_PATH = 'output2.jpg'

    result_img, boundbox = main(INPUT_PATH)
    cv2.imwrite(OUTPUT_PATH, result_img)
    print(f"OK! Time: {round(time.time() - start_time, 2)}s")
