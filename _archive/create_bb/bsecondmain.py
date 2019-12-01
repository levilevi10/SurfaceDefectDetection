import os
import time
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")


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


def crate_bb(path):
    img = cv2.imread(path)
    h, w, _ = img.shape
    n = 8
    img = cv2.addWeighted(img, 3, img, 0, 1)
    img = cv2.blur(img, (9, 9))
    img = cv2.resize(img, (int(w / n), int(h / n)))

    img = cv2.blur(img, (2, 2))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_blur = cv2.bilateralFilter(img_rgb, d=7, sigmaSpace=75, sigmaColor=75)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
    a = img_gray.max()
    _, thresh = cv2.threshold(img_gray, a * 0.45, a, cv2.THRESH_BINARY_INV)
    image, contours_, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    contours_ = sorted(contours_, key=cv2.contourArea, reverse=True)
    #COMMENT
    contours = []
    max_v = w / n * h / n * 0.2
    for cnt in contours_:
        area = cv2.contourArea(cnt)
        if area < 50 or area > max_v:
            continue
        contours.append(cnt)

    #COMMENT
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

    #COMMENT
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
        box = cv2.boundingRect(rect)
        box = np.int0(box)
        box = np.array([[b[0]*n, b[1]*n] for b in box])
        res_coords.append(box)
    return contours, res_coords


def draw_bb(res_coords, img_path, output_path):
    img = cv2.cv2.imread(img_path)
    for box in res_coords:
        cv2.drawContours(img, [box], -1, (0, 100, 255), 4)
        # break
    cv2.imwrite(output_path, img)


def save_bb(res_coords, img_name):
    with open(f"{os.path.basename(img_name)[:-4]}.txt", 'w', encoding='utf-8') as f:
        for idx, box in enumerate(res_coords):
            f.write(f"{idx} {box[0]} {box[1]} {box[2]} {box[3]}\n")


if __name__ == '__main__':
    start_time = time.time()
    name = 'Material-1-1-1-1_blur5_Otsu'
    INPUT_PATH = r'C:\Users\lvinzenz\Documents\Data\Image Recognition\SurfaceDefectDetection\LeoderBachelor\Example_Images_Surface_Defects\OtsuThresholding\Material-1-1-1-1_blur5_Otsu.jpg'
    OUTPUT_PATH_IMG = f'OUTPUT_{name}.jpg'

    contours, res_coords = crate_bb(INPUT_PATH)
    draw_bb(res_coords, INPUT_PATH, OUTPUT_PATH_IMG)
    save_bb(res_coords, INPUT_PATH)

    print(f"OK! Time: {round(time.time() - start_time, 2)}s")
