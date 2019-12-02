import os
import time
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")


def non_max_suppression_fast(boxes, overlapThresh):
    boxes = np.array(boxes, dtype=np.float32)
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")


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
    img1 = cv2.resize(img, (int(w / n), int(h / n)))

    img = cv2.blur(img1, (2, 2))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_blur = cv2.bilateralFilter(img_rgb, d=7, sigmaSpace=75, sigmaColor=75)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
    a = img_gray.max()
    _, thresh = cv2.threshold(img_gray, a * 0.45, a, cv2.THRESH_BINARY_INV)
    image, contours_, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    contours_ = sorted(contours_, key=cv2.contourArea, reverse=True)
    contours = []
    max_v = w / n * h / n * 0.3
    for cnt in contours_:
        area = cv2.contourArea(cnt)
        if area < 50 or area > max_v:
            continue
        contours.append(cnt)

    LENGTH = len(contours)
    if LENGTH is 0:
        raise ValueError

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

    res = []
    for u in unified:
        x, y, w, h = cv2.boundingRect(u)
        res.append([x, y, x + w, y + h])

    res = non_max_suppression_fast(res, 0.0001)
    return contours, res


def draw_bb(res_coords, img_path, output_path):
    if res_coords == []:
        return
    img = cv2.cv2.imread(img_path)
    for box in res_coords:
        x, y, x_w, y_h = box
        cv2.rectangle(img, (x*8, y*8), (x_w*8, y_h*8), (0, 255, 0), 4)
    cv2.imwrite(output_path, img)


def save_bb(res_coords, img_name):
    if res_coords == []:
        return
    with open(f"../output_txt/{os.path.basename(img_name)[:-4]}.txt", 'w', encoding='utf-8') as f:
        for box in res_coords:
            f.write(f"Damage 1.00 {box[0]*8} {box[1]*8} {box[2]*8} {box[3]*8}\n")
def save_empty_bb(res_coords, img_name):
    if res_coords == []:
        return
    with open(f"../output_txt/{os.path.basename(img_name)[:-4]}.txt", 'w', encoding='utf-8') as f:
        for box in res_coords:
            f.write(f"Damage 1.00 0 0 0 0\n")


if __name__ == '__main__':
    start_time = time.time()

    files = os.listdir('../OtsuThresholding')
    for name in files:
        INPUT_PATH = f'../OtsuThresholding/{name}'
        OUTPUT_PATH_IMG = f'../output_img/{name}'
        try:
            contours, res_coords = crate_bb(INPUT_PATH)
            draw_bb(res_coords, INPUT_PATH, OUTPUT_PATH_IMG)
            save_bb(res_coords, INPUT_PATH)
            print(f'{name} -> result saved.')
        except ValueError:
            save_empty_bb(res_coords, INPUT_PATH)
            print(f'{name} -> boundbox not found.')

    print(f"OK! Time: {round(time.time() - start_time, 2)}s")
