from cv2.typing import NumPyArrayNumeric
import numpy as np
import cv2 as cv
import os

clear = lambda: os.system('cls')
clear()

object_img: NumPyArrayNumeric = cv.imread('cv02_vzor_hrnecek.bmp')
object_hsv: NumPyArrayNumeric = cv.cvtColor(object_img, cv.COLOR_BGR2HSV)
mask: NumPyArrayNumeric = cv.inRange(object_hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist: NumPyArrayNumeric = cv.calcHist([object_hsv], [0], mask, [180], [0, 180])
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

cap = cv.VideoCapture('cv02_hrnecek.mp4')
ret, frame = cap.read()

res: NumPyArrayNumeric = cv.matchTemplate(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), cv.cvtColor(object_img, cv.COLOR_BGR2GRAY), cv.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

x, y = max_loc
h, w = object_img.shape[:2]
track_window: tuple[int, int, int, int] = (x, y, w, h)

term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    ret, track_window = cv.CamShift(dst, track_window, term_crit)
    x,y,w,h = track_window
    img2 = cv.rectangle(frame, (x,y), (x+w,y+h), 255,2)
    cv.imshow('Image', img2)

    if cv.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
