from core import align_images
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="scans/scan_01.jpg", help="Path to input image")
ap.add_argument("-t", "--template", default="form_w4.png", help="Path to input template image")
args = vars(ap.parse_args())

print("[INFO] Loading Images...")
image = cv2.imread(args["image"])
template = cv2.imread(args["template"])

print("[INFO] Aligning Images...")
aligned = align_images(image, template, debug=True)

aligned = imutils.resize(aligned, width=700)
template = imutils.resize(template, width=700)

stacked = np.hstack([aligned, template])

overlay = template.copy()
output = aligned.copy()
cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)

cv2.imshow("Image Alignment Stacked", stacked)
cv2.imshow("Image Alignment Overlay", output)
cv2.waitKey(0)
