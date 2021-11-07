from core import align_images
from collections import namedtuple
import pytesseract
import argparse
import imutils
import cv2

def cleanup_text(text):
    #Strip Non-ASCII characters because OpenCV can't show it
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="scans/scan_01.jpg", help="Path to input image")
ap.add_argument("-t", "--template", default="form_w4.png", help="Path to input template image")
args = vars(ap.parse_args())

OCRLocation = namedtuple("OCRLocation", ["id", "bbox", "filter_keywords"])

OCR_LOCATIONS = [
    OCRLocation("step1_first_name", (265, 237, 751, 106), ["middle", "initial", "first", "name"]),
    OCRLocation("step1_last_name", (1020, 237, 835, 106), ["last", "name"]),
    OCRLocation("step1_address", (265, 336, 1588, 106), ["address"]),
    OCRLocation("step1_city_state_zip", (265, 436, 1588, 106), ["city", "zip", "town", "state"]),
    OCRLocation("step5_employee_signature", (319, 2516, 1487, 156), ["employee", "signature", "form", "valid", "unless", "you", "sign"]),
    OCRLocation("step5_date", (1804, 2516, 504, 156), ["date"]),
    OCRLocation("employee_name_address", (265, 2706, 1224, 180), ["employer", "name", "address"]),
    OCRLocation("employee_ein", (1831, 2706, 448, 180), ["employer", "identification", "number", "ein"]),
]

image = cv2.imread(args["image"])
template = cv2.imread(args["template"])

print("[INFO] Aligning Images...")
aligned = align_images(image, template)

print("[INFO] OCR'ing Document...")
parsing_results = []

for loc in OCR_LOCATIONS:
    x, y, w, h = loc.bbox
    roi = aligned[y:y+h, x:x+w]
    
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(rgb)
    
    for line in text.split("\n"):
        if len(line) == 0:
            continue
        
        lower = line.lower()
        count = sum([lower.count(x) for x in loc.filter_keywords])
        
        if count == 0:
            parsing_results.append((loc, line))
            
results = {}

for loc, line in parsing_results:
    r = results.get(loc.id, None)
    
    if r is None:
        results[loc.id] = (line, loc._asdict())
        
    else:
        existing_text, loc = r
        text = "{}\n{}".format(existing_text, line)
        
        results[loc["id"]] = (text, loc)
        
for loc_id, result in results.items():
    text, loc = result
    
    print(loc["id"])
    print("=" * len(loc["id"]))
    print("{}\n\n".format(text))
    
    x, y, w, h = loc["bbox"]
    clean_text = cleanup_text(text)
    
    cv2.rectangle(aligned, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    for i, line in enumerate(text.split("\n")):
        start_y = y + (i * 70) + 40
        cv2.putText(aligned, line, (x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 5)
        
cv2.imshow("Input", imutils.resize(image, width=700))
cv2.imshow("Output", imutils.resize(aligned, width=700))
cv2.waitKey(0)
