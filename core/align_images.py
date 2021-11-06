import numpy as np
import imutils
import cv2

def align_images(image, template, max_features=500, keep_percent=0.2, debug=False):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    orb = cv2.ORB_create(max_features)
    kps_a, descs_a = orb.detectAndCompute(image_gray, None)
    kps_b, descs_b = orb.detectAndCompute(template_gray, None)
    
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descs_a, descs_b, None)
    
    matches = sorted(matches, key=lambda x:x.distance)
    
    keep = int(len(matches) * keep_percent)
    matches = matches[:keep]
    
    if debug:
        matched_vis = cv2.drawMatches(image, kps_a, template, kps_b, matches, None)
        matched_vis = imutils.resize(matched_vis, width=1000)
        cv2.imshow("Matched Keypoints", matched_vis)
        cv2.waitKey(0)
        
    pts_a = np.zeros((len(matches), 2), dtype="float")
    pts_b = np.zeros((len(matches), 2), dtype="float")
    
    for i, m in enumerate(matches):
        pts_a[i] = kps_a[m.queryIdx].pt
        pts_b[i] = kps_b[m.trainIdx].pt
        
    H, mask = cv2.findHomography(pts_a, pts_b, method=cv2.RANSAC)
    
    h, w = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))
    
    return aligned
