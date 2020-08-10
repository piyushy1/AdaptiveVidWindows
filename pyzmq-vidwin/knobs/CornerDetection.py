import numpy as np
import cv2 as cv
from scipy.spatial.distance import cdist
filename = 'chessboard.png'
img = cv.imread('/home/dhaval/piyush/NEW Evaluation/Fall Detection/Images/image+1.png')
img1 = cv.imread('/home/dhaval/piyush/NEW Evaluation/Fall Detection/Images/image+20.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)

gray1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
gray1 = np.float32(gray1)
dst1 = cv.cornerHarris(gray1,2,3,0.04)

dst_f = np.array(dst).flatten()
dst1_f = np.array(dst1).flatten()


cos_sim = np.dot(dst_f, dst1_f)/(np.linalg.norm(dst_f)*np.linalg.norm(dst1_f))

print('the distance is ', cos_sim)

#result is dilated for marking the corners, not important
dst = cv.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
cv.imshow('dst',img)
cv.imshow('dst1',img1)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()