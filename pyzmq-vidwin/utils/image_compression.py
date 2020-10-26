import cv2
import sys
import numpy as np
from sys import getsizeof

# https://pypi.org/project/memory-profiler/

img=cv2.imread('/home/dhaval/Desktop/Car-Image.jpg', 1)
print('Image Size', getsizeof(img))
#encode to jpeg format
#encode param image quality 0 to 100. default:95
#if you want to shrink data size, choose low image quality.

encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),10]
encode_param1=[int(cv2.IMWRITE_JPEG2000_COMPRESSION_X1000),500]
encode_param2=[int(cv2.IMWRITE_PNG_COMPRESSION),8]

result,encimg=cv2.imencode('.jpg',img,encode_param)
result1,encimg1=cv2.imencode('.jpg',img,encode_param1)
result2,encimg2=cv2.imencode('.jpg',img,encode_param2)

print('JPEG Compression Image Size', getsizeof(encimg))
print('JPEG200 Compression Image Size', getsizeof(encimg1))
print('PNG Compression Image Size', getsizeof(encimg2))

if False==result:
    print('could not encode image!')
    quit()

#decode from jpeg format
decimg=cv2.imdecode(encimg,1)
decimg1=cv2.imdecode(encimg1,1)
decimg2=cv2.imdecode(encimg2,1)
print('DecCompression JPEGImage Size', getsizeof(decimg),np.array(decimg).nbytes)
print('DecCompression JPEG2000Image Size', getsizeof(decimg))
print('DecCompression PNGImage Size', getsizeof(decimg))

#cv2.imshow('Source Image',img)
# cv2.imshow('JPEG Decoded image',decimg)
# cv2.imshow('JPEG 2000 Decoded image',decimg1)
# cv2.imshow('PNG Decoded image',decimg2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()