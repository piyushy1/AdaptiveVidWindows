from PIL import Image
import datetime
import imagehash
import cv2
from scipy.stats import wasserstein_distance
from imagededup.methods import CNN
from imagededup.utils.image_utils import load_image, preprocess_image
import numpy
#from memory_profiler import profile


# tutorial to look for explanation https://fullstackml.com/wavelet-image-hash-in-python-3504fdd282b5

# function to calculate average_hash distance (hamming distance with hash size 8)

###################################################################################
################# HASHING BASED SIMILARITY METHODS ################################
###################################################################################

#@profile(precision=4)
def hamming_hash_distance(left_hash, right_hash):
    """Compute the hamming distance between two hashes"""
    if len(left_hash) != len(right_hash):
        raise ValueError('Hamming distance requires two strings of equal length')

    return sum(map(lambda x: 0 if x[0] == x[1] else 1, zip(left_hash, right_hash)))

def average_hash_distance(img1, img2):
    time1 = datetime.datetime.now()
    hash1 = imagehash.average_hash(Image.open(img1))
    hash2 = imagehash.average_hash(Image.open(img2))
    distance = hash1-hash2
    time2 = datetime.datetime.now()
    delta = time2 - time1
    execution_time = int(delta.total_seconds() * 1000)
    print('Average hash_Normal_Distance: ', distance , execution_time)


def average_hash_distance1(img1, img2):
    time1 = datetime.datetime.now()
    hash1 = str(imagehash.average_hash(Image.open(img1)))
    hash2 = str(imagehash.average_hash(Image.open(img2)))
    distance = hamming_hash_distance(hash1, hash2)
    time2 = datetime.datetime.now()
    delta = time2 - time1
    execution_time = int(delta.total_seconds() * 1000)
    print('Average hash_Hamming_Distance: ',distance , execution_time)

def difference_hash_distance(img1, img2):
    time1 = datetime.datetime.now()
    hash1 = imagehash.dhash(Image.open(img1))
    hash2 = imagehash.dhash(Image.open(img2))
    distance = hash1-hash2
    time2 = datetime.datetime.now()
    delta = time2 - time1
    execution_time = int(delta.total_seconds() * 1000)
    print('Difference hash_Normal_Distance: ',distance , execution_time)

def difference_hash_distance1(img1, img2):
    time1 = datetime.datetime.now()
    hash1 = str(imagehash.dhash(Image.open(img1)))
    hash2 = str(imagehash.dhash(Image.open(img2)))
    distance = hamming_hash_distance(hash1, hash2)
    time2 = datetime.datetime.now()
    delta = time2 - time1
    execution_time = int(delta.total_seconds() * 1000)
    print('Difference hash_Hamming_Distance: ', distance , execution_time)

def wavelet_hash_distance(img1, img2):
    time1 = datetime.datetime.now()
    hash1 = imagehash.whash(Image.open(img1))
    hash2 = imagehash.whash(Image.open(img2))
    distance = hash1-hash2
    time2 = datetime.datetime.now()
    delta = time2 - time1
    execution_time = int(delta.total_seconds() * 1000)
    print('Wavelet hash_Normal_Distance: ',distance , execution_time)

def wavelet_hash_distance1(img1, img2):
    time1 = datetime.datetime.now()
    hash1 = str(imagehash.whash(Image.open(img1)))
    hash2 = str(imagehash.whash(Image.open(img2)))
    distance = hamming_hash_distance(hash1, hash2)
    time2 = datetime.datetime.now()
    delta = time2 - time1
    execution_time = int(delta.total_seconds() * 1000)
    print('Wavelet hash_Hamming_Distance: ', distance , execution_time)

def perceptual_hash_distance(img1, img2):
    time1 = datetime.datetime.now()
    hash1 = imagehash.phash(Image.open(img1))
    hash2 = imagehash.phash(Image.open(img2))
    distance = hash1-hash2
    #distance = wasserstein_distance(hash1, hash2)
    time2 = datetime.datetime.now()
    delta = time2 - time1
    execution_time = int(delta.total_seconds() * 1000)
    print('Perceptual hash_Normal_Distance: ',distance , execution_time)

def perceptual_hash_distance1(img1, img2):
    time1 = datetime.datetime.now()
    hash1 = str(imagehash.phash(Image.open(img1)))
    hash2 = str(imagehash.phash(Image.open(img2)))
    distance = hamming_hash_distance(hash1, hash2)
    time2 = datetime.datetime.now()
    delta = time2 - time1
    execution_time = int(delta.total_seconds() * 1000)
    print('Perceptual hash_Hamming_Distance: ', distance , execution_time)

###################################################################################
################# HISTOGRAM BASED SIMILARITY METHODS ##############################
###################################################################################

# function to get distance between histograms of two frames.
#@profile(precision=4)
def Histogram_frame_distance(frame1, frame2):
    time1 = datetime.datetime.now()
    frame1 = cv2.imread(frame1)
    frame2 = cv2.imread(frame2)
    hsv_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    hsv_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    h_bins = 50
    s_bins = 60
    histSize = [h_bins, s_bins]
    # hue varies from 0 to 179, saturation from 0 to 255
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges # concat lists
    # Use the 0-th and 1-st channels
    channels = [0, 1]
    hist_frame1 = cv2.calcHist([hsv_frame1], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(hist_frame1, hist_frame1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hist_frame2 = cv2.calcHist([hsv_frame2], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(hist_frame2, hist_frame2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    ## https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html

    # four compare methods corr, chi, intersect and bhattacharya [0,1,2,3]
    dist_value = cv2.compareHist(hist_frame1, hist_frame2, 0)
    time2 = datetime.datetime.now()
    delta = time2 - time1
    execution_time = int(delta.total_seconds() * 1000)
    print('Histogram_Distance: ', dist_value , execution_time)

###################################################################################
################# CNN BASED COSINE SIMILARITY METHODS ############################~
###################################################################################


def CNN_distance(image_file1, image_file2, target_size = (224, 224)):
    time1 = datetime.datetime.now()
    image_array1 = load_image(image_file=image_file1, target_size=target_size, grayscale=False)
    image_array2 = load_image(image_file=image_file2, target_size=target_size, grayscale=False)
    cnn = CNN()
    # generate the CNN based encoding. The CNN model used is mobilenet where last layer is replaced with global average pooling.
    encoding1 = cnn._get_cnn_features_single(image_array1)
    encoding2 = cnn._get_cnn_features_single(image_array2)
    # calculate the cosine distance
    cos_distance = numpy.dot(encoding1[0], encoding2[0]) / (numpy.sqrt(numpy.dot(encoding1[0], encoding1[0])) * numpy.sqrt(numpy.dot(encoding2[0], encoding2[0])))
    time2 = datetime.datetime.now()
    delta = time2 - time1
    execution_time = int(delta.total_seconds() * 1000)
    print('CNN_Cosine_Distance: ', cos_distance , execution_time)


if __name__ == "__main__":
    video = cv2.VideoCapture("/home/dhaval/piyush/Usecases_dataset/P3_car_left_car.mp4");
    #video.set(cv2.CAP_PROP_FPS, int(5))
    print("Frame rate : {0}".format(video.get(cv2.CAP_PROP_FPS)))

    while video.isOpened():
        ret1, frame2 = video.read()
        cv2.imshow("Changed", frame2)

        if cv2.waitKey(int(1000/video.get(cv2.CAP_PROP_FPS))) & 0xFF == ord('q'):  # press q to quit
            break

    video.release()
    cv2.destroyAllWindows()


    # average_hash_distance('/home/dhaval/piyush/NEW Evaluation/images/image+1.png', '/home/dhaval/piyush/NEW Evaluation/images/image+100.png')
    # average_hash_distance1('/home/dhaval/piyush/NEW Evaluation/images/image+1.png', '/home/dhaval/piyush/NEW Evaluation/images/image+100.png')
    # difference_hash_distance('/home/dhaval/piyush/NEW Evaluation/images/image+1.png', '/home/dhaval/piyush/NEW Evaluation/images/image+100.png')
    # difference_hash_distance1('/home/dhaval/piyush/NEW Evaluation/images/image+1.png', '/home/dhaval/piyush/NEW Evaluation/images/image+100.png')
    # wavelet_hash_distance('/home/dhaval/piyush/NEW Evaluation/images/image+1.png', '/home/dhaval/piyush/NEW Evaluation/images/image+100.png')
    # wavelet_hash_distance1('/home/dhaval/piyush/NEW Evaluation/images/image+1.png', '/home/dhaval/piyush/NEW Evaluation/images/image+100.png')
    # perceptual_hash_distance('/home/dhaval/piyush/NEW Evaluation/images/image+1.png', '/home/dhaval/piyush/NEW Evaluation/images/image+100.png')
    # perceptual_hash_distance1('/home/dhaval/piyush/NEW Evaluation/images/image+1.png', '/home/dhaval/piyush/NEW Evaluation/images/image+100.png')
    # Histogram_frame_distance('/home/dhaval/piyush/NEW Evaluation/images/image+1.png', '/home/dhaval/piyush/NEW Evaluation/images/image+100.png')
    # CNN_distance('/home/dhaval/piyush/NEW Evaluation/images/image+1.png', '/home/dhaval/piyush/NEW Evaluation/images/image+100.png')

