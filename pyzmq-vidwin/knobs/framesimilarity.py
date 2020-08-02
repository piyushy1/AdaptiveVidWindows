from PIL import Image
import datetime
import imagehash
import cv2
from scipy.stats import wasserstein_distance
from imagededup.methods import CNN
from imagededup.utils.image_utils import load_image, preprocess_image
import numpy
import time
import matplotlib.pyplot as plt
#from memory_profiler import profile


time_list = []

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
    # hash1 = imagehash.average_hash(Image.open(img1))
    # hash2 = imagehash.average_hash(Image.open(img2))
    hash1 = imagehash.average_hash(img1)
    hash2 = imagehash.average_hash(img2)
    distance = hash1-hash2
    time2 = datetime.datetime.now()
    delta = time2 - time1
    execution_time = int(delta.total_seconds() * 1000)
    print('Average hash_Normal_Distance: ', distance , execution_time)
    return distance


def average_hash_distance1(img1, img2):
    time1 = datetime.datetime.now()
    # hash1 = str(imagehash.average_hash(Image.open(img1)))
    # hash2 = str(imagehash.average_hash(Image.open(img2)))
    hash1 = str(imagehash.average_hash(img1))
    hash2 = str(imagehash.average_hash(img2))
    distance = hamming_hash_distance(hash1, hash2)
    time2 = datetime.datetime.now()
    delta = time2 - time1
    execution_time = int(delta.total_seconds() * 1000)
    print('Average hash_Hamming_Distance: ',distance , execution_time)
    return distance

def difference_hash_distance(img1, img2):
    time1 = datetime.datetime.now()
    # hash1 = imagehash.dhash(Image.open(img1))
    # hash2 = imagehash.dhash(Image.open(img2))
    hash1 = imagehash.dhash(img1)
    hash2 = imagehash.dhash(img2)
    distance = hash1-hash2
    time2 = datetime.datetime.now()
    delta = time2 - time1
    execution_time = int(delta.total_seconds() * 1000)
    print('Difference hash_Normal_Distance: ',distance , execution_time)
    return distance

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
    # hash1 = imagehash.whash(Image.open(img1))
    # hash2 = imagehash.whash(Image.open(img2))
    hash1 = imagehash.whash(img1)
    hash2 = imagehash.whash(img2)
    distance = hash1-hash2
    time2 = datetime.datetime.now()
    delta = time2 - time1
    execution_time = int(delta.total_seconds() * 1000)
    print('Wavelet hash_Normal_Distance: ',distance , execution_time)
    return distance, execution_time

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
    #hash1 = imagehash.phash(Image.open(img1))
    #hash2 = imagehash.phash(Image.open(img2))
    hash1 = imagehash.phash(img1)
    hash2 = imagehash.phash(img2)
    distance = hash1-hash2
    #distance = wasserstein_distance(hash1, hash2)
    time2 = datetime.datetime.now()
    delta = time2 - time1
    execution_time = int(delta.total_seconds() * 1000)
    print('Perceptual hash_Normal_Distance: ',distance , execution_time)
    return distance, execution_time

def perceptual_hash_distance1(img1, img2):
    time1 = datetime.datetime.now()
    # hash1 = str(imagehash.phash(Image.open(img1)))
    # hash2 = str(imagehash.phash(Image.open(img2)))
    hash1 = str(imagehash.phash(img1))
    hash2 = str(imagehash.phash(img2))
    distance = wasserstein_distance(hash1, hash2)
    time2 = datetime.datetime.now()
    delta = time2 - time1
    execution_time = int(delta.total_seconds() * 1000)
    print('Perceptual hash_Hamming_Distance: ', distance , execution_time)
    return distance

###################################################################################
################# HISTOGRAM BASED SIMILARITY METHODS ##############################
###################################################################################

# function to get distance between histograms of two frames.
#@profile(precision=4)
def Histogram_frame_distance(frame1, frame2):
    time1 = datetime.datetime.now()
    #frame1 = cv2.imread(frame1)
    #frame2 = cv2.imread(frame2)
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
    return dist_value

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

# opencv image to PIL format
def CVtoPILformat(img):
    # You may need to convert the color.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil

# function to stream video to get frame distance of different similarity algo.
def stream_video(video):
    video = cv2.VideoCapture(video)
    # video.set(cv2.CAP_PROP_FPS, int(5))
    print("Frame rate : {0}".format(video.get(cv2.CAP_PROP_FPS)))

    # temporary block to hold first frame
    temp_block = []
    counter = []
    i = 1

    while video.isOpened():

        ret1, frame = video.read()
        # for hash need to convert to PIL format

        frame = CVtoPILformat(frame)

        if temp_block == []:
            temp_block.append(frame)
            counter.append(i)

        else:
            # print(new_frame_data[0].shape)

            # if len(counter) != 0:
            print('Frame distance', counter[0], i)

            # dist = difference_hash_distance(frame, temp_block[0])
            dist, latency = wavelet_hash_distance(frame, temp_block[0])
            #dist, latency = perceptual_hash_distance(frame, temp_block[0])
            #dist = Histogram_frame_distance(frame, temp_block[0])
            #dist = average_hash_distance(frame, temp_block[0])
            time_list.append(latency)
            if dist < 0.97:
                print("New block found separate - len ", len(temp_block))
                temp_block.clear()
                counter.clear()
            else:
                temp_block.append(frame)

        i = i + 1

        #cv2.imshow("Changed", frame)

        time.sleep(1/video.get(cv2.CAP_PROP_FPS))

        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            break

    video.release()
    cv2.destroyAllWindows()


def violinplot(data, x_pos):
    # fake data
    # pos = [1, 2, 4, 5, 7, 8]
    # data = [np.random.normal(0, std, size=100) for std in pos]
    # print(data)
    # fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10, 6))
    plt.violinplot(data, x_pos, points=20, widths=0.3,
                   showmeans=True, showextrema=True, showmedians=True)
    plt.title('Custom violinplot 1', fontsize=10)

    plt.show()


def boxplot():
    print('write code of box plot')

if __name__ == "__main__":
    #video_path = "/home/dhaval/piyush/Usecases_dataset/P3_car_left_car.mp4"
    video_path = "/home/dhaval/piyush/Usecases_dataset/fall_detection/Lecture room/Videos/video (1).avi"
    stream_video(video_path)
    violinplot(time_list, ['PPhash'])





    #average_hash_distance('/home/dhaval/piyush/NEW Evaluation/images/image+1.png', '/home/dhaval/piyush/NEW Evaluation/images/image+100.png')
    # average_hash_distance1('/home/dhaval/piyush/NEW Evaluation/images/image+1.png', '/home/dhaval/piyush/NEW Evaluation/images/image+100.png')
    #difference_hash_distance('/home/dhaval/piyush/NEW Evaluation/images/image+1.png', '/home/dhaval/piyush/NEW Evaluation/images/image+100.png')
    # difference_hash_distance1('/home/dhaval/piyush/NEW Evaluation/images/image+1.png', '/home/dhaval/piyush/NEW Evaluation/images/image+100.png')
    #wavelet_hash_distance('/home/dhaval/piyush/NEW Evaluation/images/image+1.png', '/home/dhaval/piyush/NEW Evaluation/images/image+100.png')
    # wavelet_hash_distance1('/home/dhaval/piyush/NEW Evaluation/images/image+1.png', '/home/dhaval/piyush/NEW Evaluation/images/image+100.png')
    # perceptual_hash_distance('/home/dhaval/piyush/NEW Evaluation/images/image+1.png', '/home/dhaval/piyush/NEW Evaluation/images/image+100.png')
    #perceptual_hash_distance1('/home/dhaval/piyush/NEW Evaluation/images/image+1.png', '/home/dhaval/piyush/NEW Evaluation/images/img00124.jpg')
    # Histogram_frame_distance('/home/dhaval/piyush/NEW Evaluation/images/image+1.png', '/home/dhaval/piyush/NEW Evaluation/images/image+100.png')
    # CNN_distance('/home/dhaval/piyush/NEW Evaluation/images/image+1.png', '/home/dhaval/piyush/NEW Evaluation/images/image+100.png')

