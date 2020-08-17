# import the necessary packages
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import argparse
import cv2
from sys import getsizeof
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import datetime
import matplotlib.pyplot as plt

latency_list = []
final_original_image_size = []
final_quant_img_size =[]
accuracy = []

def perform_quantization(image, cluster):
    orgsize = getsizeof(image)
    time1 = datetime.datetime.now()
    # load the image and grab its width and height
    #image = cv2.imread('/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/Test Images/car.jpg')
    (h, w) = image.shape[:2]
    # convert the image from the RGB color space to the L*a*b*
    # color space -- since we will be clustering using k-means
    # which is based on the euclidean distance, we'll use the
    # L*a*b* color space where the euclidean distance implies
    # perceptual meaning
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # reshape the image into a feature vector so that k-means
    # can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters = cluster)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    image = image.reshape((h, w, 3))
    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    time2 = datetime.datetime.now()
    delta = time2 - time1
    execution_time = int(delta.total_seconds() * 1000)
    final_size = getsizeof(quant)
    return quant, execution_time, orgsize, final_size

def stream_video(video, cluster):
    i = 1
    cluster_latency = []
    original_image_size = []
    quant_img_size = []
    # early loading of the model
    model = MobileNetV2(input_shape=None, alpha=1.0, include_top=True, weights='imagenet', input_tensor=None,
                        pooling=None, classes=1000)

    video = cv2.VideoCapture(video)
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print('total frames', total)
    while video.isOpened():
        ret1, frame = video.read()

        if i == 42:
            # get the latency for all similarity algo.
            latency_list.append(cluster_latency)
            final_original_image_size.append(original_image_size)
            final_quant_img_size.append(quant_img_size)
            break

        else:
            quant_img, latency, orgsize, quantsize =  perform_quantization(frame,cluster)
            cluster_latency.append(latency)
            quant_img_size.append(quantsize)
            original_image_size.append(orgsize)
            # resize the image as mobilenet take 224,224. perform early or late resizing
            quant_img_resize = cv2.resize(quant_img, (224, 224), interpolation=cv2.INTER_AREA)
            x = np.expand_dims(quant_img_resize, axis=0)
            x = preprocess_input(x)
            #preds1 = model.predict(x)
            #print('Predicted:', decode_predictions(preds1, top=3)[0])
        i = i+1
        print('Frame: ', i)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            break

    video.release()
    cv2.destroyAllWindows()

def line_plot(latency_plot):
    markers = ['o', '*', '^','1', '2', '3']
    cnt = 0
    for i in latency_plot:
        plt.plot(i, marker = markers[cnt] )
        cnt = cnt+1

    plt.show()


if __name__ == "__main__":
    cluster = [10,32,64,100,150,200]

    #cluster = [10,64]
    for i in cluster:
        stream_video('/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/test3.mp4', i)
    line_plot(latency_list)
    line_plot(final_quant_img_size)
    line_plot(final_original_image_size)




# image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
# image1 = cv2.resize(image, (224,224), interpolation=cv2.INTER_AREA)
# quant1 = cv2.resize(quant, (224,224), interpolation=cv2.INTER_AREA)
# print('Quantize Image SIze:', getsizeof(quant1))
# print('Original Image SIze:', getsizeof(image1))
# model = MobileNetV2(input_shape=None, alpha=1.0, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
# x = np.expand_dims(image1, axis=0)
# x = preprocess_input(x)
# preds1 = model.predict(x)
# print('Predicted:', decode_predictions(preds1, top=3)[0])
# # display the images and wait for a keypress
# #cv2.imshow("image", np.hstack([image1, quant1]))
# #cv2.waitKey(0)