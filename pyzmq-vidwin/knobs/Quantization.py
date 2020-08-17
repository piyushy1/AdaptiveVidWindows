from PIL import Image
import datetime
from sys import getsizeof
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np

if __name__ == "__main__":
    im = Image.open("/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/Test Images/car.jpg")
    time1 = datetime.datetime.now()
    #0 media, 1, max convergenc and 2 - fast octree
    im1 = np.asarray(im.resize((224, 224), Image.ANTIALIAS))
    im2 = im.quantize(200,0)
    im3 = np.asarray(im2.resize((224,224), Image.ANTIALIAS))
    time2 = datetime.datetime.now()
    delta = time2 - time1
    execution_time = int(delta.total_seconds() * 1000)
    print(execution_time)
    print('Size of Original Image', getsizeof(im))
    print('Size of Quantize Image', getsizeof(im2))
    print('Size of Quantize Image', getsizeof(im3))
    im.show()
    im2.show()


    model = MobileNetV2(input_shape=None, alpha=1.0, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)

    # mobilenet model core
    img_path = '/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/Test Images/car.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    #without quantization core
    y = image.img_to_array(im1)
    y = np.expand_dims(y, axis=0)
    y = preprocess_input(y)
    #
    # with quantization
    z = image.img_to_array(im3)
    z = np.expand_dims(z, axis=0)
    z = preprocess_input(z)

    time1 = datetime.datetime.now()
    preds1 = model.predict(x)
    time2 = datetime.datetime.now()
    delta = time2 - time1
    execution_time1 = int(delta.total_seconds() * 1000)

    time1 = datetime.datetime.now()
    preds2 = model.predict(y)
    time2 = datetime.datetime.now()
    delta = time2 - time1
    execution_time2 = int(delta.total_seconds() * 1000)

    time1 = datetime.datetime.now()
    preds3 = model.predict(z)
    time2 = datetime.datetime.now()
    delta = time2 - time1
    execution_time3 = int(delta.total_seconds() * 1000)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('Predicted:', decode_predictions(preds1, top=3)[0], execution_time1)
    print('Predicted:', decode_predictions(preds2, top=3)[0],execution_time2)
    print('Predicted:', decode_predictions(preds3, top=3)[0], execution_time3)