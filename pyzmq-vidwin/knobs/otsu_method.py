import cv2
from matplotlib.ticker import FuncFormatter
from matplotlib import pyplot as plt
import numpy as np


def otsu_implementation(img_title="/home/dhaval/piyush/NEW Evaluation/Fall Detection/Images/image+50.png", is_normalized=False, is_reduce_noise=False):
    # Read the image in a greyscale mode
    image = cv2.imread(img_title, 0)

    # Apply GaussianBlur to reduce image noise if it is required
    if is_reduce_noise:
        image = cv2.GaussianBlur(image, (5, 5), 0)

    # Set total number of bins in the histogram
    bins_num = 256

    # Get the image histogram
    hist, bin_edges = np.histogram(image, bins=bins_num)

    # Get normalized histogram if it is required
    if is_normalized:
        hist = np.divide(hist.ravel(), hist.max())

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)

    threshold = bin_mids[:-1][index_of_max_val]
    print("Otsu's algorithm implementation thresholding result: ", threshold)
    return threshold


def call_otsu_threshold(img_title="/home/dhaval/piyush/NEW Evaluation/Fall Detection/Images/image+1.png", is_reduce_noise=False):
    # Read the image in a greyscale mode
    image = cv2.imread(img_title, 0)

    # Apply GaussianBlur to reduce image noise if it is required
    if is_reduce_noise:
        image = cv2.GaussianBlur(image, (5, 5), 0)

    # View initial image histogram
    plt.hist(image.ravel(), 256)
    plt.xlabel('Colour intensity')
    plt.ylabel('Number of pixels')
    plt.show()
    #plt.savefig("image_hist.png")
    plt.close()

    # Applying Otsu's method setting the flag value into cv.THRESH_OTSU.
    # Use bimodal image as an input.
    # Optimal threshold value is determined automatically.
    otsu_threshold, image_result = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    print("Obtained threshold: ", otsu_threshold)

    # View the resulting image histogram
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(image_result.ravel(), 256)
    ax.set_xlabel('Colour intensity')
    ax.set_ylabel('Number of pixels')
    # Get rid of 1e7
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: ('%1.1fM') % (x*1e-6)))
    #plt.savefig("image_hist_result.png")
    plt.close()

    # Visualize the image after the Otsu's method application
    cv2.imshow("Otsu's thresholding result", image_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    #call_otsu_threshold()
    otsu_implementation()


if __name__ == "__main__":
    main()