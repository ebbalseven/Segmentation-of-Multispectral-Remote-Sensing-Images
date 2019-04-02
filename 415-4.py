from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import cv2
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import numpy as np


def superfeature(segment, size, path):
    # image read for each band
    for i in range(4):
        # Load an images in grayscale
        # rgb and infrared bands
        bband = cv2.imread(path+str(1)+'.tif', 0)
        gband = cv2.imread(path+str(2)+'.tif', 0)
        rband = cv2.imread(path+str(3)+'.tif', 0)
        infraband = cv2.imread(path+str(4)+'.tif', 0)

        # bonus bands
        midinfrared = cv2.imread(path+str(5)+'.tif', 0)
        thermalinfrared = cv2.imread(path+str(6)+'.tif', 0)
        farinfrared = cv2.imread(path+str(7)+'.tif', 0)

        # merging channels to get rgb format
        colorformat = np.zeros((bband.shape[0], bband.shape[1], 3))
        colorformat[:, :, 0] = rband
        colorformat[:, :, 1] = gband
        colorformat[:, :, 2] = bband

        # converting lab format with rgbtolab
        labformat = cv2.cvtColor(colorformat.astype(np.uint8), cv2.COLOR_RGB2LAB)

    # converting float type before superpixel clustering
    colorformat = colorformat.astype(np.float)

    #  superpixel clustering
    superpixels = slic(colorformat, n_segments=segment, compactness=10, sigma=5, slic_zero=True)

    # feature matrix creating
    featurematrix = np.zeros((np.unique(superpixels).size, size))

    # for each superpixel creating feature vector
    for superpixel in np.unique(superpixels):
        # location feature
        superLocation = np.where(superpixels == superpixel)
        # only r,g,b feature vectors for part1
        featurematrix[superpixel][0] = np.average(rband[superLocation])
        featurematrix[superpixel][1] = np.average(gband[superLocation])
        featurematrix[superpixel][2] = np.average(bband[superLocation])

        # appending on location, infraredband and lab channel vectors for part2
        if size == 6 or size == 9 or size == 11 or size == 12:
            # i decrease the weight of the location feature by dividing it 2, it looks better
            featurematrix[superpixel][3] = np.average(superLocation)/2
            # i multiply by 2 infrared band, because it
            # makes the river and lake more pronounced
            featurematrix[superpixel][4] = np.average(infraband[superLocation])*2
            lChannel = labformat[:, :, 0]
            featurematrix[superpixel][5] = np.average(lChannel[superLocation])

        # appending extra bands for bonus part
        if size == 8 or size == 9 or size == 11 or size == 12:
            featurematrix[superpixel][6] = np.average(midinfrared[superLocation])
            featurematrix[superpixel][7] = np.average(farinfrared[superLocation])

        # appending extra band for lake and valley(iowa doesn't have thermal infrared band)
        if size == 9:
            featurematrix[superpixel][8] = np.average(thermalinfrared[superLocation])

        # increasing infrared weight, for example to make the river more pronounced
        if size == 12:
            featurematrix[superpixel][8] = np.average(infraband[superLocation])
            featurematrix[superpixel][9] = np.average(infraband[superLocation])
            # this is for iowa also,it makes location more important
            featurematrix[superpixel][11] = np.average(superLocation)

    # normalizing
    cv2.normalize(featurematrix, featurematrix, 0, 100, norm_type=cv2.NORM_MINMAX)
    return featurematrix, superpixels, colorformat


# description of arguments in order: number of superpixel segments, feature vector size, images path
featurematrix, superpixels, colorformat = superfeature(1500, 12, 'data/iowa/iowa-band')
featurematrix1, superpixels1, colorformat1 = superfeature(2500, 6, 'data/owens_valley/owens_valley-band')
featurematrix2, superpixels2, colorformat2 = superfeature(2500, 6, 'data/salt_lake/salt_lake-band')


def segment(colorformat, superpixels, featurematrix, clusternumber, imgnumber,superpixelnumber):
    segmentedimage = np.zeros_like(colorformat)
    # K-means clustering
    kmeans = KMeans(n_clusters=clusternumber, random_state=0).fit(featurematrix)
    boundry = np.zeros(colorformat.shape[:2])
    for superpixel in np.unique(superpixels):
        superlocation = np.where(superpixels == superpixel)
        segmentedimage[superlocation] = kmeans.labels_[superpixel]
        boundry[superlocation] = kmeans.labels_[superpixel]
    segmentedimage = cv2.normalize(segmentedimage, None, 0, 255, cv2.NORM_MINMAX)

    # random values to colorize image
    segmentedimage[:, :, 0] += 50
    segmentedimage[:, :, 1] += 100
    segmentedimage[:, :, 2] += 150

    colorformat = colorformat.astype(np.uint8)

    plt.figure('results', figsize=(18, 18))
    plt.subplot(223), plt.imshow(segmentedimage.astype(np.uint8))
    plt.title('image labels, '+ str(clusternumber) +'clusters'), plt.xticks([]), plt.yticks([])
    # segmented version with the boundries
    plt.subplot(224), plt.imshow(mark_boundaries(colorformat, boundry.astype(np.int)))
    plt.title('image segments'), plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(mark_boundaries(colorformat, superpixels))
    plt.title(str(superpixelnumber)+' superpixels'), plt.xticks([]), plt.yticks([])
    plt.subplot(221), plt.imshow(colorformat)
    plt.title('input'), plt.xticks([]), plt.yticks([])
    plt.savefig('results/segmented_image'+str(imgnumber)+'.jpg', bbox_inches='tight', pad_inches=0)
    plt.show()


# description of arguments in order:
# rgb image, superpixel segmented image, features, k-means cluster numbers, image number for plt.savefig, superpixels
segment(colorformat, superpixels, featurematrix, 10, 1, 1500)
segment(colorformat1, superpixels1, featurematrix1, 10, 2, 2500)
segment(colorformat2, superpixels2, featurematrix2, 10, 3, 2500)
