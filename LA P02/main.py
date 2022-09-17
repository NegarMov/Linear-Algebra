from utils import *
import numpy as np


def warpPerspective(img, transform_matrix, output_width, output_height):
    width, height, _ = img.shape
    res = np.zeros((output_width, output_height, _), dtype='int')

    for i in range(width):
        for j in range(height):
            tmp = np.dot(transform_matrix, [i, j, 1])
            x = int(tmp[0]/tmp[2])
            y = int(tmp[1]/tmp[2])
            if (x >= 0 and x < output_width) and (y >= 0 and y < output_height):
                res[x, y] = img[i, j]

    return res


def grayScaledFilter(img):
    transform_matrix = np.array([[1/3, 1/3, 1/3], [1/3, 1/3, 1/3], [1/3, 1/3, 1/3]], dtype='float')
    return Filter(img, transform_matrix)


def crazyFilter(img):
    transform_matrix = np.array([[0, 1, 1], [1, 0, 0], [0, 0, 0]], dtype='float')
    return Filter(img, transform_matrix)


def customFilter(img):
    transform_matrix = np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]], dtype='float')
    transformedImage = Filter(img, transform_matrix)
    showImage(transformedImage, "Custom Filter")

    inverse_matrix = np.linalg.inv(transform_matrix)
    reversed_image = Filter(transformedImage, inverse_matrix)
    showImage(reversed_image, "Reversed Image", False)


def scaleImg(img, scale_width, scale_height):
    width, height, _ = img.shape
    res = np.zeros((width * scale_width, height * scale_height, _), dtype='int')

    for i in range(width * scale_width):
        for j in range(height * scale_height):
            res[i, j] = img[int(i / scale_width), int(j / scale_height)]

    return res


def cropImg(img, start_row, end_row, start_column, end_column):
    return img[start_column:end_column, start_row:end_row]


if __name__ == "__main__":
    image_matrix = get_input('pic.jpg')

    width, height = 300, 400

    showImage(image_matrix, title="Input Image")

    pts1 = np.float32([[105, 215], [370, 180], [160, 645], [485, 565]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    m = getPerspectiveTransform(pts1, pts2)

    warpedImage = warpPerspective(image_matrix, m, width, height)
    showWarpPerspective(warpedImage)
    
    grayScalePic = grayScaledFilter(warpedImage)
    showImage(grayScalePic, title="Gray Scaled")

    crazyImage = crazyFilter(warpedImage)
    showImage(crazyImage, title="Crazy Filter")

    customFilter(warpedImage)
    
    croppedImage = cropImg(warpedImage, 50, 300, 50, 225)
    showImage(croppedImage, title="Cropped Image")

    scaledImage = scaleImg(warpedImage, 2, 3)
    showImage(scaledImage, title="Scaled Image")
