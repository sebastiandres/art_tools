"""
This script is used to process images, and generate the "closest" line pattern for the image.
The line pattern is composed of a series of black and white straight lines, that are overlapped to form a pattern.
The lines do not need to be parallel, and they do not need to have same starting and ending points.
The lines are completed and continuous, they cannot be broken (short lines are not allowed).

The script will take an image as input, and generate the closest line pattern for the image.

Inputs:
- Image

Outputs:
- Points: An indexed and complete list of points of shape (i, x_i, y_i)
- Lines: The list of lines defined by the starting and ending point (p_i, q_j)

Algorithm:
- We read the image, and convert it to a binary image: We will have a list of pixels (x_i, y_i, b_i) 
where b_i is the brightness of the pixel.
- We create a list of points (x_i, y_i) that rely on a "encompasing" circle of radius r (r>width and height of the image). 
The points can be uniformly distributed on the circle, or can be distributed with some random noise to make the final pattern more interesting.
- We then create a list of lines that connect the points. The lines are completed and continuous, they cannot be broken (short lines are not allowed).
- We apply a greedy algorithm to find the next best line to add to the list of lines.
- We repeat the above steps until we have a list of lines that covers the entire image.
- We then output the list of points and lines.

The greedy algorithm is as follows:
- We start with an empty list of lines.
- For each line, we evaluate a "fitness" score for the line, were:
    - We look for the pixels that are on the line. 
    - For any pixel, we compute the pixel_scoring as a function of the brightness of the pixel 
    and the number of lines that go through the pixel.
    - We then compute the line_scoring as the sum of the pixel_scoring of all the pixels that are on the line.
    - We then add the line that has the highest line_scoring to the list of lines.
- We repeat the above steps until we have a list of lines that covers the entire image.

"""

from IPython import embed

from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os

def is_point_on_line(point_i, point_j, point_k):
    """
    Checks if point_k lies on the line defined by point_i and point_j
    """
    epsilon = 1e-3
    x_i, y_i = point_i
    x_j, y_j = point_j
    x_k, y_k = point_k
    is_on_line = np.cross(np.array([x_j - x_i, y_j - y_i]), np.array([x_k - x_i, y_k - y_i])) < epsilon
    return is_on_line

def get_pixels_in_line(point_i, point_j, pixel_brightness):
    pixels_in_line = []
    for i in range(pixel_brightness.shape[0]):
        for j in range(pixel_brightness.shape[1]):
            if is_point_on_line(point_i, point_j, (i, j)):
                pixels_in_line.append((i, j))
    return pixels_in_line

# The score of a line is the sum of the scores of the pixels that are on the line
def compute_line_score(point_i, point_j, pixel_brightness, lines_list):
    pixels_in_line = get_pixels_in_line(point_i, point_j, pixel_brightness)
    return np.sum(compute_pixel_score(pixel, pixel_brightness, lines_list) for pixel in pixels_in_line)

# The overall score of a pixel is the brightness of the pixel
def compute_pixel_score(pixel, pixel_brightness, lines_list):
    n_lines_through_pixel = 0
    for line in lines_list:
        if is_point_on_line(line[0], line[1], pixel):
            n_lines_through_pixel += 1
    return abs(pixel_brightness[pixel[0], pixel[1]] - n_lines_through_pixel / (n_lines_through_pixel + 1 ))


# PARAMETERS
N_POINTS = 30
N_LINES = 100 # Maximum number of lines to add
N_AVAILABLE_LINES = int(N_POINTS*(N_POINTS-1)/2)
print("N_LINES: ", N_LINES, N_AVAILABLE_LINES, 100*N_LINES/N_AVAILABLE_LINES)

# Read the image
image_path = os.path.join("inputs", "uno.jpg")
image = Image.open(image_path)

# Get a matrix of all the pixels and their brightness
pixel_brightness = np.array(image).sum(axis=2)/(256*3)

# Show the image
if False:
    plt.imshow(pixel_brightness, cmap="gray")
    plt.show()

# Create a list of points that are uniformly distributed on a circle of radius r
height, width = pixel_brightness.shape
r = max(width, height)/2

# Create a list of points that are uniformly distributed on a circle of radius r
points = []
for i in range(N_POINTS):
    points.append((width/2 + r*np.cos(2*np.pi*i/N_POINTS), height/2 + r*np.sin(2*np.pi*i/N_POINTS)))

# Show the points and lines
if False:
    plt.imshow(pixel_brightness, cmap="gray")
    for point_i in points:
        plt.scatter(point_i[0], point_i[1], color="grey")
        for point_j in points:
            plt.plot([point_i[0], point_j[0]], [point_i[1], point_j[1]], color="red", alpha=0.1)
    plt.show()

# Iterate
lines_list = []
for i in range(N_LINES):
    print("Line ", i)
    current_line_score = -1
    current_line = None
    for point_i in points:
        for point_j in points:
            if point_i == point_j:
                continue
            line_score = compute_line_score(point_i, point_j, pixel_brightness, lines_list)
            if line_score > current_line_score:
                current_line_score = line_score
                current_line = (point_i, point_j)
    lines_list.append(current_line)


