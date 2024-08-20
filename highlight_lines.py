import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from scipy.signal import find_peaks

def place_tick_marks(axis_line, num_ticks, axis_type):
    x1, y1, x2, y2 = axis_line[0]
    tick_positions = []
    for i in range(num_ticks + 1):
        if axis_type == 'horizontal':
            x = x1 + (x2 - x1) * i / num_ticks
            tick_positions.append((int(x), y1))
        else:
            y = y1 + (y2 - y1) * i / num_ticks
            tick_positions.append((x1, int(y)))
    return tick_positions

def draw_tick_marks(ax, tick_positions, axis_type, tick_length=10):
    for x, y in tick_positions:
        if axis_type == 'horizontal':
            ax.plot([x, x], [y, y-tick_length], 'g-', linewidth=1)
        else:
            ax.plot([x, x+tick_length], [y, y], 'g-', linewidth=1)

def read_axis_labels(image, tick_positions, axis_type):
    labels = []
    for x, y in tick_positions:
        if axis_type == 'horizontal':
            roi = image[y + 5:y + 30, x - 20:x + 20]
        else:
            roi = image[y - 20:y + 20, x - 50:x - 5]

        # Increase image size for better OCR results
        roi = cv2.resize(roi, (0, 0), fx=2, fy=2)

        # Apply thresholding to make text clearer
        _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        text = pytesseract.image_to_string(roi, config='--psm 7 -c tessedit_char_whitelist=0123456789.-')
        text = text.strip()
        if text:
            try:
                value = float(text)
                labels.append((value, (x, y)))
            except ValueError:
                pass  # Ignore if conversion to float fails

    return labels

def determine_axis_scale(labels):
    if len(labels) < 2:
        print(f"Not enough labels detected: {len(labels)} label(s)")
        return None, None

    labels.sort(key=lambda x: x[1][0])  # Sort by x-coordinate

    values = [label[0] for label in labels]
    pixels = [label[1][0] for label in labels]

    try:
        scale = (values[-1] - values[0]) / (pixels[-1] - pixels[0])
        intercept = values[0] - scale * pixels[0]
        return scale, intercept
    except ZeroDivisionError:
        print("Error: Division by zero when calculating scale")
        return None, None

def pixel_to_value(pixel, scale, intercept):
    return scale * pixel + intercept

# Load the rotated image
rotated_image_path = '/mnt/data/image.png'
rotated_image = cv2.imread(rotated_image_path)

# Convert to grayscale
rotated_gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
rotated_blurred = cv2.GaussianBlur(rotated_gray, (5, 5), 0)

# Apply Canny edge detection with specified thresholds
rotated_edges = cv2.Canny(rotated_blurred, 300, 400)

# Detect lines using Hough Line Transform
rotated_lines = cv2.HoughLinesP(rotated_edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# Function to find the longest line in each direction
def find_longest_line(lines):
    longest_line = None
    max_length = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length > max_length:
            max_length = length
            longest_line = line
    return longest_line

# Filter horizontal and vertical lines based on the position constraints
rotated_horizontal_lines = [line for line in rotated_lines if
                            np.mean([line[0][1], line[0][3]]) > rotated_image.shape[0] * 0.75]
rotated_vertical_lines = [line for line in rotated_lines if
                          np.mean([line[0][0], line[0][2]]) < rotated_image.shape[1] * 0.25]

# Find the longest horizontal and vertical lines
longest_rotated_horizontal_line = find_longest_line(rotated_horizontal_lines)
longest_rotated_vertical_line = find_longest_line(rotated_vertical_lines)

# Function to truncate the line at multiple consecutive white pixels
def truncate_line_at_consecutive_white_pixels(line, image, direction='horizontal', consecutive_white=20):
    x1, y1, x2, y2 = line[0]
    if direction == 'horizontal':
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2
        white_pixel_count = 0

        # Extend to the left within black pixels
        for x in range(x1, -1, -1):
            y = int(slope * x + intercept)
            if y < 0 or y >= image.shape[0] or image[y, x] > 240:
                white_pixel_count += 1
                if white_pixel_count >= consecutive_white:
                    new_x1 = x + consecutive_white
                    new_y1 = int(slope * new_x1 + intercept)
                    break
            else:
                white_pixel_count = 0

        # Extend to the right within black pixels
        white_pixel_count = 0
        for x in range(x1, image.shape[1]):
            y = int(slope * x + intercept)
            if y < 0 or y >= image.shape[0] or image[y, x] > 240:
                white_pixel_count += 1
                if white_pixel_count >= consecutive_white:
                    new_x2 = x - consecutive_white
                    new_y2 = int(slope * new_x2 + intercept)
                    break
            else:
                white_pixel_count = 0

        return [[new_x1, new_y1, new_x2, new_y2]]

    elif direction == 'vertical':
        slope = (x2 - x1) / (y2 - y1)
        intercept = x1 - slope * y1
        new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2
        white_pixel_count = 0

        # Extend upwards within black pixels
        for y in range(y1, -1, -1):
            x = int(slope * y + intercept)
            if x < 0 or x >= image.shape[1] or image[y, x] > 240:
                white_pixel_count += 1
                if white_pixel_count >= consecutive_white:
                    new_y1 = y + consecutive_white
                    new_x1 = int(slope * new_y1 + intercept)
                    break
            else:
                white_pixel_count = 0

        # Extend downwards within black pixels
        white_pixel_count = 0
        for y in range(y1, image.shape[0]):
            x = int(slope * y + intercept)
            if x < 0 or x >= image.shape[1] or image[y, x] > 240:
                white_pixel_count += 1
                if white_pixel_count >= consecutive_white:
                    new_y2 = y - consecutive_white
                    new_x2 = int(slope * new_y2 + intercept)
                    break
            else:
                white_pixel_count = 0

        return [[new_x1, new_y1, new_x2, new_y2]]

# Truncate the longest horizontal and vertical lines at multiple consecutive white pixels
truncated_horizontal_line = truncate_line_at_consecutive_white_pixels(longest_rotated_horizontal_line, rotated_gray,
                                                                      'horizontal', consecutive_white=20)
truncated_vertical_line = truncate_line_at_consecutive_white_pixels(longest_rotated_vertical_line, rotated_gray,
                                                                    'vertical', consecutive_white=20)

# Calculate the angles of the truncated lines
def calculate_angle(line):
    x1, y1, x2, y2 = line[0]
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    return angle

horizontal_angle = calculate_angle(truncated_horizontal_line)
vertical_angle = calculate_angle(truncated_vertical_line)

# Determine the necessary rotation
def determine_rotation(horizontal_angle, vertical_angle):
    if abs(horizontal_angle) < abs(vertical_angle):
        # Horizontal line needs less rotation
        rotation_angle = horizontal_angle
        direction = "clockwise" if rotation_angle > 0 else "anticlockwise"
    else:
        # Vertical line needs less rotation
        rotation_angle = 90 - vertical_angle if vertical_angle > 0 else -(90 + vertical_angle)
        direction = "clockwise" if rotation_angle > 0 else "anticlockwise"

    return rotation_angle, direction

rotation_angle, direction = determine_rotation(horizontal_angle, vertical_angle)

# Create a matplotlib figure and axis for plotting
fig, ax = plt.subplots(figsize=(12, 8))

# Detect and draw tick marks
x_tick_positions = place_tick_marks(truncated_horizontal_line, 10, 'horizontal')
y_tick_positions = place_tick_marks(truncated_vertical_line, 10, 'vertical')

# Function to rotate tick positions
def rotate_tick_positions(tick_positions, M):
    rotated_positions = []
    for (x, y) in tick_positions:
        rotated_point = cv2.transform(np.array([[[x, y]]]), M)[0][0]
        rotated_positions.append((int(rotated_point[0]), int(rotated_point[1])))
    return rotated_positions

# Rotate tick positions
rotated_x_tick_positions = rotate_tick_positions(x_tick_positions, M)
rotated_y_tick_positions = rotate_tick_positions(y_tick_positions, M)

# Draw tick marks on the corrected image
draw_tick_marks(ax, rotated_x_tick_positions, 'horizontal')
draw_tick_marks(ax, rotated_y_tick_positions, 'vertical')

# Add labels on the corrected image
for i, (x, y) in enumerate(rotated_x_tick_positions):
    ax.text(x, y + 15, f'{i}', ha='center', va='top')

for i, (x, y) in enumerate(rotated_y_tick_positions):
    ax.text(x - 15, y, f'{i / 10:.1f}', ha='right', va='center')

# Read axis labels
x_labels = read_axis_labels(rotated_gray, rotated_x_tick_positions, 'horizontal')
y_labels = read_axis_labels(rotated_gray, rotated_y_tick_positions, 'vertical')

print("X-axis labels:", x_labels)
print("Y-axis labels:", y_labels)

# Determine axis scales
x_scale, x_intercept = determine_axis_scale(x_labels)
y_scale, y_intercept = determine_axis_scale(y_labels)

# Print axis scales and intercepts
if x_scale is not None and x_intercept is not None:
    print(f"X-axis scale: {x_scale:.4f}, intercept: {x_intercept:.4f}")
else:
    print("Unable to determine X-axis scale and intercept")

if y_scale is not None and y_intercept is not None:
    print(f"Y-axis scale: {y_scale:.4f}, intercept: {y_intercept:.4f}")
else:
    print("Unable to determine Y-axis scale and intercept")

# Print the rotation angle and direction
print(f"The image should be rotated by {abs(rotation_angle):.2f} degrees {direction}.")

# Function to rotate the image
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated, M

# Determine the correct angle for rotation based on the direction
correct_rotation_angle = -rotation_angle if direction == "clockwise" else rotation_angle

# Rotate the image and the truncated lines
rotated_image_corrected, M = rotate_image(rotated_image, correct_rotation_angle)

# Draw the truncated lines with a thicker weight on the corrected image
if truncated_horizontal_line is not None:
    x1, y1, x2, y2 = truncated_horizontal_line[0]
    rotated_x1, rotated_y1 = cv2.transform(np.array([[[x1, y1]]]), M)[0][0]
    rotated_x2, rotated_y2 = cv2.transform(np.array([[[x2, y2]]]), M)[0][0]
    cv2.line(rotated_image_corrected, (int(rotated_x1), int(rotated_y1)), (int(rotated_x2), int(rotated_y2)),
             (255, 0, 0), 10)

if truncated_vertical_line is not None:
    x1, y1, x2, y2 = truncated_vertical_line[0]
    rotated_x1, rotated_y1 = cv2.transform(np.array([[[x1, y1]]]), M)[0][0]
    rotated_x2, rotated_y2 = cv2.transform(np.array([[[x2, y2]]]), M)[0][0]
    cv2.line(rotated_image_corrected, (int(rotated_x1), int(rotated_y1)), (int(rotated_x2), int(rotated_y2)),
             (0, 0, 255), 10)

fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(cv2.cvtColor(rotated_image_corrected, cv2.COLOR_BGR2RGB))
ax.set_title('Corrected Image with Axes and Tick Marks')

# Draw the axes and tick marks
if truncated_horizontal_line is not None:
    x1, y1, x2, y2 = truncated_horizontal_line[0]
    ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2)
    rotated_x_tick_positions = rotate_tick_positions(x_tick_positions, M)
    draw_tick_marks(ax, rotated_x_tick_positions, 'horizontal')

if truncated_vertical_line is not None:
    x1, y1, x2, y2 = truncated_vertical_line[0]
    ax.plot([x1, x2], [y1, y2], 'r-', linewidth=2)
    rotated_y_tick_positions = rotate_tick_positions(y_tick_positions, M)
    draw_tick_marks(ax, rotated_y_tick_positions, 'vertical')

# Add labels
for i, (x, y) in enumerate(rotated_x_tick_positions):
    ax.text(x, y + 15, f'{i}', ha='center', va='top')

for i, (x, y) in enumerate(rotated_y_tick_positions):
    ax.text(x - 15, y, f'{i / 10:.1f}', ha='right', va='center')

plt.tight_layout()
plt.show()

# Print axis scales (if needed)
print("X-axis scale: 0 to 10")
print("Y-axis scale: 0 to 1")
