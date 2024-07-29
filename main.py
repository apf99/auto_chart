import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import importlib.util

# Load utility functions from utils.py
spec = importlib.util.spec_from_file_location("utils", "utils.py")
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

# Load the rotated image
rotated_image_path = 'Chart_Image_1_rotated.jpg'
rotated_image = cv2.imread(rotated_image_path)

if rotated_image is None:
    raise FileNotFoundError(f"Image file '{rotated_image_path}' not found.")

# Convert to grayscale
rotated_gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
rotated_blurred = cv2.GaussianBlur(rotated_gray, (5, 5), 0)

# Apply Canny edge detection with specified thresholds
rotated_edges = cv2.Canny(rotated_blurred, 300, 400)

# Detect lines using Hough Line Transform
rotated_lines = cv2.HoughLinesP(rotated_edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# Filter horizontal and vertical lines based on the position constraints
rotated_horizontal_lines = [line for line in rotated_lines if
                            np.mean([line[0][1], line[0][3]]) > rotated_image.shape[0] * 0.75]
rotated_vertical_lines = [line for line in rotated_lines if
                          np.mean([line[0][0], line[0][2]]) < rotated_image.shape[1] * 0.25]

# Find the longest horizontal and vertical lines
longest_rotated_horizontal_line = utils.find_longest_line(rotated_horizontal_lines)
longest_rotated_vertical_line = utils.find_longest_line(rotated_vertical_lines)

# Truncate the longest horizontal and vertical lines at multiple consecutive white pixels
truncated_horizontal_line = utils.truncate_line_at_consecutive_white_pixels(longest_rotated_horizontal_line, rotated_gray,
                                                                      'horizontal', consecutive_white=20)
truncated_vertical_line = utils.truncate_line_at_consecutive_white_pixels(longest_rotated_vertical_line, rotated_gray,
                                                                    'vertical', consecutive_white=20)

# Calculate the angles of the truncated lines
horizontal_angle = utils.calculate_angle(truncated_horizontal_line)
vertical_angle = utils.calculate_angle(truncated_vertical_line)

# Determine the necessary rotation
rotation_angle, direction = utils.determine_rotation(horizontal_angle, vertical_angle)

# Rotate the image
correct_rotation_angle = -rotation_angle if direction == "clockwise" else rotation_angle
rotated_image_corrected, M = utils.rotate_image(rotated_image, correct_rotation_angle)

# Rotate the tick positions
x_tick_positions = utils.place_tick_marks(truncated_horizontal_line, 10, 'horizontal')
y_tick_positions = utils.place_tick_marks(truncated_vertical_line, 10, 'vertical')
rotated_x_tick_positions = utils.rotate_tick_positions(x_tick_positions, M)
rotated_y_tick_positions = utils.rotate_tick_positions(y_tick_positions, M)

# Ensure tick marks line up exactly with original image
rotated_x_tick_positions[0] = (int(truncated_horizontal_line[0][0]), int(truncated_horizontal_line[0][1]))
rotated_x_tick_positions[-1] = (int(truncated_horizontal_line[0][2]), int(truncated_horizontal_line[0][3]))
rotated_y_tick_positions[0] = (int(truncated_vertical_line[0][0]), int(truncated_vertical_line[0][1]))
rotated_y_tick_positions[-1] = (int(truncated_vertical_line[0][2]), int(truncated_vertical_line[0][3]))

# Correct label order for y-axis
def correct_label_order(labels, axis_type):
    if axis_type == 'horizontal':
        labels.sort(key=lambda x: x[1][0])  # Sort by x-coordinate
    else:
        labels.sort(key=lambda x: x[1][1], reverse=True)  # Sort by y-coordinate (higher y-value means lower on the axis)

    return labels

# Enhanced OCR for y-axis labels
def read_axis_labels_enhanced(image, tick_positions, axis_type):
    labels = []
    for x, y in tick_positions:
        if axis_type == 'horizontal':
            roi = image[y + 5:y + 30, x - 20:x + 20]
        else:
            roi = image[y - 20:y + 20, x - 50:x - 5]

        # Increase image size for better OCR results
        roi = cv2.resize(roi, (0, 0), fx=3, fy=3)

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

# Create a matplotlib figure and axis for plotting
fig, ax = plt.subplots(figsize=(12, 8))

# Draw the corrected image
ax.imshow(cv2.cvtColor(rotated_image_corrected, cv2.COLOR_BGR2RGB))
ax.set_title('Corrected Image with Axes and Tick Marks')

# Draw tick marks on the corrected image
utils.draw_tick_marks(ax, rotated_x_tick_positions, 'horizontal')
utils.draw_tick_marks(ax, rotated_y_tick_positions, 'vertical')

# Add labels on the corrected image
for i, (x, y) in enumerate(rotated_x_tick_positions):
    ax.text(x, y + 15, f'{i}', ha='center', va='top')

for i, (x, y) in enumerate(rotated_y_tick_positions):
    ax.text(x - 15, y, f'{i / 10:.1f}', ha='right', va='center')

# Read axis labels
x_labels = utils.read_axis_labels(rotated_gray, rotated_x_tick_positions, 'horizontal')
y_labels = read_axis_labels_enhanced(rotated_gray, rotated_y_tick_positions, 'vertical')

# Correct label order
x_labels = correct_label_order(x_labels, 'horizontal')
y_labels = correct_label_order(y_labels, 'vertical')

print("X-axis labels:", x_labels)
print("Y-axis labels:", y_labels)

# Determine axis scales
x_scale, x_intercept = utils.determine_axis_scale(x_labels)
y_scale, y_intercept = utils.determine_axis_scale(y_labels)

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

# Draw the axes and tick marks
if truncated_horizontal_line is not None:
    x1, y1, x2, y2 = truncated_horizontal_line[0]
    rotated_x1, rotated_y1 = cv2.transform(np.array([[[x1, y1]]]), M)[0][0]
    rotated_x2, rotated_y2 = cv2.transform(np.array([[[x2, y2]]]), M)[0][0]
    ax.plot([rotated_x1, rotated_x2], [rotated_y1, rotated_y2], 'b-', linewidth=2)
    utils.draw_tick_marks(ax, rotated_x_tick_positions, 'horizontal')

if truncated_vertical_line is not None:
    x1, y1, x2, y2 = truncated_vertical_line[0]
    rotated_x1, rotated_y1 = cv2.transform(np.array([[[x1, y1]]]), M)[0][0]
    rotated_x2, rotated_y2 = cv2.transform(np.array([[[x2, y2]]]), M)[0][0]
    ax.plot([rotated_x1, rotated_x2], [rotated_y1, rotated_y2], 'r-', linewidth=2)
    utils.draw_tick_marks(ax, rotated_y_tick_positions, 'vertical')

plt.tight_layout()
plt.show()
