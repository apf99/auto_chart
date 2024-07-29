# utils.py
import cv2
import numpy as np
import pytesseract

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

def calculate_angle(line):
    x1, y1, x2, y2 = line[0]
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    return angle

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

def rotate_tick_positions(tick_positions, M):
    rotated_positions = []
    for (x, y) in tick_positions:
        rotated_point = cv2.transform(np.array([[[x, y]]]), M)[0][0]
        rotated_positions.append((int(rotated_point[0]), int(rotated_point[1])))
    return rotated_positions

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated, M

def correct_label_order(labels, axis_type):
    if axis_type == 'horizontal':
        labels.sort(key=lambda x: x[1][0])  # Sort by x-coordinate
    else:
        labels.sort(key=lambda x: x[1][1], reverse=True)  # Sort by y-coordinate in reverse

    return labels
