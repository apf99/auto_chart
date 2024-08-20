import cv2
import numpy as np
import pytesseract
import os


def remove_pixels(image):
    mask = np.logical_and(image >= 80, image <= 255)
    image[mask] = 255
    return image

def detect_axes(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    horizontal_lines = []
    vertical_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 45:  # Horizontal line
                horizontal_lines.append((length, line[0]))
            else:  # Vertical line
                vertical_lines.append((length, line[0]))
    
    # Select the longest horizontal and vertical lines as axes
    x_axis = max(horizontal_lines, key=lambda x: x[0])[1] if horizontal_lines else None
    y_axis = max(vertical_lines, key=lambda x: x[0])[1] if vertical_lines else None
    
    return x_axis, y_axis

def extend_axes(x_axis, y_axis):
    if x_axis is None or y_axis is None:
        return x_axis, y_axis

    # Calculate slopes
    x_slope = (x_axis[3] - x_axis[1]) / (x_axis[2] - x_axis[0]) if x_axis[2] != x_axis[0] else float('inf')
    y_slope = (y_axis[3] - y_axis[1]) / (y_axis[2] - y_axis[0]) if y_axis[2] != y_axis[0] else float('inf')

    # Check if lines are parallel
    if x_slope == y_slope:
        return x_axis, y_axis

    # Calculate intersection point
    x_intercept = x_axis[1] - x_slope * x_axis[0]
    y_intercept = y_axis[1] - y_slope * y_axis[0]

    if x_slope == float('inf'):
        intersection_x = x_axis[0]
        intersection_y = y_slope * intersection_x + y_intercept
    elif y_slope == float('inf'):
        intersection_x = y_axis[0]
        intersection_y = x_slope * intersection_x + x_intercept
    else:
        intersection_x = (y_intercept - x_intercept) / (x_slope - y_slope)
        intersection_y = x_slope * intersection_x + x_intercept

    # Extend x_axis
    x_axis = [int(intersection_x), int(intersection_y), int(x_axis[2]), int(x_axis[3])]

    # Extend y_axis
    y_axis = [int(intersection_x), int(intersection_y), int(y_axis[2]), int(y_axis[3])]

    return x_axis, y_axis

def flood_fill(image, x, y):
    h, w = image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    flood_fill_image = image.copy()
    cv2.floodFill(flood_fill_image, mask, (x,y), 255, loDiff=5, upDiff=5)
    return cv2.bitwise_not(flood_fill_image - image)

def get_text_in_region(original_image, bbox, index):
    x1, y1, x2, y2 = bbox
    roi = original_image[y1:y2, x1:x2]
    
    # Check if the image is already in grayscale
    if len(roi.shape) == 2:
        roi_gray = roi
    else:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    roi_adaptive_thresh = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Apply contrast adjustment
    roi_contrast = cv2.equalizeHist(roi_gray)
    
    # Save intermediate images for debugging
    cv2.imwrite(f'debug_roi_{index}.png', roi)
    cv2.imwrite(f'debug_roi_adaptive_thresh_{index}.png', roi_adaptive_thresh)
    cv2.imwrite(f'debug_roi_contrast_{index}.png', roi_contrast)
    
    # Perform OCR on the preprocessed ROI
    text = pytesseract.image_to_string(roi_adaptive_thresh, config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789. -c tessedit_min_confidence=60')
    text = text.strip()
    
    # Get confidence scores
    data = pytesseract.image_to_data(roi_adaptive_thresh, output_type=pytesseract.Output.DICT)
    confidences = data['conf']
    
    return text, confidences, roi

def find_measure(image, start_x, start_y, direction):
    x, y = start_x, start_y
    print(f"Starting measure search at ({x}, {y}) with direction {direction}")
    
    # Find white space
    while 0 <= x < image.shape[1] and image[y, x] < 128:
        x += direction[0]
    
    # Find first dark pixel
    while 0 <= x < image.shape[1] and image[y, x] >= 128:
        x += direction[0]
    
    if x < 0 or x >= image.shape[1]:
        return None
    
    # Find all connected dark pixels to the left edge
    right_x = x
    top_y, bottom_y = y, y
    left_x = 0  # Start from the left edge of the image
    
    while left_x < right_x and any(image[top_y:bottom_y+1, left_x] < 128):
        left_x += 1
    
    while top_y > 0 and any(image[top_y-1, left_x:right_x+1] < 128):
        top_y -= 1
    
    while bottom_y < image.shape[0]-1 and any(image[bottom_y+1, left_x:right_x+1] < 128):
        bottom_y += 1

    # Adjust left_x to touch the left edge of the measure
    for col in range(left_x, right_x):
        if any(image[top_y:bottom_y+1, col] < 128):
            left_x = col
            break
    
    # Expand the bounding box by 5 pixels in each direction
    print("Expanding Y-axis bounding box")
    print(f"Original coordinates: ({left_x}, {top_y}, {right_x}, {bottom_y})")
    
    left_x = max(0, left_x - 5)
    right_x = min(image.shape[1] - 1, right_x + 5)
    top_y = max(0, top_y - 5)
    bottom_y = min(image.shape[0] - 1, bottom_y + 5)

    print(f"Expanded coordinates: ({left_x}, {top_y}, {right_x}, {bottom_y})")
    
    return (left_x, top_y, right_x, bottom_y)

def find_x_measure(image, start_x):
    print(f"Starting x-axis measure search from the bottom at x={start_x}")
    
    # Start from the bottom of the image and move upwards
    y = image.shape[0] - 1
    dark_pixel_count = 0
    
    # Search for three consecutive rows of dark pixels
    while y >= 0:
        if any(image[y, max(0, start_x-15):min(image.shape[1], start_x+16)] < 100):
            dark_pixel_count += 1
            print(f"Row {y} is dark. Consecutive dark rows: {dark_pixel_count}")
            if dark_pixel_count >= 3:
                print(f"Found three consecutive rows of dark pixels at y={y}")
                break
        else:
            dark_pixel_count = 0
        y -= 1
    if y < 0:
        print("Failed to find three consecutive rows of dark pixels")
        return None
    
    # Record the y-coordinate where the three rows of dark pixels were detected
    dark_pixel_y = y
    
    # Continue searching upwards until we find three consecutive rows of white pixels
    white_pixel_count = 0
    while y >= 0:
        if all(image[y, max(0, start_x-15):min(image.shape[1], start_x+16)] > 200):
            white_pixel_count += 1
            print(f"Row {y} is white. Consecutive white rows: {white_pixel_count}")
            if white_pixel_count >= 3:
                print(f"Found three consecutive rows of white pixels at y={y}")
                break
        else:
            white_pixel_count = 0
        y -= 1
    if y < 0:
        print("Failed to find three consecutive rows of white pixels")
        return None
    
    # Record the y-coordinate where the three rows of white pixels were detected
    white_pixel_y = y
    
    # Flood fill to find all connected dark pixels
    mask = np.zeros((image.shape[0]+2, image.shape[1]+2), np.uint8)
    flood_fill_image = image.copy()
    
    # Limit the flood fill region
    flood_region = image[max(0, dark_pixel_y-50):min(image.shape[0], dark_pixel_y+50), 
                         max(0, start_x-100):min(image.shape[1], start_x+100)]
    flood_mask = np.zeros((flood_region.shape[0]+2, flood_region.shape[1]+2), np.uint8)
    
    cv2.floodFill(flood_region, flood_mask, (min(15, flood_region.shape[1]-1), min(50, flood_region.shape[0]-1)), 
                  128, loDiff=30, upDiff=30)
    
    # Find bounding box of the filled region
    filled_region = np.where(flood_region == 128)
    if len(filled_region[0]) == 0:
        print("No filled region found")
        return None
    
    left_x = max(0, start_x-100) + filled_region[1].min()
    right_x = max(0, start_x-100) + filled_region[1].max()
    
    # Set the top and bottom edges based on the detected white and dark pixels
    top_y = white_pixel_y + 1
    bottom_y = dark_pixel_y

    # Adjust the left edge to touch a dark pixel
    while left_x < right_x and np.all(image[top_y:bottom_y+1, left_x] >= 128):
        left_x += 1
    
    # Adjust the right edge to touch a dark pixel
    while right_x > left_x and np.all(image[top_y:bottom_y+1, right_x] >= 128):
        right_x -= 1
    
    # Adjust the top edge to touch a dark pixel if necessary
    while top_y < bottom_y and np.all(image[top_y, left_x:right_x+1] >= 128):
        top_y += 1
    
    # Adjust the bottom edge to touch a dark pixel if necessary
    while bottom_y > top_y and np.all(image[bottom_y, left_x:right_x+1] >= 128):
        bottom_y -= 1

    # Adjust the left edge to touch a white pixel
    while left_x > 0 and np.any(image[top_y:bottom_y+1, left_x] < 128):
        left_x -= 1
    
    # Adjust the right edge to touch a white pixel
    while right_x < image.shape[1] - 1 and np.any(image[top_y:bottom_y+1, right_x] < 128):
        right_x += 1
    
    # Adjust the top edge to touch a white pixel
    while top_y > 0 and np.any(image[top_y, left_x:right_x+1] < 128):
        top_y -= 1
    
    # Adjust the bottom edge to touch a white pixel
    while bottom_y < image.shape[0] - 1 and np.any(image[bottom_y, left_x:right_x+1] < 128):
        bottom_y += 1

    # Expand the bounding box by 5 pixels in each direction
    print("Expanding X-axis bounding box")
    print(f"Original coordinates: ({left_x}, {top_y}, {right_x}, {bottom_y})")
    
    left_x = max(0, left_x - 5)
    right_x = min(image.shape[1] - 1, right_x + 5)
    top_y = max(0, top_y - 5)
    bottom_y = min(image.shape[0] - 1, bottom_y + 5)

    print(f"Expanded coordinates: ({left_x}, {top_y}, {right_x}, {bottom_y})")
    
    print(f"X-axis measure found: ({left_x}, {top_y}, {right_x}, {bottom_y})")
    return (left_x, top_y, right_x, bottom_y)

def find_measures(image, x_axis, y_axis):
    measures = []
    
    # Y-axis measures
    y_max = find_measure(image, y_axis[2], y_axis[3], (-1, 0))
    y_min = find_measure(image, y_axis[0], y_axis[1], (-1, 0))

    if y_max:
        measures.append(tuple(map(int, y_max)))
    if y_min:
        measures.append(tuple(map(int, y_min)))

    # X-axis measures
    min_x_measure = find_x_measure(image, x_axis[0])
    max_x_measure = find_x_measure(image, x_axis[2])
    
    if min_x_measure:
        measures.append(tuple(map(int, min_x_measure)))
    if max_x_measure:
        measures.append(tuple(map(int, max_x_measure)))
    
    return measures

def find_tickmarks(image, x_axis, y_axis):
    # Assume min tickmarks are at the intersection point
    min_x_tickmark = (int(x_axis[0]), int(x_axis[1]))
    min_y_tickmark = (int(y_axis[0]), int(y_axis[1]))
    
    # Max tickmarks are at the other ends of the axes
    max_x_tickmark = (int(x_axis[2]), int(x_axis[3]))
    max_y_tickmark = (int(y_axis[2]), int(y_axis[3]))
    
    return min_x_tickmark, max_x_tickmark, min_y_tickmark, max_y_tickmark

def remove_axes_and_ticks(image, x_axis, y_axis, tick_length=10, line_thickness=3):
    cleaned_image = image.copy()
    
    # Remove x-axis with increased thickness
    cv2.line(cleaned_image, (x_axis[0], x_axis[1]), (x_axis[2], x_axis[3]), 255, line_thickness * 2)
    
    # Remove y-axis with increased thickness
    cv2.line(cleaned_image, (y_axis[0], y_axis[1]), (y_axis[2], y_axis[3]), 255, line_thickness * 2)
    
    # Remove tick marks on x-axis with further increased width and extended area
    for x in range(x_axis[0] - tick_length, x_axis[2] + tick_length + 1, tick_length):
        cv2.rectangle(cleaned_image, (x-3, x_axis[1] - 15), (x+3, x_axis[1] + 15), 255, -1)
    
    # Remove tick marks on y-axis with increased width
    for y in range(y_axis[1] - tick_length, y_axis[3] + tick_length + 1, tick_length):
        cv2.rectangle(cleaned_image, (y_axis[0] - 15, y-3), (y_axis[0] + 15, y+3), 255, -1)
    
    return cleaned_image

def trace_curves(cleaned_image, result_image, x_axis, y_axis):
    print("Starting trace_curves function")
    
    # Define the bounding rectangle
    min_x, max_x = min(x_axis[0], x_axis[2]), max(x_axis[0], x_axis[2])
    min_y, max_y = min(y_axis[1], y_axis[3]), max(y_axis[1], y_axis[3])
    
    # Extract the ROI containing the curves from the cleaned image
    roi = cleaned_image[min_y:max_y, min_x:max_x]
    cv2.imwrite('debug_1_roi.png', roi)
    
    # Apply edge detection
    edges = cv2.Canny(roi, 150, 300)
    cv2.imwrite('debug_2_canny_edges.png', edges)
    
    # Apply dilation to merge double lines
    kernel_size = 2  # Adjust this value as needed
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    cv2.imwrite('debug_3_dilated_edges.png', dilated_edges)
    
    # Optional: Apply erosion to thin the lines if they became too thick
    eroded_edges = cv2.erode(dilated_edges, kernel, iterations=1)
    cv2.imwrite('debug_4_eroded_edges.png', eroded_edges)
    
    # Find starting points for each curve
    starting_points = find_curve_starting_points(eroded_edges, x_axis, y_axis)
    
    # Process each curve individually
    curves = []
    for start_point in starting_points:
        curve = trace_single_curve(eroded_edges, start_point)
        curves.append(curve)
    
    # Create debug images
    debug_image_white = create_debug_image(curves, x_axis, y_axis, white_background=True)
    debug_image_original = create_debug_image(curves, x_axis, y_axis, white_background=False, original_image=cleaned_image)
    
    cv2.imwrite('debug_curves_white_background.png', debug_image_white)
    cv2.imwrite('debug_curves_original_background.png', debug_image_original)
    
    # Draw filtered contours on the result image
    for curve in curves:
        curve_np = np.array(curve, dtype=np.int32)
        cv2.polylines(result_image[min_y:max_y, min_x:max_x], [curve_np], False, (0, 255, 0), 1)
    
    cv2.imwrite('debug_7_result_with_contours.png', result_image)
    
    print("Finishing trace_curves function")
    return result_image, curves

def find_curve_starting_points(edges, x_axis, y_axis):
    starting_points = []
    height, width = edges.shape
    
    # Check along x-axis
    for x in range(width):
        if edges[0, x] > 0:  # Check top edge
            starting_points.append((x, 0))
        if edges[-1, x] > 0:  # Check bottom edge
            starting_points.append((x, height - 1))
    
    # Check along y-axis
    for y in range(height):
        if edges[y, 0] > 0:  # Check left edge
            starting_points.append((0, y))
        if edges[y, -1] > 0:  # Check right edge
            starting_points.append((width - 1, y))
    
    # Filter out noise and ensure minimum curve length
    min_curve_length = min(height, width) // 10  # Adjust as needed
    filtered_starting_points = []
    for point in starting_points:
        if is_valid_curve_start(edges, point, min_curve_length):
            filtered_starting_points.append(point)
    
    return filtered_starting_points

def is_valid_curve_start(edges, start, min_length):
    x, y = start
    length = 0
    visited = set()
    
    while 0 <= x < edges.shape[1] and 0 <= y < edges.shape[0] and edges[y, x] > 0:
        length += 1
        visited.add((x, y))
        if length >= min_length:
            return True
        
        # Check neighboring pixels
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
            new_x, new_y = x + dx, y + dy
            if (new_x, new_y) not in visited and 0 <= new_x < edges.shape[1] and 0 <= new_y < edges.shape[0] and edges[new_y, new_x] > 0:
                x, y = new_x, new_y
                break
        else:
            break  # No unvisited neighbors found
    
    return False

def trace_single_curve(edges, start):
    curve = [start]
    x, y = start
    visited = set([start])
    
    while True:
        next_point = find_next_point(edges, (x, y), curve, visited)
        if next_point is None:
            break
        curve.append(next_point)
        visited.add(next_point)
        x, y = next_point
    
    return curve

def find_next_point(edges, current, curve, visited):
    x, y = current
    if len(curve) > 1:
        dx, dy = x - curve[-2][0], y - curve[-2][1]
        priority = [(dx, dy), (dy, -dx), (-dy, dx), (-dx, -dy)]
    else:
        priority = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    
    for dx, dy in priority:
        new_x, new_y = x + dx, y + dy
        if (new_x, new_y) not in visited and 0 <= new_x < edges.shape[1] and 0 <= new_y < edges.shape[0] and edges[new_y, new_x] > 0:
            return (new_x, new_y)
    return None

def create_debug_image(curves, x_axis, y_axis, white_background=True, original_image=None):
    if white_background:
        debug_image = np.ones((max(y_axis[1], y_axis[3]), max(x_axis[0], x_axis[2]), 3), dtype=np.uint8) * 255
    else:
        debug_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    
    # Draw axes in red
    cv2.line(debug_image, (x_axis[0], x_axis[1]), (x_axis[2], x_axis[3]), (0, 0, 255), 2)
    cv2.line(debug_image, (y_axis[0], y_axis[1]), (y_axis[2], y_axis[3]), (0, 0, 255), 2)
    
    # Draw curves
    colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]  # Add more colors if needed
    for i, curve in enumerate(curves):
        color = colors[i % len(colors)]
        curve_np = np.array(curve, dtype=np.int32)
        cv2.polylines(debug_image, [curve_np], False, color, 1)
    
    # Draw max and min points
    cv2.circle(debug_image, (x_axis[0], x_axis[1]), 5, (0, 255, 0), -1)
    cv2.circle(debug_image, (x_axis[2], x_axis[3]), 5, (0, 255, 0), -1)
    cv2.circle(debug_image, (y_axis[0], y_axis[1]), 5, (0, 255, 0), -1)
    cv2.circle(debug_image, (y_axis[2], y_axis[3]), 5, (0, 255, 0), -1)
    
    return debug_image

def create_white_background_image(cleaned_image, result_image, x_axis, y_axis, filtered_contours, 
                                  min_x_tickmark, max_x_tickmark, min_y_tickmark, max_y_tickmark,
                                  measures):
    # Create a white image of the same size as the original
    white_background = np.ones_like(result_image) * 255

    # Define the bounding rectangle
    min_x, max_x = min(x_axis[0], x_axis[2]), max(x_axis[0], x_axis[2])
    min_y, max_y = min(y_axis[1], y_axis[3]), max(y_axis[1], y_axis[3])

    # Draw the axes in red
    cv2.line(white_background, (x_axis[0], x_axis[1]), (x_axis[2], x_axis[3]), (0, 0, 255), 2)
    cv2.line(white_background, (y_axis[0], y_axis[1]), (y_axis[2], y_axis[3]), (0, 0, 255), 2)

    # Draw the filtered contours in green
    for curve in filtered_contours:
        curve_np = np.array(curve, dtype=np.int32)
        cv2.polylines(white_background[min_y:max_y, min_x:max_x], [curve_np], False, (0, 255, 0), 2)

    # Function to add text with a white background
    def put_text_with_background(img, text, position, font_scale=0.9, thickness=2, text_color=(255,0,0), bg_color=(255,255,255)):
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_offset_x, text_offset_y = position
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
        cv2.rectangle(img, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

    # Draw green dots and value labels for max and min locations
    for i, measure in enumerate(measures):
        text, _, _ = get_text_in_region(original_image, measure, i)
        if i == 0:  # max Y
            cv2.circle(white_background, (max_y_tickmark[0], max_y_tickmark[1]), 5, (0, 255, 0), -1)
            put_text_with_background(white_background, text, (max_y_tickmark[0]-70, max_y_tickmark[1]+10))
        elif i == 1:  # min Y
            cv2.circle(white_background, (min_y_tickmark[0], min_y_tickmark[1]), 5, (0, 255, 0), -1)
            put_text_with_background(white_background, text, (min_y_tickmark[0]-70, min_y_tickmark[1]+10))
        elif i == 2:  # min X
            cv2.circle(white_background, (min_x_tickmark[0], min_x_tickmark[1]), 5, (0, 255, 0), -1)
            put_text_with_background(white_background, text, (min_x_tickmark[0]-20, min_x_tickmark[1]+30))
        elif i == 3:  # max X
            cv2.circle(white_background, (max_x_tickmark[0], max_x_tickmark[1]), 5, (0, 255, 0), -1)
            put_text_with_background(white_background, text, (max_x_tickmark[0]-20, max_x_tickmark[1]+30))

    return white_background

def trace_original_image(cleaned_image, result_image, x_axis, y_axis, filtered_contours, 
                                  min_x_tickmark, max_x_tickmark, min_y_tickmark, max_y_tickmark,
                                  measures, original_with_traces):
    # Create a white image of the same size as the original

    # Define the bounding rectangle
    min_x, max_x = min(x_axis[0], x_axis[2]), max(x_axis[0], x_axis[2])
    min_y, max_y = min(y_axis[1], y_axis[3]), max(y_axis[1], y_axis[3])

    # Draw the axes in red
    cv2.line(original_with_traces, (x_axis[0], x_axis[1]), (x_axis[2], x_axis[3]), (0, 0, 255), 2)
    cv2.line(original_with_traces, (y_axis[0], y_axis[1]), (y_axis[2], y_axis[3]), (0, 0, 255), 2)

    # Draw the filtered contours in green
    for curve in filtered_contours:
        curve_np = np.array(curve, dtype=np.int32)
        cv2.polylines(original_with_traces[min_y:max_y, min_x:max_x], [curve_np], False, (0, 255, 0), 2)

    # Function to add text with a white background
    def put_text_with_background(img, text, position, font_scale=0.9, thickness=2, text_color=(255,0,0), bg_color=(255,255,255)):
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_offset_x, text_offset_y = position
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
        cv2.rectangle(img, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

    # Draw green dots and value labels for max and min locations
    for i, measure in enumerate(measures):
        text, _, _ = get_text_in_region(original_image, measure, i)
        if i == 0:  # max Y
            cv2.circle(original_with_traces, (max_y_tickmark[0], max_y_tickmark[1]), 5, (0, 255, 0), -1)
            put_text_with_background(original_with_traces, text, (max_y_tickmark[0]-70, max_y_tickmark[1]+10))
        elif i == 1:  # min Y
            cv2.circle(original_with_traces, (min_y_tickmark[0], min_y_tickmark[1]), 5, (0, 255, 0), -1)
            put_text_with_background(original_with_traces, text, (min_y_tickmark[0]-70, min_y_tickmark[1]+10))
        elif i == 2:  # min X
            cv2.circle(original_with_traces, (min_x_tickmark[0], min_x_tickmark[1]), 5, (0, 255, 0), -1)
            put_text_with_background(original_with_traces, text, (min_x_tickmark[0]-20, min_x_tickmark[1]+30))
        elif i == 3:  # max X
            cv2.circle(original_with_traces, (max_x_tickmark[0], max_x_tickmark[1]), 5, (0, 255, 0), -1)
            put_text_with_background(original_with_traces, text, (max_x_tickmark[0]-20, max_x_tickmark[1]+30))

    return original_with_traces


print("Script started")

# Load the original image
print("Loading original image...")
original_image = cv2.imread('Chart_Image_1.jpg', cv2.IMREAD_GRAYSCALE)
if original_image is None:
    raise FileNotFoundError(f"Image file 'Chart_Image_1.jpg' not found.")
print("Original image loaded.")
original_with_traces = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)


# Remove pixels in the 80-255 range
print("Removing pixels in the 80-255 range...")
processed_image = remove_pixels(original_image.copy())
print("Pixels removed.")

# Detect axes
print("Detecting axes...")
x_axis, y_axis = detect_axes(processed_image)
print(f"Axes detected: x_axis={x_axis}, y_axis={y_axis}")

# Extend axes
print("Extending axes...")
x_axis, y_axis = extend_axes(x_axis, y_axis)
print(f"Axes extended: x_axis={x_axis}, y_axis={y_axis}")

# Create a color image for displaying the result
result_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)

# Draw detected axes in red
if x_axis is not None:
    cv2.line(result_image, (x_axis[0], x_axis[1]), (x_axis[2], x_axis[3]), (0, 0, 255), 2)
if y_axis is not None:
    cv2.line(result_image, (y_axis[0], y_axis[1]), (y_axis[2], y_axis[3]), (0, 0, 255), 2)

# Find tickmarks
print("Finding tickmarks...")
min_x_tickmark, max_x_tickmark, min_y_tickmark, max_y_tickmark = find_tickmarks(processed_image, x_axis, y_axis)
print(f"Tickmarks found: min_x_tickmark={min_x_tickmark}, max_x_tickmark={max_x_tickmark}, min_y_tickmark={min_y_tickmark}, max_y_tickmark={max_y_tickmark}")

# Find and highlight measures
print("Finding measures...")
measures = find_measures(original_image, x_axis, y_axis)
print("Measures after find_measures:", measures)
print("Number of measures found:", len(measures))
for i, bbox in enumerate(measures):
    text, confidences, roi = get_text_in_region(original_image, bbox, i)
    print(f"Measure {i}: {bbox}")
    print(f"Text detected: '{text}'")
    print(f"Confidence scores: {confidences}")
    
    if not text:
        print(f"Warning: No text detected for measure {i}")

# Save the intermediate result with axes, tick marks, and measures
cv2.imwrite('result_with_axes_and_measures.png', result_image)
print("Intermediate result saved as 'result_with_axes_and_measures.png'")

# Create a cleaned image for curve tracing
print("Removing axes and ticks...")
cleaned_image = remove_axes_and_ticks(processed_image, x_axis, y_axis)
print("Axes and ticks removed.")

# Trace the curves on the result_image (which contains all previously detected elements)
print("About to call trace_curves")
result_image, curves = trace_curves(cleaned_image, result_image, x_axis, y_axis)
print("Finished calling trace_curves")

# Define the ROI based on the min and max locations of the axes
roi_top_left = (min(x_axis[0], x_axis[2]), min(y_axis[1], y_axis[3]))
roi_bottom_right = (max(x_axis[0], x_axis[2]), max(y_axis[1], y_axis[3]))

# Print the coordinates for debugging
print(f"ROI Top Left: ({int(roi_top_left[0])}, {int(roi_top_left[1])})")
print(f"ROI Bottom Right: ({int(roi_bottom_right[0])}, {int(roi_bottom_right[1])})")

# Create a copy of the original image to work on
highlighted_image = result_image.copy()

# Comment out or remove the following line to eliminate the blue bounding box
# cv2.rectangle(highlighted_image, roi_top_left, roi_bottom_right, (255, 0, 0), 2)  # Blue color in BGR
print("Blue bounding box drawing skipped")

# Draw the max and min locations as green dots
min_x_tickmark = (int(min_x_tickmark[0]), int(min_x_tickmark[1]))
max_x_tickmark = (int(max_x_tickmark[0]), int(max_x_tickmark[1]))
min_y_tickmark = (int(min_y_tickmark[0]), int(min_y_tickmark[1]))
max_y_tickmark = (int(max_y_tickmark[0]), int(max_y_tickmark[1]))

print(f"min_x_tickmark: {min_x_tickmark}, max_x_tickmark: {max_x_tickmark}, min_y_tickmark: {min_y_tickmark}, max_y_tickmark: {max_y_tickmark}")
cv2.circle(highlighted_image, min_x_tickmark, 10, (0, 255, 0), -1)
cv2.circle(highlighted_image, max_x_tickmark, 10, (0, 255, 0), -1)
cv2.circle(highlighted_image, min_y_tickmark, 10, (0, 255, 0), -1)
cv2.circle(highlighted_image, max_y_tickmark, 10, (0, 255, 0), -1)
print("Green dots for max and min locations drawn")

# Draw the max and min values in blue
for i, bbox in enumerate(measures):
    text, confidences, roi = get_text_in_region(result_image, bbox, i)
    if text:
        cv2.putText(highlighted_image, text, (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
print("Max and min values drawn in blue")

# Draw the axes in red last with increased thickness
if x_axis is not None:
    cv2.line(highlighted_image, (x_axis[0], x_axis[1]), (x_axis[2], x_axis[3]), (0, 0, 255), 5)  # Increased thickness
    print("Red x-axis drawn")
if y_axis is not None:
    cv2.line(highlighted_image, (y_axis[0], y_axis[1]), (y_axis[2], y_axis[3]), (0, 0, 255), 5)  # Increased thickness
    print("Red y-axis drawn")

# Save the result image with the specified highlights
cv2.imwrite('result_with_highlights.png', highlighted_image)
print("Result with highlights saved as 'result_with_highlights.png'")

# Create a white background image with traced elements
white_background_image = create_white_background_image(cleaned_image, result_image, x_axis, y_axis, curves, 
                                                       min_x_tickmark, max_x_tickmark, min_y_tickmark, max_y_tickmark,
                                                       measures)

original_with_traces = trace_original_image(cleaned_image, result_image, x_axis, y_axis, curves, 
                                                       min_x_tickmark, max_x_tickmark, min_y_tickmark, max_y_tickmark,
                                                       measures, original_with_traces)



# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to the image
image_path = os.path.join(script_dir, 'Chart_Image_1.jpg')

# Read the image
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Unable to load image from {image_path}")
    exit(1)

# Get and print image dimensions
height, width = img.shape[:2]
print(f"Image dimensions: {width} x {height}")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blur_size = 3
blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

# Apply thresholding
thresh_value = 40
_, binary = cv2.threshold(blurred, thresh_value, 255, cv2.THRESH_BINARY)

# Set the specified parameters
min_length = 282
min_area = 570
line_thickness = 7

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on length, area, and proximity to edges
filtered_contours = []
edge_margin = 10  # Margin from the edge to consider as border

for cnt in contours:
    length = cv2.arcLength(cnt, True)
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    
    if (length > min_length and
        area > min_area and
        x > edge_margin and y > edge_margin and
        x + w < width - edge_margin and
        y + h < height - edge_margin):
        filtered_contours.append(cnt)

# Create a white background image
contour_img = np.ones_like(img) * 255

# Draw filtered contours with the specified thickness
cv2.drawContours(contour_img, filtered_contours, -1, (0, 0, 255), line_thickness)

# Store a copy of the contour image before adding circles
contour_img_copy = contour_img.copy()

# After creating contour_img_copy but before any further processing
contour_copy_path = os.path.join(script_dir, 'initial_contour_image_copy.jpg')
cv2.imwrite(contour_copy_path, contour_img_copy)
print(f"Initial contour image copy saved to: {contour_copy_path}")

def is_red_pixel(b, g, r):
    return r > 200 and g < 50 and b < 50

# Find starting points (from left to right)
start_points = []
scan_width = int(width * 0.2)  # Scan the first 20% of the image width

for x in range(scan_width):
    y = 0
    while y < height:
        if is_red_pixel(*map(int, contour_img[y, x])):
            top = bottom = y
            while bottom < height - 1 and is_red_pixel(*map(int, contour_img[bottom + 1, x])):
                bottom += 1
            
            connected = False
            if x > 0:
                for check_y in range(max(0, top - 2), min(height, bottom + 3)):
                    if is_red_pixel(*map(int, contour_img[check_y, x - 1])):
                        connected = True
                        break
                
                if not connected:
                    middle_y = (top + bottom) // 2
                    if is_red_pixel(*map(int, contour_img[middle_y, x - 1])):
                        connected = True
            
            if not connected:
                center_y = (top + bottom) // 2
                start_points.append((x, center_y))
            
            y = bottom + 1
        else:
            y += 1

# Find ending points (from right to left)
end_points = []
scan_width = int(width * 0.1)  # Scan the last 10% of the image width

for x in range(width - 1, width - scan_width - 1, -1):
    y = 0
    while y < height:
        if is_red_pixel(*map(int, contour_img[y, x])):
            top = bottom = y
            while bottom < height - 1 and is_red_pixel(*map(int, contour_img[bottom + 1, x])):
                bottom += 1
            
            connected = False
            if x < width - 1:
                for check_y in range(max(0, top - 2), min(height, bottom + 3)):
                    if is_red_pixel(*map(int, contour_img[check_y, x + 1])):
                        connected = True
                        break
                
                if not connected:
                    middle_y = (top + bottom) // 2
                    if is_red_pixel(*map(int, contour_img[middle_y, x + 1])):
                        connected = True
            
            if not connected:
                center_y = (top + bottom) // 2
                end_points.append((x, center_y))
            
            y = bottom + 1
        else:
            y += 1

# Draw circles at the starting points (orange)
for point in start_points:
    cv2.circle(contour_img, point, 10, (0, 165, 255), -1)

# Draw circles at the ending points (cyan)
for point in end_points:
    cv2.circle(contour_img, point, 10, (255, 255, 0), -1)
# Convert the contour image copy to a binary bitmap
_, bitmap = cv2.threshold(cv2.cvtColor(contour_img_copy, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)

def save_debug_image(bitmap, filename):
    # Convert boolean or 0-1 float to 0-255 uint8
    if bitmap.dtype == bool or (bitmap.dtype in [np.float32, np.float64] and bitmap.max() <= 1):
        image_to_save = (1 - bitmap.astype(np.uint8)) * 255
    else:
        image_to_save = bitmap.astype(np.uint8)
    
    cv2.imwrite(filename, image_to_save)

# Save the initial bitmap for debugging
save_debug_image(bitmap, 'debug_initial_bitmap.jpg')

def apply_secondary_filter(group, prev_groups):
    start_y, end_y, _ = group
    current_group_pixels = set(range(min(start_y, end_y), max(start_y, end_y) + 1))
    
    prev_groups_pixels = set()
    for prev_start_y, prev_end_y, _ in prev_groups:
        prev_groups_pixels.update(range(min(prev_start_y, prev_end_y), max(prev_start_y, prev_end_y) + 1))
    
    return bool(current_group_pixels.intersection(prev_groups_pixels))

def identify_pixel_groups(bitmap, column, min_height, start_points, end_points, prev_groups):
    height = bitmap.shape[0]
    groups = []
    final_proposed_groups = []
    in_group = False
    start_y = 0
    
    start_ys = set(y for x, y in start_points if x == column)
    end_ys = set(y for x, y in end_points if x == column)
    
    for y in range(height - 1, -1, -1):
        if bitmap[y, column] == 0 and not in_group:
            in_group = True
            start_y = y
        elif (bitmap[y, column] == 255 or y == 0) and in_group:
            end_y = y if bitmap[y, column] == 255 else y + 1
            group_height = start_y - end_y + 1
            
            contains_start_end = any(py in start_ys or py in end_ys for py in range(end_y, start_y + 1))
            
            if group_height >= min_height or contains_start_end:
                group = (start_y, end_y, group_height)
                groups.append(group)
                
                if contains_start_end:
                    final_proposed_groups.append(group)
                elif prev_groups:
                    would_keep = apply_secondary_filter(group, prev_groups)
                    if would_keep:
                        final_proposed_groups.append(group)
                    # else:
                    #     # Log when a group would be eliminated by the secondary filter
                    #     print(f"\nColumn {column}: Secondary filter eliminated a group")
                    #     print(f"prev_groups: {prev_groups}")
                    #     print(f"current_groups: {groups}")
                    #     print(f"final_proposed_groups: {final_proposed_groups}")
                else:
                    final_proposed_groups.append(group)
            else:
                for py in range(end_y, start_y + 1):
                    bitmap[py, column] = 255  # Set to white (eliminate)
            in_group = False
    
    # Update bitmap to remove eliminated groups
    eliminated_groups = [group for group in groups if group not in final_proposed_groups]
    for group in eliminated_groups:
        start_y, end_y, _ = group
        for y in range(min(start_y, end_y), max(start_y, end_y) + 1):
            bitmap[y, column] = 255  # Set to white (eliminate)

    # Then return as before
    return len(final_proposed_groups), final_proposed_groups

def initialize_curve_thicknesses(bitmap, start_points, end_points):
    right_start = max(point[0] for point in start_points)
    left_end = min(point[0] for point in end_points)
    
    sample_range = left_end - right_start
    num_samples = 10
    sample_step = max(1, sample_range // num_samples)
    
    expected_groups = len(start_points)
    valid_thicknesses = []
    
    for x in range(right_start, left_end, sample_step):
        group_count, groups = identify_pixel_groups(bitmap, x, 1, start_points, end_points, [])  # Use empty list for prev_groups
        
        if group_count == expected_groups:
            avg_thickness = sum(group[2] for group in groups) / group_count
            valid_thicknesses.append(avg_thickness)
    
    if valid_thicknesses:
        return sum(valid_thicknesses) / len(valid_thicknesses)
    else:
        return None  # Or some default value

# Initialize curve thicknesses
initial_thickness = initialize_curve_thicknesses(bitmap, start_points, end_points)
if initial_thickness is None:
    print("Failed to initialize curve thicknesses")
    exit(1)

print(f"Initial average curve thickness: {initial_thickness:.2f}")

# Initialize variables for the main analysis
left_x = min(point[0] for point in start_points)
right_x = max(point[0] for point in end_points)
prev_group_count = len(start_points)
avg_heights = [initial_thickness] * len(start_points)
intersection_starts = []
intersection_ends = []
in_intersection = False
prev_groups = []

# Analyze columns from left to right
for x in range(left_x, right_x + 1):
    # Find the minimum of the two moving averages
    min_avg = min(avg_heights)
    
    # Set min_height to 60% of the minimum average
    min_height = max(1, int(min_avg * 0.6))

    group_count, groups = identify_pixel_groups(bitmap, x, min_height, start_points, end_points, prev_groups)
    
    # Update running averages of heights
    if group_count == len(start_points):
        for i, group in enumerate(groups):
            avg_heights[i] = (avg_heights[i] * 0.1 + group[2] * 0.9)  # Weighted average
    
    # Check for intersection start
    if not in_intersection and group_count < prev_group_count and prev_group_count > 0:
        if groups:
            max_height = max(group[2] for group in groups)
            avg_height = sum(avg_heights) / len(avg_heights) if avg_heights else initial_thickness
            
            if max_height > 1.8 * avg_height:
                intersection_starts.append(x)
                in_intersection = True
                print(f"Intersection start detected at column {x}")
                save_debug_image(bitmap, f'debug_artifact_detected_column_{x}.jpg')
    
    # Check for intersection end
    elif in_intersection and group_count > 1:
        max_height = max(group[2] for group in groups)
        if max_height < 1.5 * initial_thickness:  # Adjust this threshold as needed
            intersection_ends.append(x)
            in_intersection = False
            print(f"Intersection end detected at column {x}")
            save_debug_image(bitmap, f'debug_artifact_cleaned_column_{x}.jpg')
    
    prev_group_count = group_count
    prev_groups = groups  # Store current groups for next iteration

# After the forward pass, initialize variables for the reverse pass
right_x = max(point[0] for point in end_points)
left_x = min(point[0] for point in start_points)
avg_heights_backward = avg_heights.copy()  # Use the final averages from the forward pass
prev_groups = []

print("\n\nEnding points:")
for x, y in end_points:
    print(f"  ({x}, {y})")

print(f"\n*** Starting reverse pass from x={right_x} to x={left_x} ***")
print(f"Initial avg_heights_backward: {avg_heights_backward}")

# Create a set of ending y-coordinates for quick lookup
end_y_coords = set(y for _, y in end_points)

# Reverse pass: Analyze first 1000 columns from right to left
for i, x in enumerate(range(right_x, max(left_x, right_x - 1000), -1)):
    group_count, groups = identify_pixel_groups(bitmap, x, 1, start_points, end_points, prev_groups)
    
    # Check for end point connectivity only if we're in the same column as an ending point
    end_connected_groups = []
    other_groups = []
    for group in groups:
        if x in [point[0] for point in end_points]:
            start_y, end_y, _ = group
            group_y_set = set(range(min(start_y, end_y), max(start_y, end_y) + 1))
            if group_y_set.intersection(end_y_coords):
                end_connected_groups.append(group)
                continue
        other_groups.append(group)
    
    # Apply secondary filter to all groups not connected to ending points
    filtered_groups = end_connected_groups.copy()
    for group in other_groups:
        if apply_secondary_filter(group, prev_groups):
            filtered_groups.append(group)
        else:
            # Remove the group from the bitmap and log
            start_y, end_y, _ = group
            for y in range(min(start_y, end_y), max(start_y, end_y) + 1):
                bitmap[y, x] = 255  # Set to white (eliminate)
            print(f"\nColumn {x}: Secondary filter removed a group {group}")
    
    # Update bitmap to remove eliminated groups
    eliminated_groups = [group for group in groups if group not in filtered_groups]
    for group in eliminated_groups:
        start_y, end_y, _ = group
        for y in range(min(start_y, end_y), max(start_y, end_y) + 1):
            bitmap[y, x] = 255  # Set to white (eliminate)
    
    # Check if we're at an intersection boundary
    in_intersection = any(right <= x <= left for left, right in zip(intersection_starts, intersection_ends))
    
    # Update running averages of heights if not in an intersection
    if len(filtered_groups) == len(end_points) and not in_intersection:
        for j, group in enumerate(filtered_groups):
            avg_heights_backward[j] = (avg_heights_backward[j] * 0.1 + group[2] * 0.9)  # Weighted average
            
        # Check bitmap status for each group
        # for group in groups:
        #     start_y, end_y, _ = group
        #     group_pixels = [bitmap[y, x] for y in range(min(start_y, end_y), max(start_y, end_y) + 1)]
        #     is_eliminated = all(pixel == 255 for pixel in group_pixels)
        #     if is_eliminated:
        #         print(f"  Group {group} eliminated from bitmap: {is_eliminated}")
    
    # Store the filtered groups for the next iteration
    prev_groups = filtered_groups

    if len(filtered_groups) > len(end_points):
        print(f'\n*** Too Many Groups in Column {x}')
        print(f'Groups: {filtered_groups}')
        break

print("\nReverse pass completed")
print(f"Final avg_heights_backward: {avg_heights_backward}")

# After the reverse pass, update the contour image
contour_img_updated = np.ones_like(img) * 255
cv2.drawContours(contour_img_updated, filtered_contours, -1, (0, 0, 255), line_thickness)

# Redraw the circles for start and end points
for point in start_points:
    cv2.circle(contour_img_updated, point, 10, (0, 165, 255), -1)
for point in end_points:
    cv2.circle(contour_img_updated, point, 10, (255, 255, 0), -1)

# Get image height for drawing intersection lines
image_height = contour_img_updated.shape[0]

# Redraw intersection lines
for left, right in zip(intersection_starts, intersection_ends):
    cv2.line(contour_img_updated, (left, 0), (left, image_height), (0, 0, 0), 2)  # Black line for left side
    cv2.line(contour_img_updated, (right, 0), (right, image_height), (255, 0, 255), 2)  # Magenta line for right side

# Save the final updated bitmap for debugging
save_debug_image(bitmap, 'debug_final_bitmap_after_reverse_pass.jpg')

# After the reverse pass

# Initialize anomaly detection
is_anomaly = False
anomaly_start_columns = []
anomaly_end_columns = []
anomaly_start_groups = []
anomaly_end_groups = []

# Use the final avg_heights_backward from the reverse pass
running_avg_heights = avg_heights_backward.copy()

# Start from the leftmost starting point
left_x = min(point[0] for point in start_points)
right_x = max(point[0] for point in end_points)

print("\n*** Starting anomaly detection pass ***")
print(f"Initial running_avg_heights: {running_avg_heights}")
print(f"Start points: {start_points}")

# Create a set of starting point coordinates for quick lookup
start_point_coords = set(start_points)

prev_groups = []

# Anomaly detection pass
for x in range(left_x, right_x + 1):
    
    if not any(start <= x <= end for start, end in zip(intersection_starts, intersection_ends)):
        group_count, groups = identify_pixel_groups(bitmap, x, 1, start_points, end_points, [])
        
        if group_count == len(start_points):
            all_within_range = True
            for i, group in enumerate(groups):
                start_y, end_y, height = group
                is_start_point = (x, start_y) in start_point_coords or (x, end_y) in start_point_coords
                
                if not is_anomaly and height > running_avg_heights[i] * 1.8 and not is_start_point:  # 80% increase
                    is_anomaly = True
                    anomaly_start_columns.append(x)
                    anomaly_start_groups.append((prev_groups, groups))
                    print(f"  Anomaly start detected! Group height: {height}, Running avg: {running_avg_heights[i]}")
                
                # Check if group is within 20% of running average
                if abs(height - running_avg_heights[i]) > 0.2 * running_avg_heights[i]:
                    all_within_range = False
            
            if is_anomaly and all_within_range:
                anomaly_end_columns.append(x)
                anomaly_end_groups.append((prev_groups, groups))
                print(f"  Anomaly end detected at column {x}")
                is_anomaly = False
        
        if not is_anomaly:
            # Update running averages if not in anomaly and not a starting point
            for i, group in enumerate(groups):
                start_y, end_y, height = group
                is_start_point = (x, start_y) in start_point_coords or (x, end_y) in start_point_coords
                if not is_start_point:
                    old_avg = running_avg_heights[i]
                    running_avg_heights[i] = running_avg_heights[i] * 0.1 + height * 0.9
        prev_groups = groups

print("\n*** Anomaly detection pass completed ***")
if anomaly_start_columns:
    for i, (start, end) in enumerate(zip(anomaly_start_columns, anomaly_end_columns)):
        print(f"\nAnomaly {i+1}:")
        print(f"Start column: {start}")
        print(f"Previous groups at start: {anomaly_start_groups[i][0]}")
        print(f"Current groups at start: {anomaly_start_groups[i][1]}")
        print(f"End column: {end}")
        print(f"Previous groups at end: {anomaly_end_groups[i][0]}")
        print(f"Current groups at end: {anomaly_end_groups[i][1]}")
else:
    print("No anomalies detected")

# Draw blue lines for anomaly starts and green lines for anomaly ends
for start in anomaly_start_columns:
    cv2.line(contour_img_updated, (start, 0), (start, image_height), (255, 0, 0), 2)
for end in anomaly_end_columns:
    cv2.line(contour_img_updated, (end, 0), (end, image_height), (0, 255, 0), 2)

# After the anomaly detection pass and before drawing the lines

print("\n*** Starting smoothing process ***")

if anomaly_start_columns:
    for region_index, (start_column, end_column) in enumerate(zip(anomaly_start_columns, anomaly_end_columns)):
        # Define the columns for gradient calculation
        col_start = start_column - 21
        col_end = start_column - 1  # Column B

        # Get groups for both columns
        _, groups_start = identify_pixel_groups(bitmap, col_start, 1, start_points, end_points, [])
        _, groups_end = identify_pixel_groups(bitmap, col_end, 1, start_points, end_points, [])

        # Extract the relevant groups using anomalous_group_index
        anomalous_group_index = next(i for i, (start, end) in enumerate(zip(anomaly_start_groups[region_index][0], anomaly_start_groups[region_index][1]))
                                 if end[2] > start[2] * 1.8)
        group_start = groups_start[anomalous_group_index]
        group_end = groups_end[anomalous_group_index]

        # Calculate average gradient
        gradient_top = (group_end[0] - group_start[0]) / 20
        gradient_bottom = (group_end[1] - group_start[1]) / 20

        # Get the starting point from column B (just before the anomaly starts)
        start_top_y, start_bottom_y, _ = group_end

        print(f"\nSmoothing Anomalous Region {region_index + 1}: columns {start_column} to {end_column}")
        print(f"Anomalous group index: {anomalous_group_index}")
        print(f"Start y-range: {start_top_y} to {start_bottom_y}")
        print(f"Gradient (top, bottom): {gradient_top}, {gradient_bottom}")

        print("\nProcessing columns:")

        # Iterate through the columns of the anomalous region, excluding the end column
        for col in range(start_column, end_column):
            # Calculate steps from column B
            steps = col - col_end

            # Apply linear interpolation
            predicted_top_y = round(start_top_y + gradient_top * steps)
            predicted_bottom_y = round(start_bottom_y + gradient_bottom * steps)
            predicted_height = abs(predicted_bottom_y - predicted_top_y) + 1

            # Get current groups for this column
            _, current_groups = identify_pixel_groups(bitmap, col, 1, start_points, end_points, [])

            # Extract the anomalous group using the anomalous_group_index
            anomalous_group = current_groups[anomalous_group_index]
            actual_top_y, actual_bottom_y, actual_height = anomalous_group

            # Ensure correct order (smaller y value first)
            predicted_min_y, predicted_max_y = min(predicted_top_y, predicted_bottom_y), max(predicted_top_y, predicted_bottom_y)
            actual_min_y, actual_max_y = min(actual_top_y, actual_bottom_y), max(actual_top_y, actual_bottom_y)

            # Initialize pixels_changed
            pixels_changed = 0

            # Modify pixels and count changes
            for y in range(actual_min_y, actual_max_y + 1):
                if y < predicted_min_y or y > predicted_max_y:  # OUTSIDE the predicted range
                    if bitmap[y, col] == 0:  # Only change if the pixel is currently black
                        bitmap[y, col] = 255  # Set to white
                        pixels_changed += 1

            print(f"\nColumn {col}:")
            print(f"  Predicted group: ({predicted_min_y}, {predicted_max_y}, {predicted_height})")
            print(f"  Anomalous group: ({actual_min_y}, {actual_max_y}, {actual_height})")
            print(f"  Pixels changed: {pixels_changed}")

        print(f"\nSmoothing process completed for Region {region_index + 1}.")

    print("\nAll anomalous regions processed.")

    # Save the final bitmap image for debugging
    final_bitmap_path = os.path.join(script_dir, 'final_smoothed_bitmap.jpg')
    cv2.imwrite(final_bitmap_path, bitmap)
    print(f"Final smoothed bitmap saved to: {final_bitmap_path}")

else:
    print("No anomalies detected, no smoothing necessary.")

# Now save the final image with all anomaly markings
final_output_path = os.path.join(script_dir, 'final_result_with_anomalies.jpg')
cv2.imwrite(final_output_path, contour_img_updated)
print(f"Final image with anomaly detections saved to: {final_output_path}")

# After all the previous processing, add this code

print("\n*** Starting curve tracing process ***")

def trace_curve(start_point, bitmap, intersection_starts, intersection_ends):
    curve = [start_point]
    x, y = start_point

    # Get the initial group
    _, initial_groups = identify_pixel_groups(bitmap, x, 1, start_points, end_points, [])
    current_group = next((group for group in initial_groups if y in range(min(group[0], group[1]), max(group[0], group[1]) + 1)), None)

    if current_group is None:
        print(f"Warning: No valid starting group found for point {start_point}")
        return curve

    while x < width - 1:
        x += 1
        # Check if we're entering an intersection
        intersection_index = next((i for i, start in enumerate(intersection_starts) if start == x), None)
        
        if intersection_index is not None:
            # We're entering an intersection, use prediction
            intersection_start = intersection_starts[intersection_index]
            intersection_end = intersection_ends[intersection_index]
            
            # Calculate gradient
            col_start = max(0, intersection_start - 11)
            col_end = intersection_start - 1
            _, groups_start = identify_pixel_groups(bitmap, col_start, 1, start_points, end_points, [])
            _, groups_end = identify_pixel_groups(bitmap, col_end, 1, start_points, end_points, [])
            
            curve_index = start_points.index(start_point)
            group_start = groups_start[curve_index % len(groups_start)]
            group_end = groups_end[curve_index % len(groups_end)]
            
            gradient_top = (group_end[0] - group_start[0]) / (col_end - col_start)
            gradient_bottom = (group_end[1] - group_start[1]) / (col_end - col_start)
            
            print(f"Curve starting at {start_point}:")
            print(f"  Intersection at x={intersection_start}")
            print(f"  Gradient calculation:")
            print(f"    Start column: {col_start}, End column: {col_end}")
            print(f"    Start group: {group_start}, End group: {group_end}")
            print(f"    Calculated gradients - Top: {gradient_top:.4f}, Bottom: {gradient_bottom:.4f}")
            
            # Predict through intersection and two columns beyond
            start_top_y, start_bottom_y = group_end[0], group_end[1]
            for col in range(intersection_start, intersection_end + 1):  
                steps = col - col_end
                predicted_top_y = round(start_top_y + gradient_top * steps)
                predicted_bottom_y = round(start_bottom_y + gradient_bottom * steps)
                middle_y = (predicted_top_y + predicted_bottom_y) // 2
                curve.append((col, middle_y))
            
            # Get the last predicted point
            last_predicted_x, last_predicted_y = curve[-1]
            
            # Find the group containing the last predicted point
            _, groups = identify_pixel_groups(bitmap, last_predicted_x, 1, start_points, end_points, [])
            current_group = next((group for group in groups if last_predicted_y in range(min(group[0], group[1]), max(group[0], group[1]) + 1)), None)
            
            if current_group is None:
                print(f"Warning: No valid group found for last predicted point ({last_predicted_x}, {last_predicted_y})")
                return curve
            
            # Move to the next column
            x = last_predicted_x + 1
            
            # Find connected group in the next column
            _, next_groups = identify_pixel_groups(bitmap, x, 1, start_points, end_points, [])
            prev_y_set = set(range(min(current_group[0], current_group[1]), max(current_group[0], current_group[1]) + 1))
            connected_group = next((group for group in next_groups if prev_y_set.intersection(set(range(min(group[0], group[1]), max(group[0], group[1]) + 1)))), None)
            
            if connected_group is None:
                print(f"Warning: No connected group found after intersection at x={x}")
                return curve
            
            # Use the middle pixel of the connected group as the next trace point
            middle_y = (connected_group[0] + connected_group[1]) // 2
            curve.append((x, middle_y))
            current_group = connected_group
            
            continue  # Skip to the next iteration to start normal tracing from the new x

        # Normal tracing
        prev_y_set = set(range(min(current_group[0], current_group[1]), max(current_group[0], current_group[1]) + 1))
        _, current_groups = identify_pixel_groups(bitmap, x, 1, start_points, end_points, [])
        
        connected_group = next((group for group in current_groups if prev_y_set.intersection(set(range(min(group[0], group[1]), max(group[0], group[1]) + 1)))), None)
        
        if connected_group is None:
            break
        
        middle_y = (connected_group[0] + connected_group[1]) // 2
        curve.append((x, middle_y))
        current_group = connected_group
        
        # Check if we've reached an end point
        if (x, middle_y) in end_points:
            break

    return curve

# Use the trace_curve function
# start_point = start_points[0]
additional_colors = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]  # Blue, Green, Cyan, Magenta, Yellow
for j, start_point in enumerate(start_points):
    traced_curve = trace_curve(start_point, bitmap, intersection_starts, intersection_ends)

    # Load the final image
    final_image = cv2.imread(final_output_path)

    if final_image is not None:
        # Draw only this single traced curve
        for i in range(1, len(traced_curve)):
            cv2.line(white_background_image, traced_curve[i-1], traced_curve[i], additional_colors[j % len(additional_colors)], 3)
        for i in range(1, len(traced_curve)):
            cv2.line(original_with_traces, traced_curve[i-1], traced_curve[i], additional_colors[j % len(additional_colors)], 3)


    else:
        print(f"Error: Unable to load the final image from {final_output_path}")
        break

# Display the white background image with traced elements
cv2.imshow('Traced Curves', white_background_image)
cv2.imshow('Original With Traces', original_with_traces)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the white background image with traced elements
cv2.imwrite('white_background_traced_elements.png', white_background_image)
print("White background image with traced elements saved as 'white_background_traced_elements.png'")

