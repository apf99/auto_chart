import cv2
import numpy as np
import pytesseract

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
    # Define the bounding rectangle
    min_x, max_x = min(x_axis[0], x_axis[2]), max(x_axis[0], x_axis[2])
    min_y, max_y = min(y_axis[1], y_axis[3]), max(y_axis[1], y_axis[3])
    
    # Extract the ROI containing the curves
    roi = cleaned_image[min_y:max_y, min_x:max_x]
    
    # Apply Canny edge detection
    edges = cv2.Canny(roi, 50, 150)
    
    # Save the edges image for debugging
    cv2.imwrite('debug_edges.png', edges)
    print("Edges image saved as 'debug_edges.png'")
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw all detected contours for debugging
    debug_all_contours = result_image.copy()
    cv2.drawContours(debug_all_contours[min_y:max_y, min_x:max_x], contours, -1, (255, 0, 0), 2)
    cv2.imwrite('debug_all_contours.png', debug_all_contours)
    print("All detected contours image saved as 'debug_all_contours.png'")
    
    # Filter contours based on length, aspect ratio, and smoothness
    min_contour_length = 30  # Lowered threshold
    filtered_contours = []
    
    for cnt in contours:
        # Calculate contour length
        length = cv2.arcLength(cnt, False)
        
        # Calculate bounding box and aspect ratio
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        
        # Calculate smoothness (number of sharp angles)
        smoothness = 0
        for i in range(1, len(cnt) - 1):
            angle = np.abs(np.arctan2(cnt[i+1][0][1] - cnt[i][0][1], cnt[i+1][0][0] - cnt[i][0][0]) -
                           np.arctan2(cnt[i][0][1] - cnt[i-1][0][1], cnt[i][0][0] - cnt[i-1][0][0]))
            if angle > np.pi / 4:  # Adjust this threshold as needed
                smoothness += 1
        
        # Filter based on length, aspect ratio, and smoothness
        if length > min_contour_length and 0.1 < aspect_ratio < 10.0 and smoothness < 20:
            filtered_contours.append(cnt)
    
    # Draw the filtered contours for debugging
    debug_filtered_contours = result_image.copy()
    cv2.drawContours(debug_filtered_contours[min_y:max_y, min_x:max_x], filtered_contours, -1, (0, 255, 0), 2)
    cv2.imwrite('debug_filtered_contours.png', debug_filtered_contours)
    print("Filtered contours image saved as 'debug_filtered_contours.png'")
    
    # Draw the filtered contours in green on the result image
    cv2.drawContours(result_image[min_y:max_y, min_x:max_x], filtered_contours, -1, (0, 255, 0), 2)
    
    return result_image

print("Script started")

# Load the original image
print("Loading original image...")
original_image = cv2.imread('Chart_Image_1.jpg', cv2.IMREAD_GRAYSCALE)
if original_image is None:
    raise FileNotFoundError(f"Image file 'Chart_Image_1.jpg' not found.")
print("Original image loaded.")

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
    print(f"Drawing bounding box {i}: {bbox}")
    cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
    
    text, confidences, roi = get_text_in_region(original_image, bbox, i)
    print(f"Bounding box {i}: {bbox}")
    print(f"Text detected: '{text}'")
    print(f"Confidence scores: {confidences}")
    
    if text:
        cv2.putText(result_image, text, (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
    else:
        print(f"Warning: No text detected for bounding box {i}")

# Save the intermediate result with axes, tick marks, and measures
cv2.imwrite('result_with_axes_and_measures.png', result_image)
print("Intermediate result saved as 'result_with_axes_and_measures.png'")

# Create a cleaned image for curve tracing
print("Removing axes and ticks...")
cleaned_image = remove_axes_and_ticks(processed_image, x_axis, y_axis)
print("Axes and ticks removed.")

# Apply binary thresholding
print("Applying binary thresholding...")
_, thresholded_image = cv2.threshold(cleaned_image, 128, 255, cv2.THRESH_BINARY_INV)
print("Binary thresholding applied.")

# Save the thresholded image for inspection
cv2.imwrite('thresholded_image.png', thresholded_image)
print("Thresholded image saved as 'thresholded_image.png'")

# Trace the curves on the result_image (which contains all previously detected elements)
print("Tracing curves...")
result_image = trace_curves(cleaned_image, result_image, x_axis, y_axis)
print("Curves traced.")

# Define the ROI based on the min and max locations of the axes
roi_top_left = (min(x_axis[0], x_axis[2]), min(y_axis[1], y_axis[3]))
roi_bottom_right = (max(x_axis[0], x_axis[2]), max(y_axis[1], y_axis[3]))

# Print the coordinates for debugging
print(f"ROI Top Left: ({int(roi_top_left[0])}, {int(roi_top_left[1])})")
print(f"ROI Bottom Right: ({int(roi_bottom_right[0])}, {int(roi_bottom_right[1])})")

# Create a copy of the original image to work on
highlighted_image = result_image.copy()

# Draw the blue bounding box around the ROI
cv2.rectangle(highlighted_image, roi_top_left, roi_bottom_right, (255, 0, 0), 2)  # Blue color in BGR
print("Blue bounding box drawn")

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

# Display the result
cv2.imshow('Traced Curves', highlighted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite('result_with_traced_curves.png', highlighted_image)
print("Result saved as 'result_with_traced_curves.png'")