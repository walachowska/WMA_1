import numpy as np
import cv2


def create_mask(hsv_image):
    lower = np.array([0, 100, 100])
    upper = np.array([100, 255, 255])
    mask = cv2.inRange(hsv_image, lower, upper)
    return mask


def add_morphological_operations(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('after_open', mask_opened)
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('after_closed', mask_closed)
    return mask_closed


def marker_the_center(original_image, mask):
    contours, hierarchy = cv2.findContours(mask, 1, 2)
    if contours:
        M = cv2.moments(contours[0])
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        image_marker = original_image.copy()
        cv2.drawMarker(image_marker, (int(cx), int(cy)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
        return image_marker

video = cv2.VideoCapture()
video.open('Square_Rotation.mp4')
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)
result = cv2.VideoWriter('3_result.avi',  cv2.VideoWriter_fourcc(*'MJPG'), 20, size)

while True:
    success, frame = video.read()
    if not success:
        break
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = create_mask(hsv_image)
    final_mask = add_morphological_operations(mask)
    frame_with_marker = marker_the_center(frame, final_mask)
    result.write(frame_with_marker)

video.release()
result.release()