import cv2
import numpy as np


def combine_masks(hsv_image):
    # BALL
    lower_ball = np.array([0, 100, 100])
    upper_ball = np.array([255, 255, 255])
    ball_mask = cv2.inRange(hsv_image, lower_ball, upper_ball)
    # LIGHT
    lower_light = np.array([120, 0, 0])
    upper_light = np.array([252, 255, 255])
    light_mask = cv2.inRange(hsv_image, lower_light, upper_light)
    # OR
    combined_mask = cv2.bitwise_or(ball_mask, light_mask)
    return combined_mask


def add_morphological_operations(combined_mask):
    kernel = np.ones((15, 15), np.uint8)
    mask_opened = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('after_open', mask_opened)
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('after_closed', mask_closed)
    return mask_closed


def marker_the_center(original_image, mask):
    contours, hierarchy = cv2.findContours(mask, 1, 2)
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
video.open('movingball.mp4')
# liczba klatek
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)
result = cv2.VideoWriter('result.avi',  cv2.VideoWriter_fourcc(*'MJPG'), 20, size)

# frame_counter = 1
while True:
    success, frame = video.read()
    if not success:
        break
    # print('klatka {} z {}'.format(frame_counter, total_frames))
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    combined_mask = combine_masks(hsv_image)
    final_mask = add_morphological_operations(combined_mask)
    frame_with_marker = marker_the_center(frame, final_mask)
    result.write(frame_with_marker)
    # counter = counter + 1

video.release()
result.release()