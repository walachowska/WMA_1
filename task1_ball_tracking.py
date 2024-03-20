import numpy as np
import cv2


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


image = cv2.imread('ball.png')
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
combined_mask = combine_masks(hsv_image)
# cv2.imshow('combined', combined_mask)
final_mask = add_morphological_operations(combined_mask)
# cv2.imshow('final_mask', final_mask)
final_image = marker_the_center(image, final_mask)
cv2.imshow('final_image', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()