import cv2
import torchvision 
import numpy as np


def warp_and_crop_face(raw_img, facial5points): 

    #reference_5pts = np.float32(reference_5pts) 
    src_5pts = np.float32(facial5points) 

    desired_face_width, desired_face_height = 224, 224 
    desiredLeftEye = (0.25, 0.25) 

    leftEyeCenter = src_5pts[0] 
    print(leftEyeCenter)
    #raw_img =cv2.circle(raw_img, (leftEyeCenter[0], leftEyeCenter[1]), 5, (255, 0,0), 1)
    rightEyeCenter = src_5pts[1]
        # compute the angle between the eye centroids
    print(rightEyeCenter)
    #raw_img =cv2.circle(raw_img, (rightEyeCenter[0], rightEyeCenter[1]), 5, (255, 0,0), 1)
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    
    angle = np.degrees(np.arctan2(dY, dX)) 

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
    desiredRightEyeX = 1.0 - desiredLeftEye[0] 

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    desiredDist *= desired_face_width
    scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,(leftEyeCenter[1] + rightEyeCenter[1]) // 2)
        # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
        # update the translation component of the matrix
    tX = desired_face_width * 0.5
    tY = desired_face_height * desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
    (w, h) = (desired_face_width, desired_face_height)
    output = cv2.warpAffine(raw_img, M, (w, h),flags=cv2.INTER_CUBIC)
        # return the aligned face
    return output 


def align_face(raw_img, facial5points) : 
    '''
    REFERENCE_FACIAL_POINTS = [
    [16.0, 16.0],
    [32.0, 16.0],
    [24.0, 24.73660278],
    [33.54930115, 92.3655014],
    [62.72990036, 92.20410156]]
    '''
    #crop_size = (image_h, image_w) 

    #facial5points = np.reshape(facial5points, (2,5)) # reshape image with (2,5)

    # get the reference 5 landmarks position in the crop settings 
    #reference_5pts = np.array(REFERENCE_FACIAL_POINTS)  

    #reference_5pts = get_reference_facial_points(output_size= (48, 48), inner_padding_factor= 0.25, outer_padding= (0, 0), default_square= True) 

    #dst_img = warp_and_crop_face(raw_img, facial5points , reference_5pts = reference_5pts, crop_size = crop_size)
    dst_img = warp_and_crop_face(raw_img, facial5points)

    return dst_img 