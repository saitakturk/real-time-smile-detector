

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
from dlib import get_frontal_face_detector
# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", required=True,
                help="path to pre-trained smile detector CNN")
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
args = vars(ap.parse_args())
#HOG frontal face detector
detector = get_frontal_face_detector()
#load model
model = load_model(args["model"])
# If a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
# Otherwise, load the video
else:
    camera = cv2.VideoCapture(args["video"])
while True:
    # Grab the current frame
    (grabbed, frame) = camera.read()
    
    # If we are viewing a video and we did not grab a frame, then we have reached the end of the video
    if args.get("video") and not grabbed:
        break
    #resize frame to get more fps
    frame_resize = imutils.resize(frame, width=300)
    #turn frame gray_scale because HOG detector does'not need rgb image
    frame_gray_scale = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
    frame_clone = frame.copy()
  
    # Detect faces in the input frame, then clone the frame so that we can draw on
    faces = detector(frame_gray_scale, 1)    
  
    #faces that found by HOG detector    
    # Loop over the face bounding boxes
    for d in faces:
        #sometimes HOG deterctor giving zero width frame that causes some errors
        try:
            top = d.top() 
            left = d.left()
            bottom = d.bottom() 
            right = d.right()
            # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
            # the ROI for classification via the CNN
            roi = frame_resize[top:top + bottom, left:left + right]
        
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            #Since frame resized to get more fps and we are showing original frame
            top = int(top * 1.15)
            left = int(left *1.5)
            bottom = int(bottom * 1.6)
            right = int(right * 1.6)

            # Determine the probabilities of both "smiling" and "not smiling", then set the label accordingly
            (not_smiling, smiling) = model.predict(roi)[0]
            label = "SMILING : {}".format(smiling) if smiling > not_smiling else "NEUTRAL : {}".format(not_smiling)

            # Display the label and bounding box rectangle on the output frame
            cv2.putText(frame_clone, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frame_clone, (left, top), (left + right, top + bottom), (0, 0, 255), 2)
        except:
            pass
    # Show our detected faces along with smiling/not smiling labels
    cv2.imshow("Face", frame_clone)

    # If the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows() 
