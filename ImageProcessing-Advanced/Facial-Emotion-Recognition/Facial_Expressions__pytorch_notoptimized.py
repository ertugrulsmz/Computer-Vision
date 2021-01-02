import torch
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier( './model/haarcascade_frontalface_default.xml' )
classifier = torch.load( "./model/model_3654" )
classifier.eval()

class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

cap = cv2.VideoCapture( 0 )

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor( frame, cv2.COLOR_BGR2RGB )
    faces = face_classifier.detectMultiScale( gray, 1.3, 5 )

    for (x, y, w, h) in faces:

        cv2.rectangle( frame, (x, y), (x + w, y + h), (0, 255, 255), 2 )
        roi_gray = gray[y:y + h, x:x + w]

        # y axis in math refers to row, x refers to column sooo ...

        roi_gray = cv2.resize( roi_gray, (48, 48), interpolation=cv2.INTER_AREA )

        if np.sum( [roi_gray] ) != 0:
            roi = roi_gray.astype( 'float' ) / 255.0
            roi = img_to_array( roi )
            roi = np.expand_dims( roi, axis=0 )
            roi = torch.tensor( roi )
            roi = roi.permute( 0, 3, 1, 2 )

            # make a prediction on the ROI, then lookup the class

            preds = classifier(roi)
            preds = preds[0]
            label = class_labels[preds.argmax()]
            label_position = (x, y)
            cv2.putText( frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2 )
        else:
            cv2.putText( frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3 )

    cv2.imshow( 'Emotion Detector', frame )
    if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
        break

cap.release()
cv2.destroyAllWindows()
