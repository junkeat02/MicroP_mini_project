import cv2
import pathlib

cascade_path = "C:/Users/USER/Documents/MicroP/face_detection/haarcascade_frontalface_default.xml"

clf = cv2.CascadeClassifier(str(cascade_path))

camera = cv2.VideoCapture(0)

if camera.isOpened():
# get vcap property 
    width  = camera.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    # or
    width  = camera.get(3)  # float `width`
    height = camera.get(4)  # float `height`

# it gives me 0.0 :/
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    fps = camera.get(cv2.CAP_PROP_FPS)
    _, frame = camera.read()
    cv2.putText(frame, str(fps), (10, 30), font, 1, (255, 255, 0), 1, cv2.LINE_AA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 255, 0), 2)
        cv2.putText(frame, str(x+width/2), (x, y - 10), font, 1, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow("Faces", frame)
    if cv2.waitKey(1) == ord("q"):
        break
    
camera.release()
cv2.destroyAllWindows()