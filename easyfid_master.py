import cv2
import numpy as np
from PIL import Image
import os


def face_data():
    cam = cv2.VideoCapture(0)
    cam.set(3, 1920)  # video width
    cam.set(4, 1080)  # video height

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    imgs_per_person = int(input("how many pictures per person would you like to collect per user?: "))  #  Take x face pictures per user and stop
    n_users = int(input("how many users would you like to use?:"))
    
    
    for i in range(n_users):
        #  For each person, enter one numeric face id
        face_id = input('\n enter user id end press <return> ==>  ')
        

        print("\n [INFO] Initializing face capture. Look the camera and wait ...")
        #  Initialize individual sampling face count
        

        for i in range(imgs_per_person):

            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                #  Save the captured image into the datasets folder
                cv2.imwrite(f"dataset/User.{str(face_id)}.{str(i)}.jpg", gray[y:y + h, x:x + w])

                cv2.imshow('image', img)

            k = cv2.waitKey(100) & 0xff  #  Press 'ESC' for exiting video
            if k == 27:
                break
            

    #  Release camera and close window
    print("\n Finished capturing data ")
    cam.release()
    cv2.destroyAllWindows()

def training_faces():
# Path for face image database
        path = 'dataset'

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        # function to get the images and label data
        def get_images_and_labels(path):

            image_paths = [os.path.join(path,f) for f in os.listdir(path)]
            face_samples=[]
            ids = []

            for image_path in image_paths:

                pil_img = Image.open(image_path).convert('L') # convert it to grayscale
                img_numpy = np.array(pil_img, 'uint8')

                id = int(os.path.split(image_path)[-1].split(".")[1])
                faces = detector.detectMultiScale(img_numpy)

                for (x, y, w, h) in faces:
                    face_samples.append(img_numpy[y:y + h, x:x + w])
                    ids.append(id)

            return face_samples,ids

        print ("\n Training faces. It will take a few seconds. Wait ...")
        faces, ids = get_images_and_labels(path)
        recognizer.train(faces, np.array(ids))

        # Save the model into trainer/trainer.yml
        recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

        # Print the numer of faces trained and end program
        print(f"\n {len(np.unique(ids))} faces trained. Exiting Program")


def face_id():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascadePath);

    font = cv2.FONT_HERSHEY_TRIPLEX

    n_names = int(input("please enter how many users you want to detect: "))
    
    names = ['None']
    
    for i in range(1, n_names+1):
        name = input(f'User {i} enter your name and press return: ')
        names.append(name)
    print(names)

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 1280)  # set video widht
    cam.set(4, 720)  # set video height

    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:

        ret, img = cam.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)), )

        for (x, y, w, h) in faces:

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            if (confidence < 100):
                id = names[id]
                confidence = f"{round(100 - confidence)}%"
            else:
                id = "unknown"
                confidence = f"{round(100 - confidence)}%"

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)

        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
