import cv2
import numpy as np
from PIL import Image
import os

class FaceRecognizer:
    def __init__(self):
        self.camera_width = 1920
        self.camera_height = 1080
        self.dataset_path = 'dataset'
        self.trainer_path = 'trainer/trainer.yml'
        self.cascade_path = 'haarcascade_frontalface_default.xml'
        self.face_detector = cv2.CascadeClassifier(self.cascade_path)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.cam = cv2.VideoCapture(0)
        self.initialize_camera()

    def initialize_camera(self):
        self.cam.set(3, self.camera_width)
        self.cam.set(4, self.camera_height)

    def capture_faces(self, imgs_per_person, n_users):
        # import ipdb; ipdb.set_trace()
        for user_id in range(n_users):
            input(f'\nUser {user_id+1} press <return> to start capturing images')
            for i in range(imgs_per_person):
                ret, img = self.cam.read()
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.imwrite(f"{self.dataset_path}/User.{user_id+1}.{i}.jpg", gray[y:y + h, x:x + w])
                    cv2.imshow('image', img)
                if cv2.waitKey(100) & 0xff == 27:  # ESC key
                    break
        print("\n Finished capturing data.")

    def train_faces(self):
        image_paths = [os.path.join(self.dataset_path, f) for f in os.listdir(self.dataset_path)]
        face_samples, ids = [], []
        for image_path in image_paths:
            pil_img = Image.open(image_path).convert('L')
            img_numpy = np.array(pil_img, 'uint8')
            id = int(os.path.split(image_path)[-1].split(".")[1])
            faces = self.face_detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)
        print("\n Training faces. It will take a few seconds. Wait ...")
        self.recognizer.train(face_samples, np.array(ids))
        self.recognizer.write(self.trainer_path)
        print(f"\n {len(np.unique(ids))} faces trained.")

    def recognize_faces(self, n_names, names):
        self.recognizer.read(self.trainer_path)
        font = cv2.FONT_HERSHEY_TRIPLEX
        while True:
            ret, img = self.cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,
                                                        minSize=(int(self.camera_width * 0.1), int(self.camera_height * 0.1)))
            for (x, y, w, h) in faces:
                id, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])
                # import ipdb; ipdb.set_trace()
                id = names[id] if confidence < 100 else "unknown"
                confidence_text = f"{round(100 - confidence)}%"
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                cv2.putText(img, confidence_text, (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
            cv2.imshow('camera', img)
            if cv2.waitKey(10) & 0xff == 27:  # ESC key
                break

        print("\n Exiting Program and cleanup stuff")
        self.cam.release()
        cv2.destroyAllWindows()

def main():
    face_recognizer = FaceRecognizer()
    imgs_per_person = int(input("Enter number of pictures per person: "))
    n_users = int(input("Enter number of users: "))
    face_recognizer.capture_faces(imgs_per_person, n_users)
    face_recognizer.train_faces()
    n_names = int(input("Enter number of users for recognition: "))
    names = ['None'] + [input(f'User {i} enter your name: ') for i in range(1, n_names + 1)]
    face_recognizer.recognize_faces(n_names, names)

if __name__ == "__main__":
    main()
