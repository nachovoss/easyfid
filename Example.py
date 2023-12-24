#-*- coding:utf-8 -*-]
from easyfid_master import FaceRecognizer

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
