import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

# Chargement du classifieur
path_class = "haarcascade_frontalface_default.xml"
face_class = cv2.CascadeClassifier(path_class)
log.basicConfig(filename='face_detection.log',level=log.INFO)

# Lancement de la webcam
video_capture = cv2.VideoCapture(0)
# Initialisation du nombre de visage détectés
n_faces = 0

while True:
    if not video_capture.isOpened():
        print('Impossible d\'ouvrir la webcam. Veuillez réessayer')
        sleep(10)
        pass

    # Récupération de chaque frame de la vidéo
    ret, frame = video_capture.read()
    # Conversion en niveau de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Détection des visages grâce au classifieur
    faces = face_class.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(25, 25)
    )

    # Dessin de rectangles autour des visages détectées
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 128, 0), 2)
    # Sauvegarde dans le fichier log du nombre de visage détectés
    if n_faces != len(faces):
        n_faces = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))
        print (" Nombre de visage détecté : {0}".format(len(faces)))

    # Affichage du résultat
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
