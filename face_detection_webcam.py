import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

# Chargement du classifieur
face_class = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_class= cv2.CascadeClassifier("haarcascade_eye.xml") 
smile_class = cv2.CascadeClassifier('haarcascade_smile.xml') 

log.basicConfig(filename='face_detection.log',level=log.INFO)

# Lancement de la webcam
video_capture = cv2.VideoCapture(0)
# Initialisation du nombre de visage détectés
n_faces = 0
n_smile=0

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
        minSize=(30,30)
    )

    # Dessin de rectangles autour des visages détectées
    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(frame,(fx,fy),(fx+fw,fy+fh),(255,255,0),2)  
        roi_gray = gray[fy:fy+fh, fx:fx+fw] 
        roi_color = frame[fy:fy+fh, fx:fx+fw] 
  
        # Detection des yeux
        eyes = eye_class.detectMultiScale(roi_gray)  
        for (ex,ey,ew,eh) in eyes: 
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2) 
        # Detection du sourire
        smiles = smile_class.detectMultiScale(roi_gray, 1.8, 20)  
        for (sx,sy,sw,sh) in smiles: 
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,255,0),2) 
    
    # Sauvegarde dans le fichier log du nombre de visage détectés
    if n_faces != len(faces):
        n_faces = len(faces)
        log.info("Nombre de visage détecté : "+str(len(faces))+" à "+str(dt.datetime.now()))

    # Affichage du résultat
    cv2.imshow('Video', frame)

    # Arrêt de l'application si une touche spéciale est appuyée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()

# Allocation mémoire
cv2.destroyAllWindows()
