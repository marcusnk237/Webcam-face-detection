import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

# Chargement du classifieur
face_path = "haarcascade_frontalface_default.xml"
face_class = cv2.CascadeClassifier(face_path)
eye_path= "haarcascade_eye.xml"
eye_class= cv2.CascadeClassifier(eye_path) 
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
    for (fx, fy, fw, fh) in faces:
        # To draw a rectangle in a face  
        cv2.rectangle(frame,(fx,fy),(fx+fw,fy+fh),(255,255,0),2)  
        roi_gray = gray[fy:fy+fh, fx:fx+fw] 
        roi_color = frame[fy:fy+fh, fx:fx+fw] 
  
        # Detects eyes of different sizes in the input image 
        eyes = eye_class.detectMultiScale(roi_gray)  
  
        #To draw a rectangle in eyes 
        for (ex,ey,ew,eh) in eyes: 
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2) 
    
    # Sauvegarde dans le fichier log du nombre de visage détectés
    if n_faces != len(faces):
        n_faces = len(faces)
        log.info("Nombre de visage détecté : "+str(len(faces))+" à "+str(dt.datetime.now()))
        print (" Nombre de visage détecté : {0}".format(len(faces)))

    # Affichage du résultat
    cv2.imshow('Video', frame)

    # Arrêt de l'application si une touche spéciale est appuyée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()

# Allocation mémoire
cv2.destroyAllWindows()
