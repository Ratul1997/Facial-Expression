import cv2
import numpy as np
import dlib
import pickle

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

recognizer = cv2.face.FisherFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"person_name":1}
with open("labels.pickle", "rb") as f:
        og_labels = pickle.load(f)
        labels = {v:k for k,v in og_labels.items()}

# # img = cv2.imread('images/Rukon/10.jpg')
# # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # faces = detector(gray)

# for face in faces:
#     x1 = face.left()
#     y1 = face.top()
#     x2 = face.right()
#     y2 = face.bottom()   
#      #cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
#     for i in range(0,68):
#         landmarks = predictor(gray,face)
#         x= landmarks.part(i).x
#         y = landmarks.part(i).y
#         cv2.circle(img,(x,y),3,(0,255,0),-1)        
while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        faces = detector(gray)
        
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()   
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)

            data = []
            for i in range(0,68):
                landmarks = predictor(gray,face)
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                data.append((x,y))

                cv2.circle(frame,(x,y),1,(0,255,0),-1)


            id_,conf = recognizer.predict(np.array(data))
            
            print(conf,labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame,name,(x1,y1-100),font,1,color,stroke,cv2.LINE_AA)
            
                    
            
            # print(conf)    

        cv2.imshow("test", frame)
         
        if not ret:
            break
        k = cv2.waitKey(30)
         
        
        
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
            

 

# cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()