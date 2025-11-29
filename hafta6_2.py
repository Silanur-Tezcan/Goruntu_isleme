import cv2
import numpy as np
model_pathh = 'haarcascade_frontalface_default.xml'

face_cascades = cv2.CascadeClassifier(cv2.data.haarcascades + model_pathh)

my_bgr_img = cv2.imread("./deneme.jpg")
my_gray_img = cv2.imread("./deneme.jpg",0)
# Ağırlıklara sahip modelimizin (face_cascades), bir yüzü tanımlarken
# aşağıdaki parametrelere ait değerlere (kurallara) bağlı kalmasını sağladık

faces = face_cascades.detectMultiScale(
    my_gray_img,# işlenecek görüntü (daima gri seviye görüntü)
    scaleFactor = 1.1,#görüntünün yakınlaştırma derecesi
    minNeighbors = 5,# yanlış pozitifleri ayrıştırmak için
    minSize = (5,5)#en küçük yüz boyutu
)

sayac = 0

for (x,y,w,h) in faces:
    cv2.rectangle(my_bgr_img,(x,y),(x+w,y+h),(0,255,0),2)
    sayac += 1

print(f"{sayac} adet futbolcu tespit edildi!!!")
cv2.imshow("Futbolcular", my_bgr_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
help(face_cascades.detectMultiScale)