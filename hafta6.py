import cv2
import matplotlib.pyplot as plt
import numpy as np
my_bgr_img = cv2.imread("./bonus.jpg")
my_gray_img = cv2.cvtColor(my_bgr_img,cv2.COLOR_BGR2GRAY)
print(type(my_bgr_img))
print(type(my_gray_img))
# ilk iş olarak, gri seviyedeki görüntünün eşiğini hesaplayarak binary
# hale getiriyoruz
# bu sayede, kenar tespitini en başarılı şekilde yapıp, bu kenarların
# birleşmesi sonucu kontoru bulmayı hedefliyoruz

my_bgr_img = cv2.imread("./bonus.jpg")
my_gray_img = cv2.imread("./bonus.jpg",0)

ret, thresh = cv2.threshold(my_gray_img,127,255,cv2.THRESH_BINARY)

# ret olarak thresh değeri dönüyor yani 127
# thresh değişkeni de eşiklenmiş görüntüyü saklıyor

# eşiklenmiş görüntüyü elde ettikten sonra konturları bulmaya başlıyoruz
# kontur bulma işleminde de 2 adet değer dönüyor. Konturların hiyerarşisi
# (anne/çocuk ilişkisi) ve konturların listesi. Bize lazım olan değer,
# konturların listesi, yani dönen ilk değişken

contour, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# eşiklenmiş görüntünün (thresh değşkeni) konturlarını bulurken
# RETRive external metodunu kullan, yani sadece dış kenarları al
# bunları zincir haline getirirken de (birleştirme işleminde de)
# CHAIN_APPROX_SIMPLE metodunu kullan diyoruz.

cv2.drawContours(my_bgr_img, contour,-1,(0,255,0),2)

# drawContours fonksiyonu, bir önceki aşamada elde edilen konturların listesini
# alır, onu renkli görüntü üzerine uygular (koordinat sistemiyle)
# ve orijinal görüntü artık, üzerindeki konturların (kenarların bir zincirle
# birleştirilmesi sonucunda) elde edilmesiyle bitirilir.

# fonksiyonun parametreleri: renkli_görüntü_değişkeni, kontur_listesi,
# -1: konturların tamamını uygula (kesintisiz-hepsi)
# (0,255,0) konturları gösterirken yeşil rengini seç (istediğiniz rengi verebilirsiniz)
# 2: konturları gösteren rengin kalınlık derecesi

cv2.imshow("Image's Contour",my_bgr_img)
cv2.imshow("Thresh Image",thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
my_bgr_img = cv2.imread("./bonus.jpg")
my_gray_img = cv2.imread("./bonus.jpg",0)

ret, thresh = cv2.threshold(my_gray_img,127,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3,3),dtype=np.uint8),iterations=2)

contour, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(my_bgr_img, contour,-1,(0,255,0),2)

cv2.imshow("Image's Contour",my_bgr_img)
cv2.imshow("Thresh Image",thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()