import cv2
import numpy as np
img_bgr = cv2.imread("./plaka.jpg")
img_gray = cv2.imread("./plaka.jpg",0)

# ilk iş olarak elimizdeki görüntülere yumuşatma/blurlama tekniği
# uyguluyoruz. Burada amaç, görüntü içerisinde bir kenar tespiti yapmadan
# önce, görüntündeki küçük detayların yok edilmesini sağlamaktır

# Görüntüye Gauss yumuşatma tekniği uyguluyoruz. Bu fonksiyon, Gauss dağılımına
# uygun bir mantıkta çalışır. Şöyle ki, kendi oluşturduğu 5x5 boyutundaki bir
# matris ile görüntünün 0,0 noktasından başlayarak son noktasına kadar bu
# 5x5'lik matrisle ilerler ve her bir pikselin değerini, kendisine en yakın
# 25 pikselin Gauss dağılımına göre hesaplanmasıyla yeniden değerlendirilir
# fonksiyonda en sonda verdiğimiz 0 değeri de sigmaX parametresine karşılık
# gelir. X koordinat sisteminde Gauss Dağılımına göre standart sapma değerini
# temsil eder. Biz burada 0 değeri vererek, pikselin oluşacak yeni değerinin
# tamamen kernel_size içerisindeki değerlendirmeden gelmesini sağladık.
# Eğer bu değeri artırırsanız daha bulanık bir görüntü elde edersiniz.
blurring_img = cv2.GaussianBlur(img_gray,(5,5),0)

#cv2.imshow("Original Image", img_gray)
#cv2.imshow("Blurring Image", blurring_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Yumuşattığımız/bulanıklaştırdığımız görüntünün kenarlarını tespit etmeye

# Canny, ilk kenar tespiti yapan algoritmalar arasında yer alır. Aşağıdaki
# fonksiyonu, gri seviye bir görüntünün kenarlarını 50 ve 150 thresh değerleri
# arasında tespit eder. Nasıl Çalışır?
# ilk olarak küçük thresh değerini, sonrasında da büyük thresh değerini alır
# küçük thresh değerinin altında kalan pikselleri görmezden gelir. yani, bu
# pikseller asla kenar sayılmaz.

# Büyük piksel değerinden büyük olan pikseller de kesinlikle kenar kabul edilir
# bu iki değer arasında kalan piksel değerleri ise, etrafındaki komşu
# piksellerin durumuna göre ya kenar kabul edilir ya da kenar kabul edilmez.
# komşu pikseller eğer büyük piksel değerinden büyükse kenar kabul edilir
# değilse, yok edilir

kenar = cv2.Canny(blurring_img, 50, 150)

#cv2.imshow("Original Image", img_gray)
#cv2.imshow("Blurring Image", blurring_img)
#cv2.imshow("Edge Detection", kenar)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Kenarları da tespit ettikten sonra konturları bulmaya başlıyoruz

konturlar, _ = cv2.findContours(kenar, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

# konturların listesini elde ettikten sonra, mevcut kontur listesi
# içerisine girip, bir formül yazacağız. Eğer bizim yazdığımız formüle
# uyan bir kontur varsa, onun etrafını yeşil dikdörtgen ile seçili hale
# getireceğiz. NOT: eğer tüm konturlar arasında bizim yazdığımız formüle
# yani bir plakaya benzer formüle uyan kontur olursa o da plaka gibi seçilecek

for kontur in konturlar:
    # ilk olarak, elimizdeki şu anki mevcut konturu bir dikdörtgen içerisine
    # alıyoruz. Dolayısıyla bunun bir başlangıç noktası (0,0) (x,y),
    # bir genişliği (sütun sayısı), bir de yüksekliği (satır sayısı) olmalı
    # yani bu işlemden bize 4 adet değer dönmeli

    x,y,w,h = cv2.boundingRect(kontur)

    # çizdiğimiz bu dikdörtgene için bir oran hesaplaması yapıyoruz
    # genişlik ile yükseklik arasında. Bir plakanın genişliği, yüksekliğine
    # nazaran çok daha fazladır. Dolayısıyla bir bilme işlemi yapıp, 
    # sonucun 1'den büyük float tipinde olmasını bekliyoruz.

    oran = w / float(h)

    # kontur, bir değer olarak çizilen sınırlar içerisinde bir alana sahiptir
    # Aşağıdaki kod bloku ile elimizdeki mevcut konturun alan bilgisini 
    # hesaplıyoruz. Plaka için yazacağımız formülde işimize yarayacak!!!
    alan = cv2.contourArea(kontur)

    if 1<oran<4 and 100<w<300 and 15<h<100 and alan>200:
        cv2.rectangle(img_bgr,(x,y),(x+w,y+h),(0,255,0),3)

cv2.imshow("Plaka Tespiti", img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()