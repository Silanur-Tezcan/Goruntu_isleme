import cv2
import numpy as np
def cropped_black_borders(img:np.ndarray, thr=12, pad=0):
    # thr parametresi, thresh değerini temsil ediyor. Arkaplan siyah
    # renginde olduğu için 0'a çok yakın bir değer verdik.
    # pad parametresi padding'i temsil ediyor. Görüntüyü, arkaplandan ayırt
    # ettikten sonra sol, sağ, üst ve alt taraflarına boşluk bırakılıp
    # bırakılmayacağına karar veriyor. Biz, varsayılan olarak 0 değeri 
    # girdiğimiz için 4 taraftan da tamamen ayırt edilmiş/kesilmiş tam bir
    # görüntü elde etmek istediğimizi belirtiyoruz

    # ilk olarak eğer görüntü renkli bir görüntüyse yani 3 kanallı ise
    # bunun kontrolünü sağlıyoruz:

    if img.ndim==3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # görüntüyü tek kanallı hale çevirdikten sonra ilk olarak
    # arkaplandaki siyah piksellerden ayırt etmek için bir maske çıkarıyoruz

    mask = (gray>thr).astype(np.uint8)*255
    kernel = np.ones((5,5),dtype=np.uint8)

    # yukarıda elimizde bir harita (mask) ve bir matris (filtre matrisi)
    # var artık.

    # siyah alandaki pikseller haricinde kalan kısımların haritasını
    # elde ettikten sonra bu mevcut görüntüye siyah alanlardak iyice
    # ayrılması için (küçük siyah detayların kaybolması için CLOSE yöntemini
    # uyguluyoruz. Sonrasında elimizde temiz bir maske görüntüsü oluşuyor

    
    thresh = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Elimizdeki temiz maske görüntüsü içerisinden konturların listesini
    # elde ediyoruz. Bu sayede, kenarları bir zincir gibi birleştirip
    # tek bir görüntü elde etmeyi amaçlıyoruz.
    
    contour, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                  cv2.CHAIN_APPROX_SIMPLE)

    # Elimize gelen contour değişkeni, görüntüdeki konturların tamamını
    # bir liste olarak tutuyor. Bizim amacımız, bu listedeki en büyük
    # kontur değerini saklayan değişkeni elde etmek

    c = max(contour, key=cv2.contourArea)

    # seçtiğimiz en büyük konturu (0,0) noktasından (x,y) tüm sütun (w) ve 
    # tüm satırları (h) elde edecek şekilde bir dikdörtgene dönüştürüyoruz
    
    x,y,w,h = cv2.boundingRect(c)

    # Kırpma işlemi esnasında görüntünün sınırlarının aşılmaması için
    # görüntünün mevcut yükseklik ve genişlik bilgisini elde etmemiz gerekiyor
    # biz görüntüyü, orijinal görüntü içerisinden crop yaparken sağ,sol,üst ve
    # alt bölgelerden herhangi bir kayma veya aşınma olmasın istiyoruz.

    H, W = img.shape[:2]
    
    # görüntünün yükseklik ve genişliğini elde ettikten sonra cropped
    # işlemine geçiyoruz

    # köşeleri tespit ediyoruz (4 köşe)

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)

    # 4 köşeyi de elde ettikten (taşma olmamasını sağladıktan) sonra
    # ilgili görüntüyü, arkaplandan (orijinal görüntüden) koparıyoruz
    
    return img[y1:y2, x1:x2]

my_bgr_img = cv2.imread("./yeni1.png")

my_cropped_img = cropped_black_borders(my_bgr_img)

cv2.imshow("Cropped Image",my_cropped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()