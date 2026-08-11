import cv2

detector_facial = cv2.CascadeClassifier(r'C:\Users\labsfiap\PycharmProjects\ReconhecimentosComandosVoz\.venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
imagem = cv2.imread("mattDamon.jpg")

imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

faces = detector_facial.detectMultiScale(imagem_cinza)
#print(faces)

for x,y,l,a  in faces:
    imagem_cinza = cv2.rectangle(imagem_cinza,(x,y),(x+l,y+a),(0,0,255),2)

cv2.imshow("Imagem", imagem_cinza)
cv2.waitKey()