import cv2

detector_facial = cv2.CascadeClassifier(r'C:\Users\labsfiap\PycharmProjects\ReconhecimentosComandosVoz\.venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)

while cv2.waitKey(1) == -1:
    status, frame = camera.read()
    print(status)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector_facial.detectMultiScale(gray)
    # print(faces)

    for x, y, l, a in faces:
        gray = cv2.rectangle(gray, (x, y), (x + l, y + a), (0, 0, 255), 2)
    camera.release()
    cv2.destroyAllWindows()