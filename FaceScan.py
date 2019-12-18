import face_recognition
import cv2
from PIL import Image
import pytesseract
import os

def picList(path):
    dirList=os.listdir(path)
    customerList=[]
    for file in dirList:
        if(file.__contains__('jpg')):
            customerList.append(file)
    return customerList

def textExtraction(path):
    #path = './CHRPB7508J.jpg'
    img = Image.open(path)
    img = img.convert('RGBA')
    pix = img.load()
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if pix[x, y][0] < 102 or pix[x, y][1] < 102 or pix[x, y][2] < 102:
                pix[x, y] = (0, 0, 0, 255)
            else:
                pix[x, y] = (255, 255, 255, 255)
    img.save('temp.jpg')
    image = cv2.imread("./temp.jpg")
    image = cv2.resize(image,(0,0),fx=7,fy=7)
    image = cv2.GaussianBlur(image,(11,11),13)
    image = cv2.medianBlur(image,9)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename,gray)

    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    os.remove("temp.jpg")
    return text

def facRecog(path):
    video_capture = cv2.VideoCapture(0)
    pictureList=picList(path)
    imageList=[]
    knownEncodingList=[]
    knownFaceList=[]
    count=0
    count2=0
    output=''
    for pic in pictureList:
        knownFaceList.append(pic.strip('.jpg'))
        imageList.append(face_recognition.load_image_file(pic))
        knownEncodingList.append(face_recognition.face_encodings(imageList[count])[0])
        count=count+1

    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(knownEncodingList, face_encoding)
                name = "Unknown"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = knownFaceList[first_match_index]
                face_names.append(name)

        process_this_frame = not process_this_frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            output=name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.imshow('Video', frame)

        if output!='Unknown' and output!='':
            break
        elif count2>1000:
            break
        count2=count2+1
    video_capture.release()
    cv2.destroyAllWindows()
    return output

if __name__=='__main__':
    path='./CHRPB7508J.jpg'
    pic=facRecog("./")
    print(pic)
    text=textExtraction('./'+pic+'.jpg')
    if text.__contains__(pic):
        print('KYC Validation successfull.')
    else:
        print('KYC Validation failed.')

