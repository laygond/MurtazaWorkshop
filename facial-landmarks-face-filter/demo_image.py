import cv2
import numpy as np
import dlib 

# Define Neural Networks models
face_detector = dlib.get_frontal_face_detector()
ldmk_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Helper Function
def color_pop(image, mask, color=-1):
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #single channel
    gray_image = cv2.cvtColor(gray_image,cv2.COLOR_GRAY2BGR) #three channel
    image = cv2.bitwise_and(image,mask)
    if color != -1: 
        lipstick_image = np.zeros_like(image)
        lipstick_image[:] = color #153,0,157
        lipstick_image = cv2.bitwise_and(lipstick_image,mask)
        lipstick_image = cv2.GaussianBlur(lipstick_image,(7,7),10)
        return cv2.addWeighted(gray_image,1,lipstick_image,0.4,0)
    else:# natural lips
        return cv2.addWeighted(gray_image,1,image,0.5,0)

# Read Image and create image helpers
img = cv2.imread('sample.jpg')
img = cv2.resize(img,(0,0),None,0.2,0.2) # Change proportions if needed
imgOriginal = img.copy()
mask_canvas = np.zeros_like(img)
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Detect faces, detect landmarks, and create mask
# Detect faces
faces = face_detector(imgGray) 
for face in faces:
    #x1,y1 = face.left(),face.top()
    #x2,y2 = face.right(),face.bottom()
    #imgOriginal = cv2.rectangle(imgOriginal, (x1,y1),(x2,y2),(0,255,0),2)
    
    # Save ldmks per face as list
    myPoints = []
    landmarks = ldmk_detector(imgGray,face)
    for i in range(68): # There are 68 landmarks
        x = landmarks.part(i).x 
        y = landmarks.part(i).y 
        myPoints.append([x,y]) 
        #cv2.circle(imgOriginal,(x,y),5,(50,50,255),cv2.FILLED) 
        #cv2.putText(imgOriginal,str(i),(x,y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,0,255),1)
    myPoints = np.array(myPoints)
    #print(myPoints)
    
    # Create Mask -> Eyes,Lips,Jaw,Eyebrows,Nose
    # Jaw         : (0-16)
    # Left Eye    : (36-41)
    # Left Eyebrow: (17-21)
    # Outer Lips  : (48-60)
    # Inner Lips  : (61-67)
    mask_canvas = cv2.fillPoly(mask_canvas,[myPoints[48:61]],(255,255,255))

# Apply Color Pop    
img = color_pop(img, mask_canvas)   
cv2.imshow("Original" , imgOriginal)
cv2.imshow("Color Pop", img)
cv2.waitKey(0)




