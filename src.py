import cv2 as cv


#If the picture is in the same folder, you only need to write the name of the picture, for example : XXXYYY.jpg
#but if the picture is in another drive or folder, you have to write the full path, for example : Downloads/Pictures/XXXYYY.jpg
original_image = cv.imread("test.jpg") 

# We use this line of code to convert the original image to grayscale so our algorithm can process it
grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

# We load the classifier
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_alt.xml")

# This line will detect the faces
detected_faces = face_cascade.detectMultiScale(grayscale_image, 1.3)

# detected_faces is an Ndarray showing the coordinates of each face
# now we just need to draw squares around those coordinates to show the detected faces on the screen
for (column, row, width, height) in detected_faces:
    cv.rectangle(
        original_image,
        (column, row),
        (column + width, row + height),
        (0, 0, 255),  #color of the squares in RGB
        2             #Thickness of squares lines
    )

cv.imshow('Image', original_image)  #show the final result
cv.waitKey(0)                  
cv.destroyAllWindows()      