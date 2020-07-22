#-------------------------------------------------------------------------------
# Imports
import face_recognition
import cv2
import numpy as np
import re
from faunadb import query as q
from faunadb.objects import Ref
from faunadb.client import FaunaClient
import array
#-------------------------------------------------------------------------------
# Variables & Setup

video_capture = cv2.VideoCapture(0) # Get a reference to the webcam
client = FaunaClient(secret="fnADwLZiwWACBmx5g5lkKxf-7ERVaw3MSqpqWcWG") # Connection To Fauna
indexdata = client.query(
    q.paginate(
    q.match(q.index('allusers')))
 )
idlist=[indexdata['data']]
result = re.findall('\d+', str(idlist))
# print (result)
known_face_encodings = []
known_face_names = []

for i in range(0, len(result)):
    # print(result[i])

    currentuserid = result[i]
    userdetails = client.query(q.get(q.ref(q.collection('Users'), currentuserid)))

    details_list=[userdetails['data']]

    user_image = details_list[0].get("Image")
    user_name = details_list[0].get("Name")
#-------------------------------------------------------------------------------
# Faces
    user_face_encodings = face_recognition.face_encodings(face_recognition.load_image_file(user_image))[0]
    known_face_encodings.append(user_face_encodings)
    known_face_names.append(user_name)

#-------------------- -----------------------------------------------------------
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]


        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font,  1.0, (255, 255, 255), 1)
        if (name=="RishabTheAI"):
            print(name,'The Dund Is Real')
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# # # Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
# # # #-------------------------------------------------------------------------------
