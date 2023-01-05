import cv2
import dlib
import imutils
from imutils import face_utils


detector = dlib.get_frontal_face_detector()
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear


def compute_ear(frame):
    frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale frame
	rects = detector(gray, 0)

    for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0
    
    return ear


def read_video(fname):
    # this is the video capture object
    video = cv2.VideoCapture(f"videos/{fname}.avi")

    # this reads the first frame
    success, image = video.read()
    count = 0
    success = True

    # so long as vidcap can read the current frame... 
    frame_l = []
    while success:

        # ...read the next frame (this is now your current frame)
        success, image = video.read()
        count += 1 # moved this to be accurate with my 'second frame' statement
        frame_l.append(image)
    
    ear_min = compute_ear(frame_l[-1])
    ear_max = compute_ear(frame_l[0])

    ear_l = [ear_min + (ear_max - ear_min) / 5 * 4 for i in range(0, 6)]

    for i in range(len(frame_l)):
        ear = compute_ear(frame_l[i])
        if ear <= ear_min + (ear_max - ear_min) / 5:
            label = 0
        elif ear <= ear_min + (ear_max - ear_min) / 5 * 2:
             label = 1
        elif ear <= ear_min + (ear_max - ear_min) / 5 * 3:
            label = 2
        elif ear <= ear_min + (ear_max - ear_min) / 5 * 4:
            label = 3
        else:
            label = 4
        # this is where you put your functionality (this just saves the frame)
        cv2.imwrite(f"../dataset/clf/eyeState{label}/{fname}_{i:04}.jpg", image)     # save frame as JPEG file


if __name__ == '__main__':
    files = list(map(lambda x: {'file': x}, glob.glob('videos/*.avi')))
    for file in files:
        fname = file['file'].split('.')[0].split('/')[1]
        read_video(fname)