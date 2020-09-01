import cv2 
import numpy as np

def draw_rect(frame):
	rows, cols, _ = frame.shape
	total_rectangle=9
	global hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

	hand_rect_one_x = np.array(
		[6 * rows / 20,  6 * rows / 20,  6 * rows / 20,
		8 * rows / 20,   8 * rows / 20,  8 * rows / 20,
		10 * rows / 20, 10 * rows / 20, 10 * rows / 20],
		dtype=np.uint32)

	hand_rect_one_y = np.array(
		[1 * cols / 20, 2 * cols / 20, 3 * cols / 20,
		1 * cols / 20,  2 * cols / 20, 3 * cols / 20,
		1 * cols / 20,  2 * cols / 20, 3 * cols / 20],
		dtype=np.uint32)

	hand_rect_two_x = hand_rect_one_x + 10
	hand_rect_two_y = hand_rect_one_y + 10

	for i in range(total_rectangle):
		cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]), (hand_rect_two_y[i], hand_rect_two_x[i]), (0, 0, 0), 1)

	return frame


vid = cv2.VideoCapture(0)
while(True):

	_,frame=vid.read()
	frame = cv2.flip(frame, 1)
	
	draw_rect(frame)

	backSub = cv2.createBackgroundSubtractorMOG2()
	fgMask = backSub.apply(frame)

	cv2.imshow('frame', frame) 
	cv2.imshow('FG Mask', fgMask)

	# the 'q' button is set as the
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# After the loop release the cap object
vid.release() 


cv2.destroyAllWindows() 