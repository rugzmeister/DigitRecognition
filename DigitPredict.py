import cv2
import numpy as np
from keras.models import load_model
model=load_model('model.h5')
cap=cv2.VideoCapture(0)
while True:
	ret,img=cap.read()
	img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	img=np.resize(img,(28,28))
	img=np.reshape(img,(1,28,28,1))
	predictions=model.predict_classes(img)
	flag=0
	for i in range(len(predictions[0])):
		if predictions[0][i]==1:
			print i
			flag=1
			break	
	if flag==0:
		print None
	comeout=cv2.waitKey(1)
	if comeout==ord('a'):
		break
	




