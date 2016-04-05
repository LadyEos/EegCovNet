import datetime, threading, time
import serial


ser = serial.Serial(11)  # open first serial port
print ser.name          # check which port was really used
 
counter = 0
flag = False

def foo():
	next_call = time.time()
	global counter, flag, ser
	while counter < 15:
		if(counter%7 == 0):
			if(flag):
				print "PUSH"
				ser.write("1")      # write a string
				flag = False
			else:
				print "REST"
				ser.write("2")      # write a string
				flag = True
		print "."
		next_call = next_call+1
		time.sleep(next_call - time.time())
		counter+=1
	print "Good bye!"
	ser.close() 

timerThread = threading.Thread(target=foo)
timerThread.start()