import datetime, threading, time
import serial
import random


ser = serial.Serial(11)  # open first serial port
print ser.name          # check which port was really used
 
counter = 0
end = 0

def foo():
	next_call = time.time()
	global counter, ser, end
	start = random.randint(0,8)
	duration = random.randint(2,8)
	print "START!"
	while counter < 13:
		if(start != 0):
			if(counter == start):
				print "START"
				ser.write("1")		#49
			elif(counter == start + duration):
				print "REST"
				ser.write("2")      # 50
		print "."
		next_call = next_call+1
		time.sleep(next_call - time.time())
		counter+=1
	print "Good bye!"
	ser.close() 

timerThread = threading.Thread(target=foo)
timerThread.start()