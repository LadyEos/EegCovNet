from datetime import datetime, time
#import serial

nowAll = datetime.now().second
passedTime = datetime.now().second
seconds = passedTime-nowAll


while(passedTime-nowAll < 10):
	print(nowAll)
	print(passedTime)
	print(seconds)
	print('not yet')
	passedTime = datetime.now().second
	seconds = passedTime-nowAll

print('finished')