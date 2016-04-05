import serial
ser = serial.Serial(11)  # open first serial port
print ser.name          # check which port was really used
ser.write("1")      # write a string
ser.close()             # close port