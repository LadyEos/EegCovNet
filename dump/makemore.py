import os
import shutil


basepath_trial = 'data2/ICA2/trial/'
basepath_event = 'data2/ICA2/event/'

for root, dirs, files in os.walk(basepath_event):
	for file in files:
		name, ext = file.split('.')
		for c in range(2,5):
			shutil.copyfile(basepath_event+file, basepath_event+name+'-'+str(c)+'.'+ext)

for root, dirs, files in os.walk(basepath_trial):
	for file in files:
		name, ext = file.split('.')
		for c in range(2,5):
			shutil.copyfile(basepath_trial+file, basepath_trial+name+'-'+str(c)+'.'+ext)