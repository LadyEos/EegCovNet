import datetime, threading, time


counter = 0

def foo():
    next_call = time.time()
    global counter
    
    while counter < 5:
        print datetime.datetime.now()
        print counter
        next_call = next_call+5
        time.sleep(next_call - time.time())
        counter+=1

timerThread = threading.Thread(target=foo)
timerThread.start()
