import datetime, time

now = time.time()
future = now + 1
while time.time() < future:
    # do stuff
    print(time.time())