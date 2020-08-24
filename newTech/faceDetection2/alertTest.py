import random
import time
x = 0
while x <= 20:
       a = random.randint(1, 100)
       print (a)
       if a <= 10:
           ctypes.windll.user32.MessageBoxW(0, "The number generated is less than 10", "ALERT", 1)
       x += 1
       time.sleep(.5)