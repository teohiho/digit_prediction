# python write to file by line
import time
with open('somefile.txt', 'a') as the_file:
    # the_file.write('Hello\n')
    for i in range(100):
    	the_file.write("" +str(i))
    	print(i)
    	time.sleep(2)
 #sau do dung tmux
