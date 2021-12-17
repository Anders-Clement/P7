from pathlib import Path
from subprocess import call
from subprocess import Popen, PIPE
import time
import signal

p = Path(r'/home/rob7/newbags/fullSysCostFunc').glob('**/*')
files = [x for x in p if x.is_file()]

call(['gnome-terminal', '--', 'roscore'])
time.sleep(5)
call(['gnome-terminal', '--', 'rosparam', 'set', '/use_sim_time', 'true'])

for file2 in files:
    print(file2.absolute())
    data_conv = Popen(['rosrun', 'tracking', 'dataConv.py'])
    time.sleep(2)
    converter = Popen(['rosbag', 'play', '{}'.format(file2.absolute()), '--clock', '-r', '0.8'])
    converter.wait()
    data_conv.send_signal(signal.SIGINT)
    data_conv.wait()
    time.sleep(2)
    call(['mv', '/home/rob7/foo.csv', '/home/rob7/09-12-csv/{}.csv'.format(file2.name)])
    time.sleep(5)


