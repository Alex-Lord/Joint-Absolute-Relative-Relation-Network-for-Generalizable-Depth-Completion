import os
import time
import datetime


while True:
    current_time = datetime.datetime.now()
    print("Current time:", current_time)
    if os.path.exists('/data/4TSSD/cby/Depth-Completion/result_g2/_rz_sb_mar/models/epoch_60.pth'):
        print('Train complete, starting evaluate.')
        os.system('/data/4TSSD/cby/Depth-Completion/application/demo.sh')
        break
    else:
        print('Waiting for training to complete')
    time.sleep(3600)  # sleep for 10 minutes
