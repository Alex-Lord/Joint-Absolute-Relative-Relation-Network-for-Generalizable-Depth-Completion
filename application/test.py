import subprocess
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# print(sys.path)
import argparse
from PIL import Image
from pathlib import Path
import setproctitle
setproctitle.setproctitle("PyThon")
from sfv2_networks import *