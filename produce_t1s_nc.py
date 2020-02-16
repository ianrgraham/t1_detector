import sys
import os
import numpy as np
import detect as dt
import glob
import argparse

parser = argparse.ArgumentParser(description="Produce birth death data from hoomd trajectories.")
parser.add_argument('-d','--dir',help="Directory of the trajectory.", type=str, default="")
cmdargs = parser.parse_args()

dt.t1_producer_nc(cmdargs.dir)