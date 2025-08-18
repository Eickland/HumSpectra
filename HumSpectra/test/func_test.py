from HumSpectra import utilits as ut
from HumSpectra import ultraviolet as uv

import glob
import os
import pandas as pd

test_folder = os.getcwd()
test_uv_path = test_folder + "\\HumSpectra\\test\\B.csv"

test_uv_spectra = uv.read_uv(test_uv_path)

test_e2_e3 = uv.e2_e3(test_uv_spectra)

print(test_e2_e3)