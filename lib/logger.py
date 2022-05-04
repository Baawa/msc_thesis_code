import os
from datetime import datetime

class Logger:
  def __init__(self, output_dir):
    self.output_file = output_dir + "/log.txt"
    try:
      os.mkdir(output_dir)
    except:
      print("output_dir exists")

  def log(self, str):
    f = open(self.output_file, "a")
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    f.write("{}\t{}\n".format(now, str))
    f.close()