"""Utility functions
"""
# MIT License
# 
# Copyright (c) 2019 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import datetime
from subprocess import Popen, PIPE
import pickle

def gettime():
    return datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')


def store_revision_info(src_path, output_dir, arg_string):
  
    # Get git hash
    gitproc = Popen(['git', 'rev-parse', 'HEAD'], stdout = PIPE, cwd=src_path)
    (stdout, _) = gitproc.communicate()
    git_hash = stdout.strip()
  
    # Get local changes
#    gitproc = Popen(['git', 'diff', 'HEAD'], stdout = PIPE, cwd=src_path)
#    (stdout, _) = gitproc.communicate()
#    git_diff = stdout.strip()
      
    # Store a text file in the log directory
    rev_info_filename = os.path.join(output_dir, 'revision_info.txt')
    with open(rev_info_filename, "w") as text_file:
        text_file.write('command line: %s\n--------------------\n' % arg_string)
        text_file.write('git hash: %s\n--------------------\n' % git_hash)
#        text_file.write('%s' % git_diff)


def get_learning_rate_from_file(filename, step):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                s = int(par[0])
                if par[1]=='-':
                    lr = -1
                else:
                    lr = float(par[1])
                if s <= step:
                    learning_rate = lr
                else:
                    return learning_rate
                  
def load_pickle(filename):
    with open(filename, 'rb') as f:
        arr = pickle.load(f)
    return arr
  
def save_pickle(filename, var_list):
    with open(filename, 'wb') as f:
        pickle.dump(var_list, f)


