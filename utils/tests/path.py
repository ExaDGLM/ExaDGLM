import sys
from os.path import abspath, dirname, join

cwd = dirname(abspath(__file__))
proj_dir = dirname(cwd)
mod_dir = join(proj_dir, 'src', 'python')
sys.path.append(mod_dir)