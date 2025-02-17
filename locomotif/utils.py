import pathlib
import shutil
import os
import IPython

def clear_cache_python():
    _ = [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]
    
    # print('Python cache cleared.')

def clear_cache_numba():
    path_parent = IPython.paths.get_ipython_cache_dir()
    path_child = os.path.join(path_parent, 'numba_cache')

    if path_parent:
        if os.path.isdir(path_child):
            shutil.rmtree(path_child)
            
    # print('numba cache cleared.')