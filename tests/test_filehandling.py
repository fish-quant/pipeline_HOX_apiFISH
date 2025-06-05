import sys
import os
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from utils.file_handling import FileProcessor


def test_save_spots_distributed_files():
    """put together points from two different rna images wich are very close.
    """
    
    numpy_file_add     = Path('.').resolve() / 'testfile.npy'
    format             = 'spots_IDzyx'
    dict_data          = {}
    dict_data['one']    = np.ones((5,3),dtype=int)
    dict_data['two']   = 2*np.ones((5,3),dtype=int)
    dict_data['three'] = 3*np.ones((5,3),dtype=int)
    dict_data['four']  = 4*np.ones((5,3),dtype=int)
    im_dim = 3
    
    fp = FileProcessor()
    fp.save_spots_distributed_files(numpy_file_add, format, dict_data, im_dim)
    
    #assert np.array_equal(list_gene1_only[0], np.array([5, 3])) 

def test_load_spots_distributed_files():
    "to be ran after running the precedent function"
    
    dict_address_path = Path('.').resolve() / 'testfile.npy'
    format = 'spots_IDzyx'
    fp = FileProcessor()
    data = fp.load_spots_distributed_files(dict_address_path, format)
    
    assert np.array_equal(data['one'], np.ones((5,3),dtype=int))
    assert np.array_equal(data['four'], 4*np.ones((5,3),dtype=int))



def test_save_masks_distributed_files():
    
    dict_address_path = Path('.').resolve() / 'testfile_masks.npy'
    fp                = FileProcessor()
    dict_data         = {}
    
    masks1 = np.zeros((5,5)); masks1[0,0] = 15
    masks2 = np.zeros((5,5)); masks2[1,1] = 1
    masks3 = np.zeros((5,5)); masks3[2,2] = 1
    masks4 = np.zeros((5,5)); masks4[3,3] = 1
    masks5 = np.zeros((5,5)); masks5[4,4] = 1500

    dict_data['m1'] = masks1
    dict_data['m2'] = masks2
    dict_data['m3'] = masks3
    dict_data['m4'] = masks4
    dict_data['m5'] = masks5

    fp.save_masks_distributed_files(dict_address_path, dict_data)
    
    
def test_load_masks_distributed_files():
        
    dict_address_path = Path('.').resolve() / 'testfile_masks.npy'
    fp                = FileProcessor()
    dict_data         = {}
    
    data = fp.load_masks_distributed_files(dict_address_path)   
           
    masks1 = np.zeros((5,5)); masks1[0,0] = 15
    masks2 = np.zeros((5,5)); masks2[1,1] = 1
    masks3 = np.zeros((5,5)); masks3[2,2] = 1
    masks4 = np.zeros((5,5)); masks4[3,3] = 1
    masks5 = np.zeros((5,5)); masks5[4,4] = 1500
    
    assert np.array_equal(data['m1'], masks1)
    assert np.array_equal(data['m5'], masks5)