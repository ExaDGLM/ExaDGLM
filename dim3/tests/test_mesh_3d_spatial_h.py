import numpy as np
import importlib
from numpy.testing import assert_array_equal as assert_ae
from numpy.testing import assert_array_almost_equal as assert_aae

import path


def test_update_spatial_h():
    '''
    An example
    A sphere is inscribed in the tetrahedron 
    whose faces are x=0,y=0,z=0 and 2x+6y+3z−14=0
    - center: (7/9, 7/9, 7/9)
    - radius: 7/9

    ref: https://www.vedantu.com/question-answer/a-sphere-is-inscribed-in-the-tetrahedron-whose-class-11-maths-cbse-5fc5dac0f3f01f6fb62636e2
    '''
    # setup
    VXYZ = np.array([[0, 0, 0], [7, 0, 0], [0, 7/3, 0], [0, 0, 14/3]], 'f8')
    EtoV = np.array([[0, 1, 2, 3]], 'i4')
    h = np.zeros(1, 'f8')
    
    # diameter of the inscribed sphere
    modpy = importlib.import_module(f"mesh_3d_py")
    modpy.update_spatial_h(VXYZ, EtoV, h)
    assert_aae(h[0], 2*7/9, 15)
    
    modcpp = importlib.import_module(f"mesh_3d_cpp")
    modcpp.update_spatial_h(VXYZ, EtoV, h)
    assert_aae(h[0], 2*7/9, 15) 
