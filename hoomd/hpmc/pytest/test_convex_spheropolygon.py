import numpy as np
from hoomd import *


import numpy as np
from hoomd import *
import pytest


def test_convex_spheropolygon_python():

    verts = [(-1, 1), (1, -1), (1, 1), (-1, -1)]
    poly = hpmc.integrate.ConvexSpheropolygon(23456)
    poly.shape['A'] = dict(vertices=verts, sweep_radius=0)
    assert not poly.shape['A']['ignore_statistics']
    assert poly.shape['A']['sweep_radius'] == 0
    np.testing.assert_allclose(poly.shape['A']['vertices'], verts)

def test_convex_spheropolygon_params():
    
    verts1 = [(-1, 1), (1, -1), (1, 1), (-1, -1)]
    verts2 = [(-1, 1), (1, -1), (1, 0)]
    verts3 = [(-1, 1), (-1, -1), (0, 1), (0, 0)]
    verts4 = [(0, 1), (1, 0), (1, 0), (0, -1), (-1, 0)]
    verts5 = [(-1, 1), (-1, -1), (1, 1), (0, 0), (1, 0), (0, 1)]
    
    poly = hpmc.integrate.ConvexSpheropolygon(23456)
    
    poly.shape['A'] = dict()
    assert poly.shape['A']['vertices'] is None
    assert poly.shape['A']['sweep_radius'] is None
    assert poly.shape['A']['ignore_statistics'] is False

    poly.shape['B'] = dict(vertices=verts1, sweep_radius=1, ignore_statistics=True)
    assert poly.shape['B']['vertices'] == verts1
    assert poly.shape['B']['sweep_radius'] == 1
    assert poly.shape['B']['ignore_statistics'] is True
    
    
    poly.shape['C'] = dict(vertices=verts1, sweep_radius=0)
    poly.shape['D'] = dict(vertices=verts2, sweep_radius=1, ignore_statistics=True)
    poly.shape['E'] = dict(vertices=verts3, sweep_radius=2)
    poly.shape['F'] = dict(vertices=verts4, sweep_radius=3, ignore_statistics=True)
    poly.shape['G'] = dict(vertices=verts5, sweep_radius=4)
    '''
    np.testing.assert_allclose(poly.shape['C']['vertices'], verts1)
    np.testing.assert_allclose(poly.shape['D']['vertices'], verts2)
    np.testing.assert_allclose(poly.shape['E']['vertices'], verts3)
    np.testing.assert_allclose(poly.shape['F']['vertices'], verts4)
    np.testing.assert_allclose(poly.shape['G']['vertices'], verts5)
    '''
    assert poly.shape['C']['vertices'] == verts1
    assert poly.shape['D']['vertices'] == verts2
    assert poly.shape['E']['vertices'] == verts3
    assert poly.shape['F']['vertices'] == verts4
    assert poly.shape['G']['vertices'] == verts5

def test_convex_poly_after_attaching(device, dummy_simulation_factory):

    verts1 = [(-1, 1), (1, -1), (1, 1), (-1, -1)]
    verts2 = [(-1, 1), (1, -1), (1, 0)]
    verts3 = [(-1, 1), (-1, -1), (0, 1), (0, 0)]
    verts4 = [(0, 1), (1, 0), (1, 0), (0, -1), (-1, 0)]
    verts5 = [(-1, 1), (-1, -1), (1, 1), (0, 0), (1, 0), (0, 1)]
    
    poly = hpmc.integrate.ConvexSpheropolygon(23456)
    poly.shape['A'] = dict(vertices=verts1, sweep_radius=0)
    poly.shape['B'] = dict(vertices=verts2, sweep_radius=1, ignore_statistics=True)
    poly.shape['C'] = dict(vertices=verts3, sweep_radius=2)
    poly.shape['D'] = dict(vertices=verts4, sweep_radius=3, ignore_statistics=True)
    poly.shape['E'] = dict(vertices=verts5, sweep_radius=4)


    sim = dummy_simulation_factory(particle_types=['A', 'B', 'C', 'D', 'E'])
    sim.operations.add(poly)
    sim.operations.schedule()

    assert not poly.shape['A']['ignore_statistics']
    assert poly.shape['B']['ignore_statistics']
    assert not poly.shape['C']['ignore_statistics']
    assert poly.shape['D']['ignore_statistics']
    assert not poly.shape['E']['ignore_statistics']
    
    np.testing.assert_allclose(poly.shape['A']['vertices'], verts1)
    np.testing.assert_allclose(poly.shape['B']['vertices'], verts2)
    np.testing.assert_allclose(poly.shape['C']['vertices'], verts3)
    np.testing.assert_allclose(poly.shape['D']['vertices'], verts4)
    np.testing.assert_allclose(poly.shape['E']['vertices'], verts5)
    

    # check for errors on invalid input
    with pytest.raises(RuntimeError):
        poly.shape['A'] = dict(vertices='invalid', sweep_radius=0)
        
    with pytest.raises(TypeError):
        poly.shape['A'] = dict(vertices=[1, 2, 3, 4], sweep_radius=0)
        
    with pytest.raises(RuntimeError):
        poly.shape['A'] = dict(vertices=verts1, sweep_radius=0, ignore_statistics='invalid')
        
    with pytest.raises(RuntimeError):
        poly.shape['A'] = dict(vertices=verts1, sweep_radius='invalid')
        
    with pytest.raises(RuntimeError):
        poly.shape['A'] = dict(vertices=verts1, sweep_radius=[1, 2, 3, 4])