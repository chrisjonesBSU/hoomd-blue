import hoomd
import hoomd.hpmc
from hoomd.hpmc import _hpmc

def test_dict_conversions():
    faceted_ell_args_1 = {"normals": [(0, 0, 1), (0, 1, 0), (1, 0, 0),
                                      (0, 1, 1), (1, 1, 0), (1, 0, 1)],
                          "offsets": [1, 3, 2, 6, 3, 1],
                          "a": 3, "b": 4, "c": 1,
                          "vertices": [(0, 0, 0), (0, 0, 1), (0, 1, 0),
                                       (1, 0, 0), (1, 1, 1), (1, 1, 0)],
                          "origin": (0, 0, 0), "ignore_statistics":1}
    faceted_ell_args_2 = {"normals": [(0, 0, 0), (2, 1, 1), (1, 3, 3),
                                      (5, 1, 1), (1, 3, 0), (1, 2, 2)],
                          "offsets": [1, 3, 3, 2, 3, 1],
                          "a": 2, "b": 1, "c": 3,
                          "vertices": [(1, 0, 0), (1, 1, 0), (1, 2, 1),
                                       (0, 1, 1), (1, 1, 2), (0, 0, 1)],
                          "origin": (0, 0, 1), "ignore_statistics":0}
    faceted_ell_args_3 = {"normals": [(0, 0, 2), (0, 1, 1), (1, 3, 5), (0, 1, 6)], "offsets": [6, 2, 2, 5], "a": 1, "b": 6, "c": 6, "vertices": [(0, 0, 0), (1, 1, 1), (1, 0, 2), (2, 1, 1)], "origin": (0, 1, 0), "ignore_statistics":1}
    faceted_ell_args_4 = {"normals": [(0, 0, 2), (2, 2, 0), (3, 1, 1), (4, 1, 1), (1, 2, 0), (3, 3, 1), (1, 2, 1), (3, 3, 2)], "offsets": [5, 3, 3, 4, 3, 4, 2, 2], "a": 2, "b": 2, "c": 4, "vertices": [(0, 1, 0), (1, 1, 1), (1, 0, 1), (0, 1, 1), (1, 1, 0), (0, 0, 1), (0, 0, 1), (0, 0, 1)], "origin": (1, 0, 0), "ignore_statistics":0}
    faceted_ell_args_5 = {"normals": [(0, 0, 1), (0, 4, 0), (2, 0, 1), (0, 3, 1), (4, 1, 0), (2, 2, 1), (1, 3, 1), (1, 9, 0), (2, 2, 2)], "offsets": [5, 4, 2, 2, 7, 3, 1, 4, 1], "a": 6, "b": 1, "c": 1, "vertices": [(0, 10, 3), (3, 2, 1), (1, 2, 1), (0, 1, 1), (1, 1, 0), (5, 0, 1), (0, 10, 1), (9, 5, 1), (0, 0, 1)], "origin": (0, 0, 0), "ignore_statistics":1}

    faceted_ell_test_args1 = {'shapes': [faceted_ell_args_1, faceted_ell_args_2], 'positions': [(0, 0, 0), (0, 0, 1)], 'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)], 'overlap': [1, 1], 'capacity': 4, 'ignore_statistics': 1}
    faceted_ell_test_args2 = {'shapes': [faceted_ell_args_3, faceted_ell_args_2], 'positions': [(1, 0, 0), (0, 0, 1)], 'orientations': [(1, 1, 0, 0), (1, 0, 0, 0)], 'overlap': [1, 2], 'capacity': 3, 'ignore_statistics': 0}
    faceted_ell_test_args3 = {'shapes': [faceted_ell_args_4, faceted_ell_args_2], 'positions': [(1, 1, 0), (0, 0, 1)], 'orientations': [(1, 0, 1, 0), (1, 0, 0, 0)], 'overlap': [1, 3], 'capacity': 2, 'ignore_statistics': 1}
    faceted_ell_test_args4 = {'shapes': [faceted_ell_args_5, faceted_ell_args_2], 'positions': [(1, 1, 1), (0, 0, 0)], 'orientations': [(1, 0, 0, 1), (1, 0, 0, 0)], 'overlap': [1, 0], 'capacity': 1, 'ignore_statistics': 0}
    faceted_ell_test_args5 = {'shapes': [faceted_ell_args_1, faceted_ell_args_3], 'positions': [(1, 0, 1), (0, 0, 0)], 'orientations': [(1, 0, 0, 0), (1, 1, 0, 0)], 'overlap': [0, 1], 'capacity': 5, 'ignore_statistics': 1}
    faceted_ell_test_args6 = {'shapes': [faceted_ell_args_1, faceted_ell_args_4], 'positions': [(1, 0, 0), (0, 1, 1)], 'orientations': [(1, 0, 0, 0), (1, 0, 1, 0)], 'overlap': [2, 1], 'capacity': 6, 'ignore_statistics': 0}
    faceted_ell_test_args7 = {'shapes': [faceted_ell_args_1, faceted_ell_args_5], 'positions': [(0, 1, 1), (0, 0, 1)], 'orientations': [(1, 0, 0, 0), (1, 0, 0, 1)], 'overlap': [3, 1], 'capacity': 4, 'ignore_statistics': 1}
    faceted_ell_test_args8 = {'shapes': [faceted_ell_args_3, faceted_ell_args_4], 'positions': [(0, 0, 0), (1, 0, 1)], 'orientations': [(1, 0, 0, 1), (1, 1, 0, 0)], 'overlap': [0, 0], 'capacity': 4, 'ignore_statistics': 0}
    faceted_ell_test_args9 = {'shapes': [faceted_ell_args_3, faceted_ell_args_5], 'positions': [(0, 0, 1), (0, 0, 0)], 'orientations': [(1, 0, 1, 0), (1, 0, 1, 0)], 'overlap': [2, 2], 'capacity': 4, 'ignore_statistics': 1}
    faceted_ell_test_args10 = {'shapes': [faceted_ell_args_4, faceted_ell_args_5], 'positions': [(0, 1, 0), (1, 0, 1)], 'orientations': [(1, 1, 0, 1), (1, 0, 0, 0)], 'overlap': [3, 3], 'capacity': 4, 'ignore_statistics': 0}

    faceted_ell_test_args11 = {'shapes': [faceted_ell_args_1, faceted_ell_args_2, faceted_ell_args_3], 'positions': [(0, 0, 0), (0, 0, 1), (1, 1, 1)], 'orientations': [(1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 1)], 'overlap': [1, 1, 1], 'capacity': 4, 'ignore_statistics': 1}
    faceted_ell_test_args12 = {'shapes': [faceted_ell_args_1, faceted_ell_args_3, faceted_ell_args_4], 'positions': [(1, 0, 0), (0, 0, 1), (1, 0, 1)], 'orientations': [(1, 1, 0, 0), (1, 0, 0, 0), (1, 0, 1, 0)], 'overlap': [1, 2, 1], 'capacity': 3, 'ignore_statistics': 0}
    faceted_ell_test_args13 = {'shapes': [faceted_ell_args_1, faceted_ell_args_4, faceted_ell_args_5], 'positions': [(1, 1, 0), (0, 0, 1), (0, 1, 1)], 'orientations': [(1, 0, 1, 0), (1, 0, 0, 0), (1, 1, 0, 0)], 'overlap': [1, 3, 0], 'capacity': 2, 'ignore_statistics': 1}
    faceted_ell_test_args14 = {'shapes': [faceted_ell_args_2, faceted_ell_args_3, faceted_ell_args_4], 'positions': [(1, 1, 1), (0, 0, 0), (0, 0, 1)], 'orientations': [(1, 0, 0, 1), (1, 0, 0, 0), (1, 0, 0, 0)], 'overlap': [1, 0, 2], 'capacity': 4, 'ignore_statistics': 0}
    faceted_ell_test_args15 = {'shapes': [faceted_ell_args_2, faceted_ell_args_4, faceted_ell_args_5], 'positions': [(0, 0, 0), (0, 1, 1), (1, 0, 1)], 'orientations': [(1, 0, 0, 0), (1, 1, 0, 0), (1, 1, 0, 0)], 'overlap': [0, 1, 1], 'capacity': 4, 'ignore_statistics': 1}
    faceted_ell_test_args16 = {'shapes': [faceted_ell_args_3, faceted_ell_args_4, faceted_ell_args_5], 'positions': [(0, 1, 0), (1, 0, 1), (0, 0, 1)], 'orientations': [(1, 0, 0, 0), (1, 0, 1, 0), (1, 0, 0, 0)], 'overlap': [2, 1, 0], 'capacity': 4, 'ignore_statistics': 0}

    faceted_ell_test_args17 = {'shapes': [faceted_ell_args_1, faceted_ell_args_2, faceted_ell_args_3, faceted_ell_args_4], 'positions': [(0, 0, 0), (0, 0, 1), (1, 1, 1), (1, 1, 0)], 'orientations': [(1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0)], 'overlap': [1, 1, 1, 1], 'capacity': 4, 'ignore_statistics': 1}
    faceted_ell_test_args18 = {'shapes': [faceted_ell_args_1, faceted_ell_args_2, faceted_ell_args_3, faceted_ell_args_5], 'positions': [(1, 0, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1)], 'orientations': [(1, 1, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 1)], 'overlap': [1, 1, 2, 0], 'capacity': 3, 'ignore_statistics': 0}
    faceted_ell_test_args19 = {'shapes': [faceted_ell_args_1, faceted_ell_args_2, faceted_ell_args_4, faceted_ell_args_5], 'positions': [(1, 1, 0), (1, 0, 1), (0, 0, 0), (1, 1, 1)], 'orientations': [(1, 0, 1, 0), (1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 1, 0)], 'overlap': [1, 2, 1, 1], 'capacity': 2, 'ignore_statistics': 1}
    faceted_ell_test_args20 = {'shapes': [faceted_ell_args_1, faceted_ell_args_3, faceted_ell_args_4, faceted_ell_args_5], 'positions': [(1, 1, 1), (0, 0, 1), (0, 0, 0), (1, 1, 0)], 'orientations': [(1, 0, 0, 1), (1, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0)], 'overlap': [0, 1, 1, 0], 'capacity': 4, 'ignore_statistics': 0}
    faceted_ell_test_args21 = {'shapes': [faceted_ell_args_2, faceted_ell_args_3, faceted_ell_args_4, faceted_ell_args_5], 'positions': [(0, 0, 0), (0, 1, 1), (1, 1, 1), (1, 0, 0)], 'orientations': [(1, 0, 0, 0), (1, 1, 0, 0), (1, 0, 0, 1), (1, 0, 0, 0)], 'overlap': [1, 2, 2, 2], 'capacity': 4, 'ignore_statistics': 1}

    faceted_ell_test_args22 = {'shapes': [faceted_ell_args_1, faceted_ell_args_2, faceted_ell_args_3, faceted_ell_args_4, faceted_ell_args_5], 'positions': [(0, 0, 0), (0, 0, 1), (1, 1, 1), (1, 1, 0), (2, 2, 0)], 'orientations': [(1, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (0, 0, 1, 1), (1, 0, 0, 0)], 'overlap': [1, 1, 0, 2, 1], 'capacity': 4, 'ignore_statistics': 1}

    test_faceted_ell_union1 = _hpmc.mfellipsoid_params(faceted_ell_test_args1)
    test_faceted_ell_dict1 = test_faceted_ell_union1.asDict()
    test_faceted_ell_union2 = _hpmc.mfellipsoid_params(faceted_ell_test_args2)
    test_faceted_ell_dict2 = test_faceted_ell_union2.asDict()
    test_faceted_ell_union3 = _hpmc.mfellipsoid_params(faceted_ell_test_args3)
    test_faceted_ell_dict3 = test_faceted_ell_union3.asDict()
    test_faceted_ell_union4 = _hpmc.mfellipsoid_params(faceted_ell_test_args4)
    test_faceted_ell_dict4 = test_faceted_ell_union4.asDict()
    test_faceted_ell_union5 = _hpmc.mfellipsoid_params(faceted_ell_test_args5)
    test_faceted_ell_dict5 = test_faceted_ell_union5.asDict()
    test_faceted_ell_union6 = _hpmc.mfellipsoid_params(faceted_ell_test_args6)
    test_faceted_ell_dict6 = test_faceted_ell_union6.asDict()
    test_faceted_ell_union7 = _hpmc.mfellipsoid_params(faceted_ell_test_args7)
    test_faceted_ell_dict7 = test_faceted_ell_union7.asDict()
    test_faceted_ell_union8 = _hpmc.mfellipsoid_params(faceted_ell_test_args8)
    test_faceted_ell_dict8 = test_faceted_ell_union8.asDict()
    test_faceted_ell_union9 = _hpmc.mfellipsoid_params(faceted_ell_test_args9)
    test_faceted_ell_dict9 = test_faceted_ell_union9.asDict()
    test_faceted_ell_union10 = _hpmc.mfellipsoid_params(faceted_ell_test_args10)
    test_faceted_ell_dict10 = test_faceted_ell_union10.asDict()
    test_faceted_ell_union11 = _hpmc.mfellipsoid_params(faceted_ell_test_args11)
    test_faceted_ell_dict11 = test_faceted_ell_union11.asDict()
    test_faceted_ell_union12 = _hpmc.mfellipsoid_params(faceted_ell_test_args12)
    test_faceted_ell_dict12 = test_faceted_ell_union12.asDict()
    test_faceted_ell_union13 = _hpmc.mfellipsoid_params(faceted_ell_test_args13)
    test_faceted_ell_dict13 = test_faceted_ell_union13.asDict()
    test_faceted_ell_union14 = _hpmc.mfellipsoid_params(faceted_ell_test_args14)
    test_faceted_ell_dict14 = test_faceted_ell_union14.asDict()
    test_faceted_ell_union15 = _hpmc.mfellipsoid_params(faceted_ell_test_args15)
    test_faceted_ell_dict15 = test_faceted_ell_union15.asDict()
    test_faceted_ell_union16 = _hpmc.mfellipsoid_params(faceted_ell_test_args16)
    test_faceted_ell_dict16 = test_faceted_ell_union16.asDict()
    test_faceted_ell_union17 = _hpmc.mfellipsoid_params(faceted_ell_test_args17)
    test_faceted_ell_dict17 = test_faceted_ell_union17.asDict()
    test_faceted_ell_union18 = _hpmc.mfellipsoid_params(faceted_ell_test_args18)
    test_faceted_ell_dict18 = test_faceted_ell_union18.asDict()
    test_faceted_ell_union19 = _hpmc.mfellipsoid_params(faceted_ell_test_args19)
    test_faceted_ell_dict19 = test_faceted_ell_union19.asDict()
    test_faceted_ell_union20 = _hpmc.mfellipsoid_params(faceted_ell_test_args20)
    test_faceted_ell_dict20 = test_faceted_ell_union20.asDict()
    test_faceted_ell_union21 = _hpmc.mfellipsoid_params(faceted_ell_test_args21)
    test_faceted_ell_dict21 = test_faceted_ell_union21.asDict()
    test_faceted_ell_union22 = _hpmc.mfellipsoid_params(faceted_ell_test_args22)
    test_faceted_ell_dict22 = test_faceted_ell_union22.asDict()

    assert test_faceted_ell_dict1 == faceted_ell_test_args1
    assert test_faceted_ell_dict2 == faceted_ell_test_args2
    assert test_faceted_ell_dict3 == faceted_ell_test_args3
    assert test_faceted_ell_dict4 == faceted_ell_test_args4
    assert test_faceted_ell_dict5 == faceted_ell_test_args5
    assert test_faceted_ell_dict6 == faceted_ell_test_args6
    assert test_faceted_ell_dict7 == faceted_ell_test_args7
    assert test_faceted_ell_dict8 == faceted_ell_test_args8
    assert test_faceted_ell_dict9 == faceted_ell_test_args9
    assert test_faceted_ell_dict10 == faceted_ell_test_args10
    assert test_faceted_ell_dict11 == faceted_ell_test_args11
    assert test_faceted_ell_dict12 == faceted_ell_test_args12
    assert test_faceted_ell_dict13 == faceted_ell_test_args13
    assert test_faceted_ell_dict14 == faceted_ell_test_args14
    assert test_faceted_ell_dict15 == faceted_ell_test_args15
    assert test_faceted_ell_dict16 == faceted_ell_test_args16
    assert test_faceted_ell_dict17 == faceted_ell_test_args17
    assert test_faceted_ell_dict18 == faceted_ell_test_args18
    assert test_faceted_ell_dict19 == faceted_ell_test_args19
    assert test_faceted_ell_dict20 == faceted_ell_test_args20
    assert test_faceted_ell_dict21 == faceted_ell_test_args21
    assert test_faceted_ell_dict22 == faceted_ell_test_args22