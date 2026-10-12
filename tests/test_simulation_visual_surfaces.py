import numpy as np
from xml.etree import ElementTree as ET

from dronedream_agent_core.optical_map import _obj_triangles
from dronedream_agent_core.simulation_visual_surfaces import normalized_obj, normalize_runtime_visuals
from test_optical_map import cube


def test_closed_inward_shell_has_outward_render_normals_and_same_geometry():
    raw = '\n'.join('f ' + ' '.join(line.split()[1:][::-1]) if line.startswith('f ') else line
                    for line in cube().decode().splitlines()).encode()
    normalized = normalized_obj(raw)
    actual = _obj_triangles(normalized, [1, 1, 1])
    expected = _obj_triangles(raw, [1, 1, 1])
    np.testing.assert_allclose(actual, expected)
    assert np.einsum('ij,ij->i', actual[:, 0], np.cross(actual[:, 1], actual[:, 2])).sum() > 0
    assert b'vn ' in normalized


def test_open_uv_surface_keeps_winding_and_negative_references():
    raw = b'v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 1\nf -4/-4/-1 -3/-3/-1 -2/-2/-1 -1/-1/-1\n'
    normalized = normalized_obj(raw)
    np.testing.assert_allclose(_obj_triangles(normalized, [1,1,1]), _obj_triangles(raw, [1,1,1]))
    assert b'/-' not in normalized
    assert len([line for line in normalized.splitlines() if line.startswith(b'f ')]) == 2


def test_material_defaults_only_fill_missing_values_in_run_copy(tmp_path):
    mesh = tmp_path/'mesh.obj'
    mesh.write_bytes(cube())
    world = ET.fromstring('''<world><model><link><visual><geometry><mesh><uri>mesh.obj</uri></mesh></geometry>
      <material><diffuse>0.2 0.3 0.4 1</diffuse><pbr><metal><albedo_map>ground.png</albedo_map></metal></pbr></material>
      </visual><collision><geometry><mesh><uri>collision.obj</uri></mesh></geometry></collision></link></model></world>''')
    collision = tmp_path/'collision.obj'; collision.write_bytes(cube())
    receipt = normalize_runtime_visuals(world, tmp_path)
    assert set(receipt) == {'mesh.obj'}
    assert collision.read_bytes() == cube()
    assert world.findtext('.//material/diffuse') == '0.2 0.3 0.4 1'
    assert world.findtext('.//material/pbr/metal/metalness') == '0'
    assert world.findtext('.//material/ambient') == '1 1 1 1'
