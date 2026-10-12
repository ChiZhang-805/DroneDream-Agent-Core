"""Render polygonal OBJ surfaces consistently without altering collision geometry."""

import hashlib
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from .optical_map import _outward_closed_faces, _triangulate_face


def normalized_obj(raw):
    vertices, normals, entries, faces, tokens = [], [], [], [], []
    texture_count = 0
    for line in raw.decode('utf-8').splitlines():
        fields = line.partition('#')[0].split()
        if fields and fields[0] == 'v':
            vertices.append([float(value) for value in fields[1:4]])
        if fields and fields[0] == 'vn':
            normals.append([float(value) for value in fields[1:4]])
        if fields and fields[0] == 'vt':
            texture_count += 1
        if fields and fields[0] == 'f':
            ids = [int(token.split('/')[0]) for token in fields[1:]]
            ids = [value - 1 if value > 0 else len(vertices) + value for value in ids]
            references = []
            for index, token in zip(ids, fields[1:]):
                parts = token.split('/')
                parts[0] = str(index + 1)
                for slot, count in ((1, texture_count), (2, len(normals))):
                    if len(parts) > slot and parts[slot] and int(parts[slot]) < 0:
                        parts[slot] = str(count + int(parts[slot]) + 1)
                references.append('/'.join(parts))
            lookup = dict(zip(ids, references))
            for triangle in _triangulate_face(vertices, ids):
                entries.append(len(faces))
                faces.append(triangle)
                tokens.append(lookup)
        else:
            entries.append(line)
    oriented = _outward_closed_faces(vertices, faces)
    lines = []
    normal_count = len(normals)
    # Existing normal indices can appear anywhere in the source. Emit them first
    # so generated normals never reuse an index from a later declaration.
    lines.extend('vn ' + ' '.join(map(str, normal)) for normal in normals)
    for entry in entries:
        if isinstance(entry, str):
            if not entry.lstrip().startswith('vn '):
                lines.append(entry)
            continue
        triangle = oriented[entry]
        source = [tokens[entry][index].split('/') for index in triangle]
        flipped = triangle != faces[entry]
        if not flipped and all(len(parts) == 3 and parts[2] for parts in source):
            lines.append('f ' + ' '.join('/'.join(parts) for parts in source))
            continue
        a, b, c = np.asarray(vertices)[triangle]
        normal = np.cross(b-a, c-a)
        normal /= np.linalg.norm(normal)
        normal_count += 1
        lines.append('vn ' + ' '.join(map(str, normal)))
        lines.append('f ' + ' '.join(
            f'{index+1}/{parts[1] if len(parts)>1 else ""}/{normal_count}'
            for index, parts in zip(triangle, source)))
    return ('\n'.join(lines) + '\n').encode('utf-8')


def normalize_runtime_visuals(world, output):
    """Normalize only run-owned OBJ copies and missing textured-material defaults."""
    output = Path(output).resolve()
    receipts = {}
    for visual in world.findall('.//visual'):
        uri = visual.findtext('geometry/mesh/uri', '')
        path = (output / uri).resolve()
        if uri and path.is_relative_to(output) and path.suffix.lower() == '.obj' and path.is_file():
            relative = path.relative_to(output).as_posix()
            if relative not in receipts:
                source = path.read_bytes()
                normalized = normalized_obj(source)
                path.write_bytes(normalized)
                receipts[relative] = {'source_sha256': hashlib.sha256(source).hexdigest(),
                                     'render_sha256': hashlib.sha256(normalized).hexdigest()}
        material = visual.find('material')
        if material is not None and material.find('pbr/metal/albedo_map') is not None:
            # SDF defaults a missing diffuse colour to black. Explicitly authored
            # colours/metalness remain untouched.
            for tag, value in [('ambient', '1 1 1 1'), ('diffuse', '1 1 1 1')]:
                if material.find(tag) is None:
                    ET.SubElement(material, tag).text = value
            metal = material.find('pbr/metal')
            if metal.find('metalness') is None:
                ET.SubElement(metal, 'metalness').text = '0'
    return receipts
