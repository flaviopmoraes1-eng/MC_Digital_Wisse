import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mesher import mesh_step_shell
import viewer
from water_section import section_contours


def prism(points, z0=0., z1=2.):
    points = np.asarray(points, dtype=float)
    n = len(points)
    nodes = np.array([[x, y, z] for z in (z0, z1) for x, y in points])
    triangles = []
    for i in range(n):
        j = (i + 1) % n
        triangles.extend(((i, j, n + j), (i, n + j, n + i)))
    return nodes, np.array(triangles)


def polygon_area(polygon):
    p = np.array(polygon)
    return abs(np.dot(p[:, 0], np.roll(p[:, 1], -1))
               - np.dot(p[:, 1], np.roll(p[:, 0], -1))) / 2


class WaterSectionTests(unittest.TestCase):
    def test_section_follows_irregular_contour_at_height_and_boundary(self):
        nodes, tri = prism([(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)])
        for z in (0., 1.):
            contours = section_contours(nodes, tri, z)
            self.assertEqual(len(contours), 1)
            self.assertAlmostEqual(polygon_area(contours[0]), 7.)
            self.assertTrue(all(not (x > 1 and y > 1) for x, y in contours[0]))

    def test_open_or_outside_section_is_rejected(self):
        nodes, tri = prism([(0, 0), (2, 0), (2, 2), (0, 2)])
        with self.assertRaisesRegex(ValueError, "não é fechado"):
            section_contours(nodes, tri[:-2], 1.)
        for level in (-1., 2., 3.):
            with self.assertRaisesRegex(ValueError, "fora"):
                section_contours(nodes, tri, level)

    def test_multiple_contours_include_hole(self):
        outer, t1 = prism([(0, 0), (4, 0), (4, 4), (0, 4)])
        inner, t2 = prism([(1, 1), (3, 1), (3, 3), (1, 3)])
        contours = section_contours(np.vstack((outer, inner)), np.vstack((t1, t2 + len(outer))), 1.)
        self.assertEqual(sorted(polygon_area(p) for p in contours), [4., 16.])

    def test_viewer_sends_actual_section_and_not_bounding_rectangle(self):
        nodes, tri = prism([(0, 0), (2, 0), (2, 1), (0, 1)])
        mesh = SimpleNamespace(nodes=nodes, triangles=tri, tri_surface=np.ones(len(tri), dtype=int),
                               bbox_diagonal=3., curves={})
        contours = section_contours(nodes, tri, 1.)
        with patch.object(viewer, "_component") as component:
            viewer.mesh_viewer(mesh, "mesh", {}, {}, {}, False,
                               water_level=1., water_contours=contours)
        water = component.return_value.call_args.kwargs["data"]["water_level"]
        self.assertEqual(water, {"z": 1., "contours": contours})

    def test_real_tank_section(self):
        path = Path(__file__).resolve().parent.parent / "teste_tanque3_costado_m.stl"
        mesh = mesh_step_shell(path.read_bytes(), suffix=".stl")
        level = float(mesh.nodes[:, 2].min() + 1.)
        contours = section_contours(mesh.nodes, mesh.triangles, level)
        self.assertEqual(len(contours), 1)
        self.assertGreater(len(contours[0]), 100)
        self.assertGreater(polygon_area(contours[0]), 1.)


if __name__ == "__main__":
    unittest.main()
