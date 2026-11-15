import unittest
from types import SimpleNamespace

import numpy as np

import fea
import report


class HydrostaticTests(unittest.TestCase):
    def test_vertical_wall_cut_by_level_force_and_moment(self):
        nodes = np.array([[0, 0, 0], [0, 0, 2], [2, 0, 2], [2, 0, 0]], dtype=float)
        tri = np.array([[0, 1, 2], [0, 2, 3]])
        f = fea.hydrostatic_loads(nodes, tri, 1.0, 1.0, 1.0).reshape(-1, 6)
        np.testing.assert_allclose(f[:, :3].sum(axis=0), [0, 1, 0], atol=1e-14)
        np.testing.assert_allclose(np.cross(nodes, f[:, :3]).sum(axis=0),
                                   [-1 / 3, 0, 1], atol=1e-14)
        self.assertTrue(np.all(f[:, 3:] == 0))

    def test_horizontal_bottom_and_dry_surface(self):
        nodes = np.array([[0, 0, 0], [0, 2, 0], [2, 0, 0]], dtype=float)
        tri = np.array([[0, 1, 2]])
        f = fea.hydrostatic_loads(nodes, tri, 1.0, 1.0, 1.0).reshape(-1, 6)
        np.testing.assert_allclose(f[:, 2], [-2 / 3] * 3)
        np.testing.assert_array_equal(fea.hydrostatic_loads(nodes, tri, -1.0, 1.0, 1.0), 0)

    def test_partial_triangle_with_vertex_on_level(self):
        nodes = np.array([[0, 0, 1], [0, 0, 2], [2, 0, 0]], dtype=float)
        tri = np.array([[0, 1, 2]])
        f = fea.hydrostatic_loads(nodes, tri, 1, 1, 1).reshape(-1, 6)
        np.testing.assert_allclose(f[:, 1].sum(), 1 / 6)
        np.testing.assert_allclose(np.cross(nodes, f[:, :3]).sum(axis=0),
                                   [-1 / 12, 0, 5 / 24])

    def test_fully_wet_linear_pressure_consistent_nodal_forces(self):
        nodes = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
        f = fea.hydrostatic_loads(nodes, np.array([[0, 1, 2]]), 2, 1, 1).reshape(-1, 6)
        np.testing.assert_allclose(f[:, 1], [7 / 24, 6 / 24, 7 / 24])

    def test_reference_is_lowest_fully_anchored_edge(self):
        nodes = np.array([[0, 0, 5], [1, 0, 4], [0, 0, -2], [1, 0, -1]], dtype=float)
        mesh = SimpleNamespace(nodes=nodes, curves={1: np.array([[0, 1]]), 2: np.array([[2, 3]])})
        anchor = {d: 0.0 for d in ("UX", "UY", "UZ", "RX", "RY", "RZ")}
        self.assertEqual(fea.hydrostatic_reference(mesh, {1: anchor, 2: anchor}), (2, -2))
        self.assertEqual(fea.hydrostatic_reference(mesh, {1: anchor, 2: {"UZ": 0}}), (1, 4))
        with self.assertRaisesRegex(ValueError, "ancorada"):
            fea.hydrostatic_reference(mesh, {2: {"UZ": 0}})

    def test_water_units_and_invalid_inputs(self):
        nodes = np.array([[0, 0, 0], [0, 1000, 0], [1000, 0, 0]], dtype=float)
        tri = np.array([[0, 1, 2]])
        f = fea.hydrostatic_loads(nodes, tri, 1000, 1e-9, 9810).reshape(-1, 6)
        self.assertAlmostEqual(f[:, 2].sum(), -0.00981 * 500000)
        for level, rho, grav in [(0, 0, 1), (0, 1, -1), (np.nan, 1, 1)]:
            with self.assertRaises(ValueError):
                fea.hydrostatic_loads(nodes, tri, level, rho, grav)

    def test_report_lists_hydrostatic_parameters_only_when_active(self):
        nodes = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        mesh = SimpleNamespace(nodes=nodes, curves={4: np.array([[0, 1]])})
        anchor = {d: 0.0 for d in ("UX", "UY", "UZ", "RX", "RY", "RZ")}
        bc = {"curves": {4: anchor}, "edge_forces": {}, "pressure": {}, "forces": []}
        self.assertFalse(any(row[0] == "Hidrostática" for row in report._bc_rows(bc, mesh)))
        bc["hydrostatic"] = {"height": 1000, "rho": 1e-9, "g": 9810}
        row = next(row for row in report._bc_rows(bc, mesh) if row[0] == "Hidrostática")
        self.assertIn("aresta 4", row[1])
        self.assertIn("0.00981 MPa", row[2])
        self.assertIn("1000 kg/m³", row[2])


if __name__ == "__main__":
    unittest.main()
