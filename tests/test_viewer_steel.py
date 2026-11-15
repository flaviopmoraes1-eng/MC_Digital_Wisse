import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import viewer


class SteelViewerTests(unittest.TestCase):
    def test_steel_is_disabled_for_result_fields(self):
        mesh = SimpleNamespace(
            nodes=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
            triangles=np.array([[0, 1, 2]]),
            tri_surface=np.array([1]),
            bbox_diagonal=2.0,
            curves={},
        )
        captured = []

        def component(**kwargs):
            captured.append(kwargs["data"])

        with patch.object(viewer, "_component", return_value=component):
            args = (mesh, "mesh", {"curves": [], "surfaces": [], "node": None}, {}, {}, False)
            viewer.mesh_viewer(*args, brushed_steel=True)
            viewer.mesh_viewer(*args, brushed_steel=True, field=np.array([1., 2., 3.]))
            viewer.mesh_viewer(*args)

        self.assertTrue(captured[0]["brushed_steel"])
        self.assertFalse(captured[1]["brushed_steel"])
        self.assertEqual(captured[1]["field"], [1., 2., 3.])
        self.assertFalse(captured[2]["brushed_steel"])
        self.assertEqual(captured[0]["nodes"], captured[1]["nodes"])
        self.assertEqual(captured[0]["tris"], captured[1]["tris"])


if __name__ == "__main__":
    unittest.main()
