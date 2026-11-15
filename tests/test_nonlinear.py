import unittest
from types import SimpleNamespace
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

import numpy as np
import streamlit as st
from streamlit.testing.v1 import AppTest

import fea
import nonlinear
from mesher import ShellMesh


class NonlinearTests(unittest.TestCase):
    def require_solver(self):
        try:
            return nonlinear.solver_path()
        except ValueError as exc:
            self.skipTest(str(exc))

    def setUp(self):
        self.mesh = SimpleNamespace(
            nodes=np.array([[0., 0., 0.], [0., 10., 0.], [10., 0., 0.], [10., 10., 0.]]),
            triangles=np.array([[0, 2, 3], [0, 3, 1]]),
            tri_surface=np.array([1, 1]),
            curves={1: np.array([[0, 1]])},
            base_thickness=None,
        )
        self.mat = fea.Material(210000., 0.3, 1.)
        self.bc = {
            "curves": {1: {d: 0. for d in fea.DOFS}},
            "pressure": {}, "corrosion": [], "edge_forces": {}, "cracks": [],
            "forces": [{"node": 2, "F": [0., 0., -1.]}, {"node": 3, "F": [0., 0., -1.]}],
        }

    def test_deck_has_nonlinear_step_and_fixed_loads(self):
        deck = nonlinear.input_deck(self.mesh, self.mat, self.bc, None, 20.).decode()
        self.assertIn("*STEP,NLGEOM", deck)
        self.assertIn("*ELEMENT,TYPE=S3", deck)
        self.assertIn("3,3,-1", deck)
        self.assertIn("*EL PRINT,ELSET=ALLSHELLS", deck)

    def test_unsupported_thermal_and_cracks_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "térmica"):
            nonlinear.input_deck(self.mesh, self.mat, self.bc, None, 100.)
        self.bc["cracks"] = [{"edge": 1}]
        with self.assertRaisesRegex(ValueError, "Trincas"):
            nonlinear.input_deck(self.mesh, self.mat, self.bc, None, 20.)

    def test_missing_and_incomplete_output_are_rejected(self):
        with self.assertRaises(ValueError):
            nonlinear.parse_displacements("displacements (vx,vy,vz) for set NALL and time 1.0\n"
                                          "1 0 0 0\n", 4)

    def test_stress_parser_and_nodal_maps(self):
        header = ("stresses (elem, integ.pnt.,sxx,syy,szz,sxy,sxz,syz) "
                  "for set ALLSHELLS and time 1.0\n\n")
        rows = ("1 1 100 0 0 0 0 0 _shell_1\n"
                "1 2 -200 0 0 0 0 0 _shell_1\n"
                "2 1 0 0 0 10 0 0 _shell_2\n"
                "2 2 0 0 0 20 0 0 _shell_2\n")
        stress = nonlinear.parse_stresses(header + rows, 2)
        fields = nonlinear.nodal_stress_fields(self.mesh, stress)
        np.testing.assert_allclose(fields["vm"], [(200 + np.sqrt(3) * 20) / 2,
                                                  np.sqrt(3) * 20, 200,
                                                  (200 + np.sqrt(3) * 20) / 2])
        np.testing.assert_allclose(fields["p1"], [60, 20, 100, 60])
        np.testing.assert_allclose(fields["p2"], [-110, -20, -200, -110])
        with self.assertRaisesRegex(ValueError, "incompletos"):
            nonlinear.parse_stresses(header + rows.splitlines()[0] + "\n", 2)
        with self.assertRaisesRegex(ValueError, "inconsistentes"):
            nonlinear.parse_stresses(header + rows + rows.splitlines()[0] + "\n", 2)
        with self.assertRaisesRegex(ValueError, "fim do carregamento"):
            nonlinear.parse_stresses(header.replace("time 1.0", "time 0.5") + rows, 2)

    def test_real_calculix_small_shell(self):
        self.require_solver()
        result = nonlinear.run_analysis(self.mesh, self.mat, self.bc, None, 20.)
        self.assertEqual(result.displacements.shape, (4, 3))
        np.testing.assert_allclose(result.displacements[:2], 0, atol=1e-12)
        self.assertLess(result.displacements[2, 2], 0)
        self.assertAlmostEqual(result.displacements[2, 2], -0.002214978, delta=0.00001)
        self.assertIn(b"stresses", result.output)
        self.assertEqual(result.stress_fields["vm"].shape, (4,))
        self.assertTrue(np.all(np.isfinite(result.stress_fields["vm"])))
        self.assertGreater(result.stress_fields["vm"].max(), 0)
        self.assertIn("0.100000E+01", result.status)

    def test_ui_keeps_linear_by_default_and_rejects_thermal(self):
        root = Path(__file__).resolve().parent.parent
        app = AppTest.from_file(str(root / "app.py"), default_timeout=120)
        app.session_state["show_calculation"] = True
        app.run()
        option = next(c for c in app.checkbox if "CalculiX experimental" in c.label)
        self.assertFalse(option.value)
        option.check().run()
        app.checkbox(key="thermal_default").check().run()
        next(n for n in app.number_input if n.label == "Temperatura do metal (°C)").set_value(100.).run()
        self.assertTrue(any("térmica" in error.value for error in app.error))
        self.assertIsNone(app.session_state["analysis"])
        self.assertFalse(app.exception)

    def test_ui_executes_real_solver_without_linear_compliance(self):
        self.require_solver()
        st.cache_data.clear()
        root = Path(__file__).resolve().parent.parent
        mesh = ShellMesh(nodes=self.mesh.nodes, triangles=self.mesh.triangles,
                         tri_surface=self.mesh.tri_surface, curves=self.mesh.curves,
                         curve_length={1: 10.}, bbox_diagonal=15.)
        with patch("mesher.mesh_step_shell", return_value=mesh), \
                patch("viewer.mesh_viewer", return_value=None) as viewer:
            app = AppTest.from_file(str(root / "app.py"), default_timeout=120)
            app.session_state["show_calculation"] = True
            app.run()
            app.number_input(key="len_scale_default").set_value(1.0).run()
            app.session_state["bc"]["forces"] = self.bc["forces"]
            next(c for c in app.checkbox if c.label == "Considerar gravidade (direção −Z)").uncheck().run()
            next(c for c in app.checkbox if "CalculiX experimental" in c.label).check().run()
            next(b for b in app.button if b.label == "Rodar").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertIsNotNone(app.session_state["analysis"].get("nonlinear"))
            self.assertTrue(any("Resultados não lineares" in h.value for h in app.subheader))
            self.assertTrue(any("Deslocamento máximo" in metric.label for metric in app.metric))
            self.assertFalse(any("Code Compliance" in h.value for h in app.markdown))
            kwargs = viewer.call_args.kwargs
            result = app.session_state["analysis"]["nonlinear"]
            np.testing.assert_allclose(kwargs["field"], result.stress_fields["vm"])
            self.assertEqual(kwargs["max_note"], "")
            self.assertTrue("Von Mises" in kwargs["field_label"])
            app.selectbox(key="nonlinear_field").set_value("Deslocamento em Z (mm) UZ").run()
            np.testing.assert_allclose(viewer.call_args.kwargs["field"], result.displacements[:, 2])
            app.slider(key="nonlinear_scale").set_value(2.0).run()
            np.testing.assert_allclose(viewer.call_args.kwargs["nodes"],
                                       mesh.nodes + 2 * result.displacements)
            app.selectbox(key="nonlinear_field").set_value(
                "Mínima tensão principal — pontos de integração (MPa)").run()
            np.testing.assert_allclose(viewer.call_args.kwargs["field"], result.stress_fields["p2"])
            self.assertTrue(viewer.call_args.kwargs["field_minimum"])
            self.assertFalse(app.exception)
            next(c for c in app.checkbox if c.label == "Mostrar deformada e mapa não linear").uncheck().run()
            self.assertIsNone(viewer.call_args.kwargs["field"])
            self.assertIsNone(viewer.call_args.kwargs["nodes"])
            self.assertFalse(app.exception)

    def test_large_rotation_differs_from_linear_calculix(self):
        executable = self.require_solver()
        nx, ny = 20, 2
        nodes = np.array([[i * 5., j * 5., 0.] for i in range(nx + 1) for j in range(ny + 1)])
        triangles = []
        for i in range(nx):
            for j in range(ny):
                a, b = i * (ny + 1) + j, (i + 1) * (ny + 1) + j
                triangles.extend([[a, b, b + 1], [a, b + 1, a + 1]])
        mesh = ShellMesh(nodes=nodes, triangles=np.array(triangles),
                         tri_surface=np.ones(len(triangles), dtype=int),
                         curves={1: np.array([[0, 1], [1, 2]])},
                         curve_length={1: 10.}, bbox_diagonal=101.)
        bc = dict(self.bc, forces=[
            {"node": nx * (ny + 1) + j, "F": [0., 0., -50. / (ny + 1)]}
            for j in range(ny + 1)])
        deck = nonlinear.input_deck(mesh, self.mat, bc, None, 20.)
        result = nonlinear.run_analysis(mesh, self.mat, bc, None, 20.)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "linear.inp").write_bytes(deck.replace(b"*STEP,NLGEOM,INC=1000", b"*STEP,INC=1000"))
            completed = subprocess.run([executable, "linear"], cwd=folder, capture_output=True,
                                       text=True, errors="replace", timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stdout[-2000:])
            linear = nonlinear.parse_displacements((path / "linear.dat").read_text(), len(nodes))
        self.assertAlmostEqual(result.displacements[-1, 2], -30.3493, delta=0.1)
        self.assertAlmostEqual(linear[-1, 2], -34.06275, delta=0.1)
        self.assertGreater(abs(result.displacements[-1, 0]), 5.)


if __name__ == "__main__":
    unittest.main()
