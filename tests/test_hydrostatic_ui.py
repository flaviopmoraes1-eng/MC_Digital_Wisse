import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class HydrostaticDensityUITests(unittest.TestCase):
    def test_density_is_converted_to_solver_units(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=120)
        app.session_state["show_calculation"] = True
        app.run()
        app.radio(key="manual_kind").set_value("Vaso de pressão").run()
        app.checkbox(key="hydro_enabled").check().run()
        height = app.number_input(key="hydro_height")
        self.assertEqual(height.proto.step, 200.0)
        height.set_value(1250.0).run()
        self.assertEqual(app.session_state["bc"]["hydrostatic"]["height"], 1250.0)
        height.set_value(1000.0).run()
        density = app.number_input(key="hydro_density_kg_m3")
        self.assertEqual(density.label, "Densidade do líquido (kg/m³)")
        self.assertEqual(density.value, 1000.0)
        self.assertAlmostEqual(app.session_state["bc"]["hydrostatic"]["rho"], 1e-9)
        self.assertTrue(any("0.00981 MPa" in c.value for c in app.caption))

        density.set_value(1200.0).run()
        self.assertAlmostEqual(app.session_state["bc"]["hydrostatic"]["rho"], 1.2e-9)
        self.assertTrue(any("0.01177 MPa" in c.value for c in app.caption))
        self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
