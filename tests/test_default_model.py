import hashlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import streamlit as st
import numpy as np
from streamlit.testing.v1 import AppTest

from mesher import mesh_step_shell

ROOT = Path(__file__).resolve().parent.parent


class DefaultModelTests(unittest.TestCase):
    def test_cover_pdf_download(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
        with patch.object(st, "download_button", wraps=st.download_button) as download:
            app.run()
        self.assertFalse(app.exception)
        self.assertFalse(any("O modelo padrão é o costado T-8003" in info.value for info in app.info))
        self.assertEqual(len(app.get("download_button")), 1)
        widget = app.get("download_button")[0]
        self.assertEqual(widget.proto.label, "Baixar Relatório PDF")
        self.assertEqual(widget.proto.type, "primary")
        self.assertEqual(download.call_args.args[1], (ROOT / "FINAL_REPORT.pdf").read_bytes())
        self.assertEqual(download.call_args.kwargs["file_name"], "FINAL_REPORT.pdf")
        self.assertEqual(download.call_args.kwargs["mime"], "application/pdf")

    def test_cover_reports_missing_pdf(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
        read_bytes = Path.read_bytes

        def missing_report(path):
            if path.name == "FINAL_REPORT.pdf":
                raise FileNotFoundError("relatório indisponível")
            return read_bytes(path)

        with patch.object(Path, "read_bytes", missing_report):
            app.run()
        self.assertFalse(app.exception)
        self.assertTrue(any("relatório indisponível" in error.value for error in app.error))
        self.assertEqual(len(app.get("download_button")), 0)
        self.assertTrue(any("Editar/Atualizar" in button.label for button in app.button))

    def test_default_and_manual_vessel(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
        app.run()
        self.assertEqual(app.title[0].value, "Memória de Cálculo - T-8003")
        self.assertEqual(len(app.get("image")), 1)
        self.assertEqual(app.get("image")[0].proto.imgs[0].alt, "Tanque T-8003")
        self.assertFalse(app.exception)
        app.button[0].click().run()
        self.assertFalse(app.get("image"))
        self.assertEqual(app.radio(key="manual_kind").value, "T-8003 (STL padrão)")
        self.assertEqual(app.number_input(key="len_scale_default").value, 1000.0)
        self.assertEqual(app.number_input(key="hydro_height_default").value, 6000.0)
        self.assertEqual(next(n.value for n in app.number_input if n.label == "Espessura da casca"), 6.35)
        self.assertTrue(app.checkbox(key="hydro_enabled").value)
        self.assertFalse(app.checkbox(key="thermal_default").value)
        self.assertFalse(app.checkbox(key="fill_default").value)
        self.assertTrue(app.session_state["analysis"]["result"].solved)
        self.assertAlmostEqual(app.session_state["bc"]["hydrostatic"]["rho"], 1e-9)
        initial = app.session_state["analysis"]
        app.run()
        self.assertIs(app.session_state["analysis"], initial)
        app.number_input(key="len_scale_default").set_value(500.0).run()
        self.assertEqual(app.number_input(key="len_scale_default").value, 500.0)
        expected = hashlib.md5((ROOT / "teste_tanque3_costado_m.stl").read_bytes()).hexdigest()
        self.assertEqual(app.session_state["file_sig"], expected)
        self.assertFalse(app.exception)
        app.radio(key="manual_kind").set_value("Vaso de pressão").run()
        self.assertEqual(app.number_input(key="len_scale_other").value, 1.0)
        self.assertNotEqual(app.session_state["file_sig"], expected)
        self.assertFalse(app.exception)
        self.assertTrue(any(c.label == "Considerar gravidade (direção −Z)" for c in app.checkbox))
        self.assertFalse(app.checkbox(key="hydro_enabled").value)
        self.assertTrue(app.checkbox(key="thermal_other").value)
        self.assertTrue(app.checkbox(key="fill_other").value)
        app.radio(key="manual_kind").set_value("T-8003 (STL padrão)").run()
        self.assertEqual(app.number_input(key="len_scale_default").value, 1000.0)
        self.assertFalse(app.exception)

    def test_uploaded_stl_uses_unity_scale(self):
        uploaded = SimpleNamespace(name="outro.stl",
                                   getvalue=lambda: (ROOT / "teste_tanque3_costado_m.stl").read_bytes())
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
        app.session_state["show_calculation"] = True
        with patch.object(st, "file_uploader", return_value=uploaded):
            app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.number_input(key="len_scale_other").value, 1.0)

    def test_default_scale_applies_to_mesh_coordinates(self):
        source = mesh_step_shell((ROOT / "teste_tanque3_costado_m.stl").read_bytes(), suffix=".stl")
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
        app.session_state["show_calculation"] = True
        st.cache_data.clear()
        with patch("mesher.mesh_step_shell", return_value=source), \
                patch("viewer.mesh_viewer", return_value=None) as render:
            app.run()
            scaled = render.call_args.args[0]
            np.testing.assert_allclose(scaled.nodes, source.nodes * 1000)
            self.assertAlmostEqual(scaled.bbox_diagonal, source.bbox_diagonal * 1000)
            next(b for b in app.button if b.label == "Recomeçar").click().run()
            self.assertIsNone(app.session_state["analysis"])
            self.assertFalse(app.checkbox(key="hydro_enabled").value)
            self.assertFalse(app.session_state["bc"]["curves"])
            app.number_input(key="len_scale_default").set_value(2.).run()
            np.testing.assert_allclose(render.call_args.args[0].nodes, source.nodes * 2)
            self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
