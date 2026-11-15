"""Cálculo e relatório do vaso de referência exibidos na abertura."""
from dataclasses import replace

import streamlit as st

import fea
import report
import vessel
from mesher import free_curves, mesh_step_shell


@st.cache_data(show_spinner=False)
def default_pdf() -> bytes:
    p = vessel.DEFAULT_VESSEL
    mesh = mesh_step_shell(vessel.build_vessel_step(p), divisions=60)
    mesh = replace(mesh, base_thickness=vessel.element_thickness(mesh.nodes, mesh.triangles, p))
    dofs = ["UX", "UY", "UZ", "RX", "RY", "RZ"]
    bc = {"curves": {tag: {d: 0.0 for d in dofs} for tag in free_curves(mesh)},
          "pressure": {int(tag): -2.1 for tag in set(mesh.tri_surface.tolist())},
          "corrosion": [], "forces": [], "edge_forces": {}, "cracks": []}
    gravity = {"g": 9810.0, "rho_steel": 7.85e-9,
               "water": {"rho": 1.0e-9, "ri": (p.shell_od - 2 * p.shell_t) / 2}}
    result = fea.run_analysis(mesh, fea.Material(210000.0, 0.3, p.shell_t), bc, gravity, 20.0, 300.0)
    if not result.solved:
        raise ValueError("Não foi possível resolver o modelo padrão para gerar o relatório inicial.")
    S, Sy, Sp = fea.allowable_sp(20.0)
    pl = float(result.vm_nodal["mid"].max())
    pb = float(max(result.vm_nodal["top"].max(), result.vm_nodal["bottom"].max()))
    rows = [("P<sub>M</sub>", "Membrana geral", S, 2.1 * p.shell_od / (2 * p.shell_t)),
            ("P<sub>L</sub>", "Membrana local", Sp, pl),
            ("P<sub>L</sub> + P<sub>B</sub>", "Membrana Local + Flexão", Sp, pb)]
    compliance = [(cat, desc, allow, act, "n/d" if allow is None or act is None else
                   ("REPROVADO" if act > allow else "APROVADO"))
                  for cat, desc, allow, act in rows]
    return report.generate({
        "mesh": mesh, "res": result, "bc": bc, "scale": 1.0,
        "S_allow": S, "Sy": Sy, "Sp": Sp, "temp": 20.0, "thick": p.shell_t,
        "source": "Entrada manual — vaso de pressão", "kind": "Vaso de pressão",
        "geometry": report.vessel_geometry(p, 2.1, vessel.CAP_T), "divisions": 60, "refinements": 0,
        "len_scale": 1.0, "inputs": report.analysis_inputs(210000.0, 0.3, p.shell_t, gravity, 20.0, 300.0),
        "compliance": compliance, "crack_rows": [], "ignore_dist": 300.0,
        "notes": report.ANALYSIS_NOTES,
    })
