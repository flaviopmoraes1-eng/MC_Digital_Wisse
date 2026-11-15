import hashlib
import json
import os
from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

import fea
import isometric
from mesher import mesh_step_shell
from viewer import mesh_viewer

DOFS = ["UX", "UY", "UZ", "RX", "RY", "RZ"]
DOF_LABELS = {"UX": "UX (translação X)", "UY": "UY (translação Y)", "UZ": "UZ (translação Z)",
              "RX": "RX (rotação X)", "RY": "RY (rotação Y)", "RZ": "RZ (rotação Z)"}

st.set_page_config(page_title="PIPE-FEA", layout="wide")
st.title("PIPE-FEA")
st.caption("Geometria (STEP/STL) → malha de casca → condições de contorno e cargas")

with st.sidebar:
    uploaded = st.file_uploader("Geometria (STEP, STL ou isométrico PDF)",
                                type=["step", "stp", "stl", "pdf"])
    is_pdf = uploaded is not None and uploaded.name.lower().endswith(".pdf")
    divisions = st.slider("Densidade da malha - STEP (divisões da diagonal)", 5, 100,
                          60 if is_pdf else 20)
    refine = st.checkbox("Refinar malha")
    refinements = st.slider("Níveis de refinamento", 1, 3, 1) if refine else 0
    show_mesh = st.checkbox("Mostrar malha (arestas dos elementos)", value=True)
    len_scale = st.number_input(
        "Fator de escala dos comprimentos", value=1.0, min_value=1e-9, format="%.6g",
        help="Multiplica as medidas numéricas do modelo (comprimentos, análise); a forma "
             "da geometria não muda. Ex.: 1000 se o arquivo está em m e você quer mm.")


@st.cache_data(show_spinner="Gerando malha...")
def build(data: bytes, suffix: str, divisions: int, refinements: int):
    return mesh_step_shell(data, divisions=divisions, refinements=refinements,
                           suffix=suffix)


def is_anchor(dofs: dict) -> bool:
    return all(dofs.get(d) == 0 for d in DOFS)


def edge_form(selected: list[int], bc: dict, lengths: dict):
    lens = [lengths.get(t, 0.0) for t in selected]
    st.markdown(f"**Arestas selecionadas:** {', '.join(map(str, selected))}")
    if len(selected) == 1:
        st.markdown(f"**Comprimento:** {lens[0]:.2f} mm")
    else:
        st.markdown("**Comprimentos:** " + "; ".join(
            f"#{t}: {l:.2f} mm" for t, l in zip(selected, lens)) + f" — **total {sum(lens):.2f} mm**")
    mode = st.radio("Restrição", ["Ancoragem (6 GDL = 0)", "Personalizada"], key="edge_mode")
    dofs = {}
    if mode.startswith("Ancoragem"):
        dofs = {d: 0.0 for d in DOFS}
    else:
        st.caption("Marque os GDL restritos e informe o valor imposto (0 = fixo). "
                   "Translações na unidade de comprimento do modelo; rotações em rad.")
        for d in DOFS:
            c1, c2 = st.columns([1.2, 1])
            on = c1.checkbox(DOF_LABELS[d], key=f"dof_on_{d}")
            val = c2.number_input("valor", value=0.0, key=f"dof_val_{d}",
                                  label_visibility="collapsed", disabled=not on,
                                  format="%.6g")
            if on:
                dofs[d] = float(val)
    c1, c2 = st.columns(2)
    if c1.button("Aplicar", key="apply_edge", type="primary", disabled=not dofs):
        for tag in selected:
            bc["curves"][tag] = dict(dofs)
        st.rerun()
    if c2.button("Remover", key="clear_edge"):
        for tag in selected:
            bc["curves"].pop(tag, None)
        st.rerun()


def edge_load_form(selected: list[int], bc: dict, lengths: dict):
    st.markdown("**Força nas arestas (total)**")
    st.caption("Força total em N nas direções globais X, Y e Z, distribuída uniformemente ao longo "
               "da(s) aresta(s) selecionada(s). Com várias arestas, o total é repartido "
               "proporcionalmente ao comprimento de cada uma.")
    cur = [sum(bc["edge_forces"].get(t, [0.0] * 3)[d] for t in selected) for d in range(3)]
    sig = "_".join(map(str, selected))
    total = [st.number_input(f"F{ax} total (N)", value=float(v), format="%.6g",
                             key=f"edge_F_{ax}_{sig}_{v}")
             for ax, v in zip("XYZ", cur)]
    c1, c2 = st.columns(2)
    if c1.button("Aplicar", key="apply_edge_q", type="primary", disabled=not any(total)):
        ltot = sum(lengths.get(t, 0.0) for t in selected)
        for tag in selected:
            share = lengths.get(tag, 0.0) / ltot if ltot > 0 else 1 / len(selected)
            bc["edge_forces"][tag] = [float(c) * share for c in total]
        st.rerun()
    if c2.button("Remover", key="clear_edge_q",
                 disabled=not any(t in bc["edge_forces"] for t in selected)):
        for tag in selected:
            bc["edge_forces"].pop(tag, None)
        st.rerun()


def surface_form(selected: list[int], bc: dict):
    st.markdown(f"**Superfícies selecionadas:** {', '.join(map(str, selected))}")
    p = st.number_input("Pressão (unidades consistentes, ex.: MPa com mm/N)",
                        value=0.0, format="%.6g", key="pressure_val",
                        help="Positiva = atua de fora para dentro (contrária à normal externa, "
                             "pressão externa). Negativa = pressão interna.")
    c1, c2 = st.columns(2)
    if c1.button("Aplicar", key="apply_surf", type="primary"):
        for tag in selected:
            bc["pressure"][tag] = float(p)
        st.rerun()
    if c2.button("Remover", key="clear_surf"):
        for tag in selected:
            bc["pressure"].pop(tag, None)
        st.rerun()


def corrosion_form(node, bc: dict, mesh, nominal: float):
    st.caption("Área corroída: círculo de raio R (medido sobre a superfície) centrado no nó, "
               "com espessura reduzida. Os elementos dentro da área usam a espessura corroída.")
    if node is None:
        st.info("Clique em um nó do modelo para definir o centro da corrosão.")
    else:
        x, y, z = mesh.nodes[node]
        st.markdown(f"**Nó selecionado:** {node}  \n({x:.1f}, {y:.1f}, {z:.1f}) mm")
    default_t = round(0.5 * nominal, 6)
    if st.session_state.get("corr_t", default_t) >= nominal:
        st.session_state.corr_t = default_t
    radius = st.number_input(
        "Raio da área corroída (mm)",
        min_value=1e-6, value=100.0, step=5.0, key="corr_radius")
    t_corr = st.number_input(
        f"Espessura corroída (nominal = {nominal:g})", min_value=1e-6, max_value=float(nominal) * 0.999,
        value=default_t, format="%.6g", key="corr_t")
    c1, c2 = st.columns(2)
    if c1.button("Adicionar", key="add_corr", type="primary", disabled=node is None):
        bc["corrosion"].append({"node": int(node), "radius": float(radius), "t": float(t_corr)})
        st.rerun()
    if c2.button("Remover todas", key="clear_corr", disabled=not bc["corrosion"]):
        bc["corrosion"].clear()
        st.rerun()


def force_form(node, bc: dict, mesh):
    st.caption("Força concentrada no nó, em N, nas direções globais X, Y e Z. "
               "Uma nova força no mesmo nó substitui a anterior.")
    current = next((l for l in bc["forces"] if l["node"] == node), None)
    if node is None:
        st.info("Clique em um nó do modelo para aplicar a força.")
    else:
        x, y, z = mesh.nodes[node]
        st.markdown(f"**Nó selecionado:** {node}  \n({x:.1f}, {y:.1f}, {z:.1f}) mm")
    default = current["F"] if current else [0.0, 0.0, 0.0]
    comps = [st.number_input(f"F{ax} (N)", value=float(v), format="%.6g",
                             key=f"force_{ax}_{node}_{v}")
             for ax, v in zip("XYZ", default)]
    c1, c2 = st.columns(2)
    if c1.button("Aplicar", key="apply_force", type="primary",
                 disabled=node is None or not any(comps)):
        bc["forces"] = [l for l in bc["forces"] if l["node"] != node]
        bc["forces"].append({"node": int(node), "F": [float(c) for c in comps]})
        st.rerun()
    if c2.button("Remover", key="clear_force", disabled=current is None):
        bc["forces"] = [l for l in bc["forces"] if l["node"] != node]
        st.rerun()


def summary(bc: dict, nominal: float, sig: str):
    """Editable table of assigned conditions; any applied edit triggers a new analysis."""
    nan = float("nan")
    cols = ["Pressão", "FX (N)", "FY (N)", "FZ (N)", "Raio (mm)", "Espessura corroída"] + DOFS
    rows, refs = [], []

    def add(kind, ident, **vals):
        rows.append({"Tipo": kind, "ID": ident, **{c: vals.get(c, nan) for c in cols}})
        refs.append(kind)

    for tag, dofs in sorted(bc["curves"].items()):
        add("Aresta", tag, **{d: dofs.get(d, nan) for d in DOFS})
        refs[-1] = ("curve", tag)
    for tag, q in sorted(bc["edge_forces"].items()):
        add("Força total (aresta)", tag, **dict(zip(["FX (N)", "FY (N)", "FZ (N)"], q)))
        refs[-1] = ("edge_force", tag)
    for tag, p in sorted(bc["pressure"].items()):
        add("Superfície", tag, **{"Pressão": p})
        refs[-1] = ("pressure", tag)
    for i, l in enumerate(bc["forces"]):
        add("Força (nó)", l["node"], **dict(zip(["FX (N)", "FY (N)", "FZ (N)"], l["F"])))
        refs[-1] = ("force", i)
    for i, d in enumerate(bc["corrosion"]):
        add("Corrosão", d["node"], **{"Raio (mm)": d["radius"], "Espessura corroída": d["t"]})
        refs[-1] = ("corrosion", i)
    if not rows:
        return
    st.markdown("**Condições atribuídas**")
    st.caption("Edite os valores diretamente na tabela; cada alteração recalcula a análise. "
               "Nas arestas, célula vazia = GDL livre; arestas sem nenhum GDL são removidas.")
    edited = st.data_editor(
        pd.DataFrame(rows), hide_index=True, width="stretch", disabled=["Tipo", "ID"],
        key=f"summary-{sig}",
        column_config={c: st.column_config.NumberColumn(format="%.6g") for c in cols})

    changed = False
    invalid = False
    for (kind, ref), (_, row) in zip(refs, edited.iterrows()):
        if kind == "curve":
            new = {d: float(row[d]) for d in DOFS if pd.notna(row[d])}
            if new != bc["curves"][ref]:
                if new:
                    bc["curves"][ref] = new
                else:
                    bc["curves"].pop(ref)
                changed = True
        elif kind == "pressure":
            if pd.notna(row["Pressão"]) and float(row["Pressão"]) != bc["pressure"][ref]:
                bc["pressure"][ref] = float(row["Pressão"])
                changed = True
        elif kind == "edge_force":
            q = [row["FX (N)"], row["FY (N)"], row["FZ (N)"]]
            if all(pd.notna(v) for v in q) and [float(v) for v in q] != bc["edge_forces"][ref]:
                bc["edge_forces"][ref] = [float(v) for v in q]
                changed = True
        elif kind == "force":
            F = [row["FX (N)"], row["FY (N)"], row["FZ (N)"]]
            if all(pd.notna(v) for v in F) and [float(v) for v in F] != bc["forces"][ref]["F"]:
                bc["forces"][ref]["F"] = [float(v) for v in F]
                changed = True
        else:
            r, t = row["Raio (mm)"], row["Espessura corroída"]
            d = bc["corrosion"][ref]
            if pd.isna(r) or pd.isna(t) or r <= 0 or not 0 < t < nominal:
                invalid = invalid or (float(r), float(t)) != (d["radius"], d["t"])
            elif (float(r), float(t)) != (d["radius"], d["t"]):
                d["radius"], d["t"] = float(r), float(t)
                changed = True
    if invalid and not changed:
        st.warning(f"Corrosão: raio deve ser > 0 e espessura corroída entre 0 e {nominal:g} (exclusive).")
    if changed:
        st.session_state.recalc = True
        st.rerun()


FIELDS = {
    "Von Mises (MPa)": "vm",
    "Deformação equivalente (%)": "eps",
    "Deslocamento resultante (mm)": "disp",
    "Deslocamento em X (mm) UX": "ux",
    "Deslocamento em Y (mm) UY": "uy",
    "Deslocamento em Z (mm) UZ": "uz",
}
COMPONENTS = {"ux": 0, "uy": 1, "uz": 2}
CORR_REF_STRESS = 220.0  # MPa
LAYER_DEFAULT = "PL+PB_top — Tensão de Membrana + Flexão"
VM_LAYERS = {
    "PL — Tensão de Membrana": ("mid", "PL"),
    "PL+PB_bottom — Tensão de Membrana + Flexão": ("bottom", "PL+PB_bottom"),
    "PL+PB_top — Tensão de Membrana + Flexão": ("top", "PL+PB_top"),
}


def fmt_vec(v):
    return ", ".join(f"{x:.4g}" for x in v)


def equilibrium_report(res):
    eq = res.equilibrium
    if eq.ok:
        st.success("Equilíbrio estático verificado: " + eq.messages[0])
    else:
        st.error("Equilíbrio estático NÃO satisfeito - análise não executada. " + " ".join(eq.messages))
    rows = [
        {"Verificação": "Resultante das cargas - força (Fx, Fy, Fz)", "Valor": fmt_vec(eq.force)},
        {"Verificação": "Resultante das cargas - momento (Mx, My, Mz)", "Valor": fmt_vec(eq.moment)},
        {"Verificação": "Movimentos de corpo rígido restringidos", "Valor": f"{eq.restrained_modes} de 6"},
    ]
    if res.solved:
        p = res.post
        rows += [
            {"Verificação": "Reações totais - força", "Valor": fmt_vec(p["reaction_force"])},
            {"Verificação": "Reações totais - momento", "Valor": fmt_vec(p["reaction_moment"])},
            {"Verificação": "Erro de equilíbrio (cargas + reações), força",
             "Valor": f"{p['force_error']:.2e}"},
            {"Verificação": "Erro de equilíbrio (cargas + reações), momento",
             "Valor": f"{p['moment_error']:.2e}"},
        ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


@st.cache_data(show_spinner="Lendo isométrico...")
def read_iso(data: bytes):
    return isometric.parse_isometric(data)


@st.cache_data(show_spinner="Gerando a casca da tubulação...")
def pipe_step(pts: list, mean_diameter: float, radius: float) -> bytes:
    return isometric.build_pipe_step(np.array(pts), mean_diameter, radius)


def iso_input(data: bytes):
    """Isométrico PDF -> bytes STEP da casca; devolve também a espessura de parede."""
    try:
        iso = read_iso(data)
    except Exception as exc:
        st.error(f"Falha ao ler o isométrico: {exc}")
        st.stop()
    sig = hashlib.md5(data).hexdigest()[:8]
    with st.expander("Isométrico (PDF) → geometria da tubulação", expanded=True):
        c_img, c_par = st.columns([1.3, 1])
        c_img.image(iso.page_png, caption="Isométrico lido (linha de centro = traço mais grosso)")
        with c_par:
            nps = st.number_input("DN (pol)", value=float(iso.nps or 6.0), min_value=0.5, step=0.5,
                                  key=f"nps-{sig}")
            od_default = isometric.OD_TABLE.get(nps, nps * isometric.IN)
            od = st.number_input("Diâmetro externo (mm)", value=float(od_default), min_value=1.0,
                                 format="%.2f", key=f"od-{sig}-{nps}")
            wall = st.number_input("Espessura de parede (mm)", value=float(iso.wall or 7.11),
                                   min_value=0.1, format="%.2f", key=f"wall-{sig}")
            factor = st.number_input("Raio das curvas (× DN)", value=isometric.BEND_FACTOR,
                                     min_value=1.0, step=0.5, key=f"bf-{sig}")
            cond = st.segmented_control(
                "Pressão lida do isométrico", ["Projeto", "Operação"], default="Projeto",
                key=f"pcond-{sig}", disabled=iso.pressure_design is None)
            p_read = iso.pressure_op if cond == "Operação" else iso.pressure_design
            p_int = st.number_input("Pressão interna (MPa)", value=float(p_read or 0.0),
                                    min_value=0.0, format="%.6g", key=f"pint-{sig}-{cond}",
                                    help="Lida da tabela do isométrico (bar g → MPa) e aplicada "
                                         "automaticamente como pressão interna em todas as superfícies.")
            radius = factor * nps * isometric.IN
            st.caption(f"Raio de curvatura = {factor:g} × {nps:g}\" = {radius / isometric.IN:.2f}\" "
                       f"({radius:.1f} mm). A casca fica na superfície média (Ø = DE − e).")
        table = pd.DataFrame({
            "Trecho": range(1, len(iso.segments) + 1),
            "Direção": [("+" if s.sign > 0 else "−") + {"U": "Elev."}.get(s.axis, s.axis)
                        for s in iso.segments],
            "Comprimento (mm)": [s.length for s in iso.segments],
            "Origem": [s.source for s in iso.segments],
        })
        edited = st.data_editor(table, hide_index=True, use_container_width=True, key=f"seg-{sig}",
                                disabled=["Trecho", "Direção", "Origem"],
                                column_config={"Comprimento (mm)": st.column_config.NumberColumn(
                                    min_value=1.0, format="%.1f")})
        for m in iso.messages:
            st.warning(m)
    segs = [isometric.Segment(s.axis, s.sign, float(L), s.source)
            for s, L in zip(iso.segments, edited["Comprimento (mm)"])]
    pts = isometric.centerline_points(segs)
    try:
        step = pipe_step(pts.round(4).tolist(), float(od - wall), float(radius))
    except Exception as exc:
        st.error(f"Não foi possível gerar a tubulação: {exc}")
        st.stop()
    return step, float(wall), float(od), float(p_int)


if uploaded is None:
    st.info("Envie um arquivo .step/.stp/.stl ou um isométrico .pdf na barra lateral.")
    st.stop()

raw = uploaded.getvalue()
suffix = os.path.splitext(uploaded.name)[1]
iso_wall = iso_od = iso_pressure = None
if is_pdf:
    raw, iso_wall, iso_od, iso_pressure = iso_input(raw)
    suffix = ".step"
file_sig = hashlib.md5(raw).hexdigest()
try:
    mesh = build(raw, suffix, divisions, refinements)
except Exception as exc:
    st.error(f"Falha ao gerar a malha: {exc}")
    st.stop()

if len_scale != 1.0:
    mesh = replace(
        mesh, nodes=mesh.nodes * len_scale, bbox_diagonal=mesh.bbox_diagonal * len_scale,
        curve_length={t: l * len_scale for t, l in mesh.curve_length.items()})

if st.session_state.get("file_sig") != file_sig:
    st.session_state.file_sig = file_sig
    st.session_state.bc = {"curves": {}, "pressure": {}, "corrosion": [], "forces": [],
                           "edge_forces": {}}
    st.session_state.sel = {"curves": [], "surfaces": [], "node": None}
bc = st.session_state.bc
bc.setdefault("corrosion", [])
bc.setdefault("forces", [])
bc.setdefault("edge_forces", {})
# Node indices change with the mesh, so corrosion defects are tied to it.
mesh_key = (file_sig, divisions, refinements)
if st.session_state.get("mesh_key") != mesh_key:
    st.session_state.mesh_key = mesh_key
    if bc["corrosion"] or bc["forces"]:
        bc["corrosion"].clear()
        bc["forces"].clear()
        st.toast("Malha alterada: os defeitos de corrosão e as forças nodais foram removidos.")
    st.session_state.sel["node"] = None
surface_tags = set(mesh.tri_surface.tolist())
# Pressão interna do isométrico: aplicada (negativa) em todas as superfícies quando lida/alterada.
if iso_pressure:
    p_key = (file_sig, iso_pressure)
    if st.session_state.get("iso_p_applied") != p_key:
        st.session_state.iso_p_applied = p_key
        for tag in surface_tags:
            bc["pressure"][tag] = -iso_pressure


@st.cache_data(show_spinner=False)
def zone_tris(_mesh, key, node: int, radius: float):
    return np.flatnonzero(fea.corrosion_zone(_mesh.nodes, _mesh.triangles, node, radius))

analysis_sig = hashlib.md5(json.dumps(
    [file_sig, divisions, refinements, len_scale, bc], sort_keys=True, default=str).encode()).hexdigest()

def reset_all():
    st.session_state.bc = {"curves": {}, "pressure": {}, "corrosion": [], "forces": [],
                           "edge_forces": {}}
    st.session_state.sel = {"curves": [], "surfaces": [], "node": None}
    st.session_state.analysis = None
    # New key remounts the viewer, discarding its stored selection.
    st.session_state.reset_n = st.session_state.get("reset_n", 0) + 1


with st.sidebar:
    st.divider()
    st.subheader("Análise")
    E = st.number_input("Módulo de Young E", value=210000.0, min_value=1e-9, format="%.6g")
    nu = st.number_input("Coeficiente de Poisson", value=0.3, min_value=0.0, max_value=0.499)
    thick = st.number_input("Espessura da casca", value=iso_wall or 10.0, min_value=1e-9, format="%.6g")
    st.caption("Use unidades consistentes (ex.: mm, N, MPa).")
    use_gravity = st.checkbox(
        "Considerar gravidade (direção −Z)", value=True,
        help="Peso próprio da casca como carga distribuída nos nós, atuando em −Z.")
    gravity = None
    if use_gravity:
        g = st.number_input("Aceleração da gravidade", value=9810.0, min_value=1e-9, format="%.6g",
                            help="9810 mm/s² para mm, N, MPa, t (tonelada).")
        rho_s = st.number_input("Densidade do aço", value=7.85e-9, min_value=1e-15, format="%.4g",
                                help="7,85e-9 t/mm³ (= 7850 kg/m³) com mm, N, MPa.")
        gravity = {"g": g, "rho_steel": rho_s}
        fill = st.checkbox("Tubulação cheia de água", value=True,
                           help="Soma o peso da água ao peso próprio (casca circular).")
        if fill:
            rho_w = st.number_input("Densidade da água", value=1.0e-9, min_value=1e-15,
                                    format="%.4g", help="1,0e-9 t/mm³ (= 1000 kg/m³).")
            d_in = st.number_input(
                "Diâmetro interno (mm)",
                value=float(iso_od - 2 * iso_wall) if iso_od else 100.0,
                min_value=1.0, format="%.2f", key=f"d_in-{iso_od}")
            gravity["water"] = {"rho": rho_w, "ri": d_in / 2}
    run = st.button("Verificar equilíbrio e calcular", type="primary")
    st.button("Recomeçar", on_click=reset_all,
              help="Remove todas as cargas, condições de contorno e defeitos de corrosão.")

mat = fea.Material(E, nu, thick)
full_sig = f"{analysis_sig}-{E}-{nu}-{thick}-{json.dumps(gravity, sort_keys=True)}"
if run or st.session_state.pop("recalc", False):
    try:
        with st.spinner("Verificando equilíbrio e calculando..."):
            st.session_state.analysis = {"sig": full_sig,
                                         "result": fea.run_analysis(mesh, mat, bc, gravity)}
    except ValueError as exc:
        st.session_state.analysis = None
        st.sidebar.error(str(exc))

analysis = st.session_state.get("analysis")
res = analysis["result"] if analysis and analysis["sig"] == full_sig else None
if analysis and res is None:
    st.sidebar.warning("Dados alterados desde o último cálculo: recalcule para ver os resultados.")

view_nodes, view_ghost, view_field, view_label = None, None, None, ""
c1, c2, c3, c4 = st.columns(4)
c1.metric("Nós", len(mesh.nodes))
c2.metric("Elementos (triângulos)", len(mesh.triangles))
c3.metric("Superfícies", len(surface_tags))
c4.metric("Arestas", len(mesh.curves))
if mesh.welded:
    st.caption(f"Pré-processamento: {mesh.welded} nós duplicados nas arestas comuns foram unidos.")
left, right = st.columns([3, 1.4])
with right:
    sel_mode = st.radio("Seleção no modelo",
                        ["Arestas e superfícies", "Nó (corrosão)", "Nó (força)"], key="sel_mode")
node_mode = sel_mode.startswith("Nó")
corr_mode = sel_mode == "Nó (corrosão)"
if res is not None and res.solved:
    auto = 0.05 * mesh.bbox_diagonal / res.u_max if res.u_max > 0 else 1.0
    with st.sidebar:
        st.divider()
        st.subheader("Resultados")
        show_res = st.checkbox("Mostrar deformada e resultados no modelo", value=True)
        with left:
            field_name = st.selectbox("Campo", list(FIELDS))
        layer = None
        if FIELDS[field_name] in ("vm", "eps"):
            with left:
                layer = st.selectbox("Posição na espessura", list(VM_LAYERS),
                                     index=list(VM_LAYERS).index(LAYER_DEFAULT))
        scale = st.slider(
            "Fator de escala da deformação", 0.0, float(auto * 10), float(auto),
            step=float(auto / 50), format="%.4g",
            help="Multiplica os deslocamentos apenas na visualização (1 = real). "
                 f"O valor inicial ({auto:.4g}) deixa o deslocamento máximo em 5% da "
                 "diagonal do modelo.")
        show_ghost = st.checkbox("Mostrar malha indeformada (referência)", value=False)
    if show_res:
        if FIELDS[field_name] == "disp":
            view_label = "Deslocamento resultante |U| (mm)"
            view_field = np.linalg.norm(res.u[:, :3], axis=1)
        elif FIELDS[field_name] in COMPONENTS:
            view_label = f"Deslocamento {field_name[-2:]} (mm)"
            view_field = res.u[:, COMPONENTS[FIELDS[field_name]]]
        else:
            suffix, view_label = VM_LAYERS[layer]
            prefix, unit = ("eps_", "Deformação equivalente (%)") if FIELDS[field_name] == "eps" \
                else ("", "Von Mises (MPa)")
            view_label = f"{unit} — {view_label}"
            view_field = res.vm_nodal[prefix + suffix]
        view_nodes = mesh.nodes + res.u[:, :3] * scale
        if show_ghost and scale > 0:
            view_ghost = mesh.nodes
        st.sidebar.caption(f"Deformada ampliada {scale:.4g}× — deslocamento máx. real "
                           f"{res.u_max:.4g}, visualizado {res.u_max * scale:.4g}.")

with left:
    zone_key = (file_sig, divisions, refinements, len_scale)
    corr_tris = [zone_tris(mesh, zone_key, d["node"], d["radius"]) for d in bc["corrosion"]]
    sel_node = st.session_state.sel.get("node")
    if corr_mode and sel_node is not None:
        corr_tris.append(zone_tris(mesh, zone_key, sel_node,
                                   float(st.session_state.get("corr_radius", 100.0))))
    # Na tela de resultados só as tensões são plotadas, sem destacar a perda de espessura
    corr_union = np.unique(np.concatenate(corr_tris)) if corr_tris and view_field is None else []
    corr_flags = []
    if view_field is not None:
        # Utilização = máx. Von Mises no meio da espessura da zona / tensão de referência
        vm_mid = res.vm_element["mid"]
        for d, zone in zip(bc["corrosion"], corr_tris):
            if len(zone) == 0:
                continue
            k = zone[np.argmax(vm_mid[zone])]
            tri = mesh.triangles[k]
            ratio = vm_mid[k] / CORR_REF_STRESS * 100
            verdict = "Reprovado" if ratio > 100 else "Aprovado"
            corr_flags.append({"node": int(tri[np.argmax(res.vm_nodal["mid"][tri])]),
                               "text": f"{ratio:.0f}% - {verdict} - ASME API-579 part 5 Nível 3"})
    result = mesh_viewer(
        mesh, f"{file_sig}-{divisions}-{refinements}-{len_scale}", st.session_state.sel,
        {t: "anchor" if is_anchor(d) else "custom" for t, d in bc["curves"].items()},
        bc["pressure"], show_mesh, nodes=view_nodes, ghost=view_ghost, field=view_field, field_label=view_label,
        mode="node" if node_mode else "edge", corr_tris=corr_union, corr_flags=corr_flags,
        key=f"viewer-{st.session_state.get('reset_n', 0)}",
    )
    picked = result.selection if result is not None else None
    if picked:
        node = picked.get("node")
        st.session_state.sel = {
            "curves": [c for c in picked.get("curves", []) if c in mesh.curves],
            "surfaces": [s for s in picked.get("surfaces", []) if s in surface_tags],
            "node": node if isinstance(node, int) and 0 <= node < len(mesh.nodes) else None,
        }
sel = st.session_state.sel

with right:
    if corr_mode:
        with st.container(border=True):
            st.subheader("Corrosão (nó)")
            corrosion_form(sel.get("node"), bc, mesh, thick)
    elif node_mode:
        with st.container(border=True):
            st.subheader("Força (nó)")
            force_form(sel.get("node"), bc, mesh)
    else:
        if not sel["curves"] and not sel["surfaces"]:
            st.info("Selecione arestas ou superfícies no modelo para atribuir condições.")
        if sel["curves"]:
            with st.container(border=True):
                st.subheader("Restrições e cargas (aresta)")
                edge_form(sel["curves"], bc, mesh.curve_length)
                st.divider()
                edge_load_form(sel["curves"], bc, mesh.curve_length)
        if sel["surfaces"]:
            with st.container(border=True):
                st.subheader("Pressão (superfície)")
                surface_form(sel["surfaces"], bc)
summary(bc, thick, analysis_sig)

if res is not None:
    st.divider()
    st.subheader("Resultados")
    equilibrium_report(res)
    if res.solved:
        m1, m2, m3 = st.columns(3)
        m1.metric("Von Mises máx. (elemento)", f"{res.vm_element['max'].max():.4g}")
        m2.metric("Von Mises máx. (nodal suavizado)", f"{res.vm_nodal['max'].max():.4g}")
        m3.metric("Deslocamento máx.", f"{res.u_max:.4g}")
        st.caption("Tensões em elemento constante (CST) na superfície topo/base da casca; o campo "
                   "mostrado é a média nodal ponderada por área. Pico de tensão em apoios/quinas "
                   "depende da malha: refine para verificar convergência.")
