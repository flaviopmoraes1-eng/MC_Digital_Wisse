"""Relatório técnico em PDF (entradas, saídas, code compliance e mapas 3D de Von Mises)."""
import io
import os
from datetime import datetime

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

import fea

REPORT_TITLE = "Memória de Cálculo - T-8003"

NAVY = colors.HexColor("#14213d")
STEEL = colors.HexColor("#3d5a80")
LIGHT = colors.HexColor("#e8edf3")
GRID = colors.HexColor("#b8c2cf")
RED = colors.HexColor("#b00020")
GREEN = colors.HexColor("#1b7a3a")

_FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
for name, file in (("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"),
                   ("DV-I", "DejaVuSans-Oblique.ttf"), ("DV-BI", "DejaVuSans-BoldOblique.ttf")):
    if name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(name, os.path.join(_FONT_DIR, file)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

BASE = ParagraphStyle("base", fontName="DV", fontSize=8.5, leading=11.5)
SMALL = ParagraphStyle("small", parent=BASE, fontSize=7.3, leading=9.5, textColor=colors.HexColor("#444444"))
CELL = ParagraphStyle("cell", parent=BASE, fontSize=7.8, leading=10)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="DV-B")
HEAD = ParagraphStyle("head", parent=CELL, fontName="DV-B", textColor=colors.white)
H1 = ParagraphStyle("h1", parent=BASE, fontName="DV-B", fontSize=12.5, leading=16, textColor=NAVY,
                    spaceBefore=10, spaceAfter=5)
H2 = ParagraphStyle("h2", parent=BASE, fontName="DV-B", fontSize=9.5, leading=13, textColor=STEEL,
                    spaceBefore=7, spaceAfter=3)
TITLE = ParagraphStyle("title", parent=BASE, fontName="DV-B", fontSize=19, leading=24, textColor=NAVY)
SUB = ParagraphStyle("sub", parent=BASE, fontSize=10, leading=14, textColor=STEEL)
CENTER = ParagraphStyle("center", parent=SMALL, alignment=TA_CENTER)


def g(x, n: int = 4) -> str:
    if x is None:
        return "—"
    try:
        if isinstance(x, (float, np.floating)) and not np.isfinite(x):
            return "—"
        return f"{float(x):.{n}g}"
    except (TypeError, ValueError):
        return str(x)


def ids(values) -> str:
    """Lista de inteiros compactada em faixas: 1–5, 8, 10–12."""
    v = sorted({int(i) for i in values})
    out, i = [], 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[j + 1] == v[j] + 1:
            j += 1
        out.append(str(v[i]) if i == j else (f"{v[i]}, {v[j]}" if j == i + 1 else f"{v[i]}–{v[j]}"))
        i = j + 1
    return ", ".join(out)


def P(text, style=CELL):
    return Paragraph(str(text), style)


def table(data, widths, header=True, align_right_from: int | None = None, repeat=1):
    rows = []
    for r, row in enumerate(data):
        st = HEAD if header and r == 0 else CELL
        rows.append([c if not isinstance(c, (str, int, float)) else P(c, st) for c in row])
    t = Table(rows, colWidths=widths, repeatRows=repeat if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, GRID), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), STEEL)]
        for r in range(2, len(rows), 2):
            style.append(("BACKGROUND", (0, r), (-1, r), LIGHT))
    else:
        for r in range(0, len(rows), 2):
            style.append(("BACKGROUND", (0, r), (-1, r), LIGHT))
    t.setStyle(TableStyle(style))
    return t


def kv_table(pairs, widths=(60 * mm, 114 * mm)):
    data = [[P(k, CELLB), P(v)] for k, v in pairs]
    t = table(data, widths, header=False)
    return t


def status_cell(txt):
    col = "#b00020" if "REPROV" in txt else ("#1b7a3a" if "APROV" in txt else "#444444")
    return P(f'<font color="{col}"><b>{txt}</b></font>')


class _Numbered(rl_canvas.Canvas):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._pages = []

    def showPage(self):
        self._pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._pages)
        for state in self._pages:
            self.__dict__.update(state)
            self._decorate(total)
            super().showPage()
        super().save()

    def _decorate(self, total):
        w, h = A4
        self.setFillColor(NAVY)
        self.rect(0, h - 14 * mm, w, 14 * mm, stroke=0, fill=1)
        self.setFillColor(colors.white)
        self.setFont("DV-B", 10)
        self.drawString(18 * mm, h - 9 * mm, "PIPE-FEA — Relatório de análise por elementos finitos")
        self.setFont("DV", 8)
        self.drawRightString(w - 18 * mm, h - 9 * mm, getattr(self, "_title", ""))
        self.setStrokeColor(GRID)
        self.line(18 * mm, 13 * mm, w - 18 * mm, 13 * mm)
        self.setFillColor(colors.HexColor("#555555"))
        self.setFont("DV", 7.5)
        self.drawString(18 * mm, 8.5 * mm, "Documento gerado automaticamente — requer verificação e aprovação "
                                           "de engenheiro habilitado.")
        self.drawRightString(w - 18 * mm, 8.5 * mm, f"Página {self._pageNumber} de {total}")


# ---------------------------------------------------------------- imagens 3D

def _shade(nodes, tri, view):
    p = nodes[tri]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    el, az = np.radians(view[0]), np.radians(view[1])
    cam = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return 0.55 + 0.45 * np.abs(n @ cam)


def _setup_axes(fig, rect, nodes, view):
    ax = fig.add_axes(rect, projection="3d")
    lo, hi = nodes.min(axis=0), nodes.max(axis=0)
    span = np.maximum(hi - lo, 1e-6 * max(float(np.max(hi - lo)), 1.0))
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(span, zoom=1.25)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_axis_off()
    return ax


def _png(fig) -> bytes:
    buf = io.BytesIO()
    FigureCanvasAgg(fig)
    fig.savefig(buf, format="png", dpi=150, facecolor="white")
    return buf.getvalue()


VIEWS = ((24, -58), (24, 122))
FIELD_COLORS = [(0, 0, 0.6), (0, 0.35, 1), (0, 0.9, 0.9), (0.2, 0.85, 0.2),
                (1, 0.9, 0), (1, 0.45, 0), (0.8, 0, 0)]
FIELD_CMAP = LinearSegmentedColormap.from_list("viewer_ramp", FIELD_COLORS, N=4096)


def _interpolated_field_faces(nodes, tri, vals):
    points = nodes[tri]
    values = vals[tri]
    midpoint = (points + np.roll(points, -1, axis=1)) / 2
    midpoint_values = (values + np.roll(values, -1, axis=1)) / 2
    faces = np.stack([
        np.stack([points[:, 0], midpoint[:, 0], midpoint[:, 2]], axis=1),
        np.stack([points[:, 1], midpoint[:, 1], midpoint[:, 0]], axis=1),
        np.stack([points[:, 2], midpoint[:, 2], midpoint[:, 1]], axis=1),
        midpoint,
    ], axis=1)
    colors = np.stack([
        (values[:, 0] + midpoint_values[:, 0] + midpoint_values[:, 2]) / 3,
        (values[:, 1] + midpoint_values[:, 1] + midpoint_values[:, 0]) / 3,
        (values[:, 2] + midpoint_values[:, 2] + midpoint_values[:, 1]) / 3,
        midpoint_values.mean(axis=1),
    ], axis=1)
    return faces.reshape(-1, 3, 3), colors.ravel()


def render_field(nodes, tri, vals, label, vmax, marker=None) -> bytes:
    """Duas vistas isométricas opostas do campo nodal `vals` com barra de cores."""
    fig = Figure(figsize=(8.2, 9.6))
    vmin = float(np.min(vals))
    norm = Normalize(vmin, vmax if vmax > vmin else vmin + 1.0)
    faces, face_values = _interpolated_field_faces(nodes, tri, vals)
    face_colors = FIELD_CMAP(norm(face_values))
    for i, view in enumerate(VIEWS):
        ax = _setup_axes(fig, [0.0, 0.50 - 0.5 * i + 0.01, 0.86, 0.47], nodes, view)
        ax.add_collection3d(Poly3DCollection(faces, facecolors=face_colors, edgecolors="none",
                                             linewidths=0, antialiaseds=False))
        if marker is not None:
            ax.scatter(*marker, s=70, facecolors="none", edgecolors="black", linewidths=1.6, depthshade=False)
        ax.text2D(0.01, 0.97, f"Vista {i + 1}", transform=ax.transAxes, fontsize=9, color="#14213d",
                  fontweight="bold")
    cax = fig.add_axes([0.89, 0.12, 0.025, 0.76])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=FIELD_CMAP), cax=cax)
    if vmax <= vmin:
        cb.set_ticks([vmin])
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return _png(fig)


def render_model(nodes, tri, curves, fixed, partial, marks) -> bytes:
    """Geometria com as arestas restritas (vermelho = ancoragem, laranja = restrição parcial)."""
    fig = Figure(figsize=(8.2, 7.2))
    base = np.tile(np.array([0.80, 0.84, 0.89, 1.0]), (len(tri), 1))
    for i, view in enumerate(VIEWS):
        ax = _setup_axes(fig, [0.0, 0.50 - 0.5 * i + 0.01, 1.0, 0.47], nodes, view)
        fc = base.copy()
        fc[:, :3] *= _shade(nodes, tri, view)[:, None]
        ax.add_collection3d(Poly3DCollection(nodes[tri], facecolors=fc, edgecolors="none", linewidths=0))
        for group, col, lw in ((curves, "#4b5563", 0.5), (partial, "#e07b00", 2.2), (fixed, "#c1121f", 2.6)):
            if len(group):
                ax.add_collection3d(Line3DCollection(group, colors=col, linewidths=lw))
        if len(marks):
            ax.scatter(marks[:, 0], marks[:, 1], marks[:, 2], s=26, c="#6a1b9a", depthshade=False)
        ax.text2D(0.01, 0.97, f"Vista {i + 1}", transform=ax.transAxes, fontsize=9, color="#14213d",
                  fontweight="bold")
    return _png(fig)


# ---------------------------------------------------------------- conteúdo

LAYERS = (("mid", "PL — Tensão de membrana (meio da espessura)", "Middle"),
          ("top", "PL+PB top — Membrana + flexão (superfície superior)", "Top"),
          ("bottom", "PL+PB bottom — Membrana + flexão (superfície inferior)", "Bottom"))


def _bc_rows(bc, mesh):
    rows = []
    dofs = ["UX", "UY", "UZ", "RX", "RY", "RZ"]
    for tag, d in sorted(bc["curves"].items()):
        if all(d.get(k) == 0 for k in dofs):
            rows.append(["Aresta", str(tag), "Ancoragem (6 GDL = 0)"])
        else:
            rows.append(["Aresta", str(tag), "; ".join(f"{k} = {g(v)}" for k, v in d.items())
                         + " (demais GDL livres)"])
    for tag, q in sorted(bc["edge_forces"].items()):
        rows.append(["Força total (aresta)", str(tag), f"FX = {g(q[0])} N; FY = {g(q[1])} N; FZ = {g(q[2])} N"])
    groups = {}
    for tag, p in bc["pressure"].items():
        groups.setdefault(float(p), []).append(tag)
    for p, tags in sorted(groups.items()):
        kind = "pressão interna" if p < 0 else ("pressão externa" if p > 0 else "sem pressão")
        rows.append(["Superfícies", ids(tags), f"{g(abs(p))} MPa — {kind}"])
    hydro = bc.get("hydrostatic")
    if hydro:
        edge, z0 = fea.hydrostatic_reference(mesh, bc["curves"])
        h, rho, grav = hydro["height"], hydro["rho"], hydro["g"]
        rows.append(["Hidrostática", f"aresta {edge}",
                     f"Pressão interna em todas as superfícies; Z referência = {g(z0)} mm; "
                     f"altura = {g(h)} mm; nível Z = {g(z0 + h)} mm; "
                     f"ρ = {rho * 1e12:.6g} kg/m³; g = {g(grav)} mm/s²; "
                     f"pressão na referência = {g(rho * grav * h)} MPa. "
                     "Peso de água da tubulação suspenso. Verificar contato com o líquido e orientação das normais."])
    for i, l in enumerate(bc["forces"], 1):
        F = l["F"]
        rows.append(["Força (nó)", str(l["node"]), f"FX = {g(F[0])} N; FY = {g(F[1])} N; FZ = {g(F[2])} N"])
    return rows


def _lta_rows(mesh, res, bc, S_allow, Sp):
    rows = []
    for i, d in enumerate(bc["corrosion"], 1):
        zone = np.flatnonzero(fea.corrosion_zone(mesh.nodes, mesh.triangles, d["node"], d["radius"]))
        if len(zone) == 0:
            continue
        cells = []
        for layer, allow in (("mid", S_allow), ("top", Sp), ("bottom", Sp)):
            vm = float(res.vm_element[layer][zone].max())
            if allow:
                r = vm / allow * 100
                cells.append(f"{vm:.1f} MPa ({r:.0f}% — {'REPROVADO' if r > 100 else 'APROVADO'})")
            else:
                cells.append(f"{vm:.1f} MPa (n/d)")
        rows.append([f"LTA {i}", str(d["node"]), g(d["radius"]), g(d["t"])] + cells)
    return rows


def analysis_inputs(E, nu, thick, gravity, temperature, ignore_dist):
    return [("Módulo de Young E", f"{E:g} MPa"), ("Coeficiente de Poisson", f"{nu:g}"),
            ("Espessura de referência (campo lateral)", f"{thick:g} mm"),
            ("Gravidade", "não considerada" if not gravity else
             f"g = {gravity['g']:g} mm/s², aço ρ = {gravity['rho_steel']:.3g} t/mm³ (−Z)"),
            ("Enchimento com água", "não" if not gravity or "water" not in gravity else
             f"ρ = {gravity['water']['rho']:.3g} t/mm³, diâmetro interno "
             f"{2 * gravity['water']['ri']:g} mm"),
            ("Dilatação térmica", "não considerada" if temperature is None else
             f"α médio = {fea.steel_alpha(temperature) * 1e6:.2f}e-6 /°C "
             f"(20 °C → {temperature:g} °C)"),
            ("Zona desprezada junto às restrições", f"{ignore_dist:g} mm")]


ANALYSIS_NOTES = [
    "Modelo de casca em superfície média com elementos triangulares de deformação constante (CST) "
    "e flexão de placa; tensões de membrana e de flexão avaliadas no meio, topo e base da espessura.",
    "A convergência de malha deve ser verificada pelo engenheiro (refinar e comparar picos de tensão).",
    "Tensões nodais são médias ponderadas por área; picos locais em apoios e quinas são sensíveis à "
    "malha e foram desprezados na zona informada junto às arestas restritas.",
    "O código de aceitação aplicado é o ASME Sec. VIII Div. 2 Part 5 (categorias PM, PL e PL+PB) e as "
    "tabelas S/Sy interpoladas por temperatura; confira com a edição vigente da norma.",
    "Resultados válidos somente para os dados de entrada e as condições de contorno listados neste "
    "relatório. Soldas, juntas, cargas de vento/sísmicas e fadiga não foram consideradas, salvo se "
    "listadas nas entradas.",
]


def vessel_geometry(params, pressure, cap_thickness):
    def pad(t, r):
        return f"{t:g} mm (raio {r:.1f} mm)" if t else "sem reforço"

    return {
        "info": [("Configuração", "Vaso vertical: costado cilíndrico (base solta), tampo semi-elíptico 2:1 "
                                  "no topo, bocal no topo e bocal no costado (a meia altura)"),
                 ("Pressão interna", f"{pressure:g} MPa"),
                 ("Tampas planas dos bocais", f"{cap_thickness:g} mm"),
                 ("Reforço do bocal do topo", pad(params.top_pad_t, params.top_pad_r)),
                 ("Reforço do bocal do costado", pad(params.side_pad_t, params.side_pad_r))],
        "tables": [("Dimensões dos componentes", ["Componente", "DE (mm)", "Espessura (mm)", "Comprimento (mm)"],
                    [["Costado", f"{params.shell_od:g}", f"{params.shell_t:g}", f"{params.shell_len:g}"],
                     ["Tampo 2:1", "—", f"{params.head_t:g}", "—"],
                     ["Bocal do topo", f"{params.top_od:g}", f"{params.top_t:g}", f"{params.top_len:g}"],
                     ["Bocal do costado", f"{params.side_od:g}", f"{params.side_t:g}", f"{params.side_len:g}"]])],
    }


def generate(ctx: dict) -> bytes:
    """Renderiza as figuras 3D e monta o PDF."""
    mesh, res, bc = ctx["mesh"], ctx["res"], ctx["bc"]
    dofs = ["UX", "UY", "UZ", "RX", "RY", "RZ"]
    dnodes = mesh.nodes + res.u[:, :3] * ctx["scale"]

    def segs(tags, nodes):
        arr = [nodes[mesh.curves[t]] for t in tags if t in mesh.curves and len(mesh.curves[t])]
        return np.concatenate(arr) if arr else np.zeros((0, 2, 3))

    fixed_tags = [t for t, d in bc["curves"].items() if all(d.get(k) == 0 for k in dofs)]
    part_tags = [t for t in bc["curves"] if t not in fixed_tags]
    other = [t for t in mesh.curves if t not in bc["curves"]]
    marks = mesh.nodes[[d["node"] for d in bc["corrosion"]]] if bc["corrosion"] else np.zeros((0, 3))
    ctx["model_png"] = render_model(mesh.nodes, mesh.triangles, segs(other, mesh.nodes),
                                    segs(fixed_tags, mesh.nodes), segs(part_tags, mesh.nodes), marks)
    ctx["field_pngs"] = {}
    for layer, _, short in LAYERS:
        vals = res.vm_nodal[layer]
        ctx["field_pngs"][layer] = render_field(
            dnodes, mesh.triangles, vals, f"Von Mises — {short} (MPa)", float(vals.max()),
            marker=dnodes[int(np.argmax(vals))])
    return build_report(ctx)


def build_report(ctx: dict) -> bytes:
    mesh, res, bc = ctx["mesh"], ctx["res"], ctx["bc"]
    S_allow, Sy, Sp = ctx["S_allow"], ctx["Sy"], ctx["Sp"]
    now = datetime.now()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=21 * mm, bottomMargin=19 * mm,
                            title=REPORT_TITLE,
                            author="PIPE-FEA")
    W = A4[0] - 36 * mm
    s = []

    comp = ctx["compliance"]
    lta = _lta_rows(mesh, res, bc, S_allow, Sp)
    bad = any("REPROV" in r[4] for r in comp) or any("REPROVADO" in c for r in lta for c in r[4:])
    known = any("APROV" in r[4] for r in comp)
    verdict = "REPROVADO" if bad else ("APROVADO" if known else "INDETERMINADO")

    # --- Identificação
    s += [Paragraph(REPORT_TITLE, TITLE),
          Paragraph("Relatório de análise de tensões por elementos finitos de casca — verificação ASME "
                    "Sec. VIII Div. 2 Part 5", SUB), Spacer(1, 4 * mm)]
    s.append(kv_table([
        ("Data de emissão", now.strftime("%d/%m/%Y %H:%M")),
        ("Arquivo / origem da geometria", ctx["source"]),
        ("Tipo de geometria", ctx["kind"]),
        ("Unidades", "mm, N, MPa, t (consistentes)"),
        ("Resultado geral da verificação", f'<b>{verdict}</b>'),
    ], (60 * mm, W - 60 * mm)))

    # --- 1. Geometria e malha
    s.append(Paragraph("1. Geometria e malha", H1))
    geo = ctx.get("geometry") or {}
    if geo.get("info"):
        s.append(kv_table(geo["info"], (60 * mm, W - 60 * mm)))
    for title, header, rows in geo.get("tables", []):
        s += [Paragraph(title, H2), table([header] + rows, [W / len(header)] * len(header))]
    s.append(Paragraph("Malha", H2))
    s.append(kv_table([
        ("Nós / elementos (triângulos)", f"{len(mesh.nodes)} / {len(mesh.triangles)}"),
        ("Superfícies / arestas", f"{len(set(mesh.tri_surface.tolist()))} / {len(mesh.curves)}"),
        ("Densidade (divisões da diagonal)", f"{ctx['divisions']}  — refinamentos: {ctx['refinements']}"),
        ("Fator de escala dos comprimentos", g(ctx["len_scale"])),
        ("Diagonal do modelo", f"{g(mesh.bbox_diagonal)} mm"),
        ("Nós duplicados unidos no pré-processamento", str(mesh.welded)),
    ], (60 * mm, W - 60 * mm)))
    s.append(Spacer(1, 3 * mm))
    s.append(Image(io.BytesIO(ctx["model_png"]), width=W, height=W * 7.2 / 8.2))
    s.append(Paragraph("Figura 1 — Geometria malhada. Vermelho: arestas ancoradas (6 GDL = 0); laranja: restrição "
                       "parcial; cinza: demais arestas; roxo: nós com defeito de corrosão.", CENTER))

    # --- 2. Dados da análise
    s.append(Paragraph("2. Dados de entrada da análise", H1))
    th = res.thickness
    th_txt = ", ".join(f"{v:g} mm ({int((np.round(th, 4) == v).sum())} elem.)"
                       for v in np.unique(np.round(th, 4))) if th is not None else g(ctx["thick"])
    s.append(kv_table(ctx["inputs"] + [("Espessuras de casca (elementos)", th_txt)], (60 * mm, W - 60 * mm)))
    s.append(Paragraph("Material e admissíveis (ASME Sec. II-D, interpolados por temperatura)", H2))
    s.append(kv_table([
        ("Temperatura do metal", f"{g(ctx['temp'])} °C"),
        ("S — tensão admissível", f"{g(S_allow, 5)} MPa" if S_allow is not None else "n/d"),
        ("Sy — limite de escoamento", f"{g(Sy, 5)} MPa" if Sy is not None else "n/d"),
        ("Sp = máx(1,5·S; Sy)", f"{g(Sp, 5)} MPa" if Sp is not None else "n/d"),
    ], (60 * mm, W - 60 * mm)))

    # --- 3. Condições de contorno e cargas
    s.append(Paragraph("3. Condições de contorno e cargas", H1))
    bc_rows = _bc_rows(bc, mesh)
    if bc_rows:
        s.append(table([["Tipo", "ID", "Valores"]] + bc_rows, [38 * mm, 28 * mm, W - 66 * mm]))
    else:
        s.append(P("Nenhuma condição atribuída.", BASE))
    s.append(Paragraph("Convenção: pressão negativa no modelo = pressão interna; reportada aqui em módulo.", SMALL))
    if bc["corrosion"]:
        s.append(Paragraph("Defeitos de perda de espessura (LTA)", H2))
        s.append(table([["Defeito", "Nó central", "Raio (mm)", "Espessura corroída (mm)"]] +
                       [[f"LTA {i}", str(d["node"]), g(d["radius"]), g(d["t"])]
                        for i, d in enumerate(bc["corrosion"], 1)], [W / 4] * 4))
    if bc["cracks"]:
        s.append(Paragraph("Trincas", H2))
        s.append(table([["Trinca", "Aresta", "Dimensões"]] +
                       [[f"Trinca {c['id']}", str(c["edge"]), ctx["crack_text"](c)] for c in bc["cracks"]],
                       [28 * mm, 28 * mm, W - 56 * mm]))

    # --- 4. Resultados
    s.append(Paragraph("4. Resultados", H1))
    eq = res.equilibrium
    post = res.post
    fv = lambda v: ", ".join(f"{x:.4g}" for x in v)
    rows = [["Resultante das cargas — força (Fx, Fy, Fz) [N]", fv(eq.force)],
            ["Resultante das cargas — momento (Mx, My, Mz) [N·mm]", fv(eq.moment)],
            ["Movimentos de corpo rígido restringidos", f"{eq.restrained_modes} de 6"],
            ["Reações totais — força [N]", fv(post["reaction_force"])],
            ["Reações totais — momento [N·mm]", fv(post["reaction_moment"])],
            ["Erro de equilíbrio (cargas + reações) — força", f"{post['force_error']:.2e}"],
            ["Erro de equilíbrio (cargas + reações) — momento", f"{post['moment_error']:.2e}"]]
    s.append(Paragraph("Equilíbrio estático", H2))
    s.append(table([["Verificação", "Valor"]] + rows, [95 * mm, W - 95 * mm]))
    s.append(Paragraph("Resumo das tensões e deslocamentos", H2))
    summ = [["Grandeza", "Meio (PL)", "Topo", "Base"]]
    for label, key in (("Von Mises máx. nodal suavizado [MPa]", None),):
        summ.append([label] + [f"{res.vm_nodal[k].max():.1f}" for k in ("mid", "top", "bottom")])
    summ.append(["Von Mises máx. por elemento [MPa]"] + [f"{res.vm_element[k].max():.1f}"
                                                         for k in ("mid", "top", "bottom")])
    summ.append(["Máxima tensão principal nodal [MPa]"] + [f"{res.vm_nodal['p1_' + k].max():.1f}"
                                                           for k in ("mid", "top", "bottom")])
    summ.append(["Deformação equivalente máx. [%]"] + [f"{res.vm_nodal['eps_' + k].max():.4f}"
                                                       for k in ("mid", "top", "bottom")])
    s.append(table(summ, [70 * mm] + [(W - 70 * mm) / 3] * 3))
    s.append(Spacer(1, 2 * mm))
    s.append(kv_table([("Deslocamento resultante máximo", f"{g(res.u_max)} mm"),
                       ("Fator de ampliação das figuras", f"{g(ctx['scale'])}×")], (60 * mm, W - 60 * mm)))
    s.append(Paragraph("Tensões em elementos de casca triangulares de deformação constante (CST); o campo exibido é "
                       "a média nodal ponderada por área. Tensões a menos de "
                       f"{g(ctx['ignore_dist'])} mm de arestas restritas são desprezadas (picos locais de apoio). "
                       "Pico de tensão em apoios e quinas depende da malha: verifique a convergência refinando.",
                       SMALL))

    # --- 5. Code compliance
    s.append(Paragraph("5. Code compliance — ASME Sec. VIII Div. 2 Part 5", H1))
    data = [["Categoria de tensões", "Descrição", "Admissível [MPa]", "Atuante [MPa]", "Utilização", "Status"]]
    for cat, desc, allow, act, st in comp:
        util = f"{act / allow * 100:.0f}%" if allow and act is not None else "—"
        data.append([cat, desc, g(allow, 5) if allow is not None else "n/d",
                     g(act, 5) if act is not None else "n/d", util, status_cell(st)])
    s.append(table(data, [28 * mm, 44 * mm, 26 * mm, 24 * mm, 20 * mm, W - 142 * mm]))
    s.append(Paragraph("P<sub>M</sub>: admissível S; atuante hoop = P·D/(2t). P<sub>L</sub>: admissível máx(1,5·S; Sy); "
                       "atuante Von Mises máx. no meio da espessura. P<sub>L</sub>+P<sub>B</sub>: admissível "
                       "máx(1,5·S; Sy); atuante Von Mises máx. no topo ou na base. Valores nodais suavizados, fora "
                       "da zona desprezada junto às restrições.", SMALL))
    if lta:
        s.append(Paragraph("Defeitos de perda de espessura (LTA) — Von Mises máx. na zona, admissível S (PL) e Sp "
                           "(PL+PB)", H2))
        s.append(table([["Defeito", "Nó", "Raio [mm]", "t corroída [mm]", "Meio (PL)", "Topo", "Base"]] +
                       [[r[0], r[1], r[2], r[3]] + [P(c) for c in r[4:]] for r in lta],
                       [16 * mm, 14 * mm, 18 * mm, 24 * mm] + [(W - 72 * mm) / 3] * 3))
        s.append(Paragraph("Critério de nível 3 (análise por elementos finitos): tensão calculada na zona corroída "
                           "comparada ao admissível da categoria correspondente.", SMALL))
    if bc["cracks"] and ctx["crack_rows"]:
        s.append(Paragraph("Tensões normais à trinca (que abrem a trinca) no cordão de solda selecionado", H2))
        s.append(table([["Trinca", "Aresta", "Dimensões", "PL [MPa]", "PL+PB top [MPa]", "PL+PB bottom [MPa]"]]
                       + ctx["crack_rows"], [20 * mm, 16 * mm, W - 116 * mm, 20 * mm, 30 * mm, 30 * mm]))

    # --- 6. Mapas de Von Mises
    for n, (layer, desc, short) in enumerate(LAYERS, 1):
        s.append(PageBreak())
        s.append(Paragraph(f"6.{n} Tensão de Von Mises — {short}", H1))
        vals = res.vm_nodal[layer]
        k = int(np.argmax(vals))
        allow = S_allow if layer == "mid" else Sp
        util = f" — utilização {vals[k] / allow * 100:.0f}% do admissível ({g(allow, 5)} MPa)" if allow else ""
        s.append(Paragraph(f"{desc}. Máximo nodal = <b>{vals[k]:.1f} MPa</b>{util}. Posição do máximo "
                           f"(X, Y, Z) = ({mesh.nodes[k][0]:.1f}; {mesh.nodes[k][1]:.1f}; {mesh.nodes[k][2]:.1f}) mm "
                           f"(círculo preto). Escala de cores de {vals.min():.1f} a {vals.max():.1f} MPa; geometria deformada "
                           f"(ampliação {g(ctx['scale'])}×).", BASE))
        png = ctx["field_pngs"][layer]
        s.append(Spacer(1, 2 * mm))
        s.append(Image(io.BytesIO(png), width=W, height=W * 9.6 / 8.2))
        s.append(Paragraph(f"Figura {n + 1} — Von Mises ({short}), duas vistas isométricas opostas.", CENTER))

    # --- 7. Verificação
    s.append(PageBreak())
    s.append(Paragraph("7. Premissas, limitações e verificação do engenheiro", H1))
    for t in ctx["notes"]:
        s.append(Paragraph("• " + t, BASE))
        s.append(Spacer(1, 1.2 * mm))
    s.append(Spacer(1, 6 * mm))
    sig = Table([[P("Elaborado por:"), P("Verificado por:"), P("Aprovado por:")],
                 [P("Nome / CREA:<br/><br/><br/>Assinatura:<br/><br/>Data:"),
                  P("Nome / CREA:<br/><br/><br/>Assinatura:<br/><br/>Data:"),
                  P("Nome / CREA:<br/><br/><br/>Assinatura:<br/><br/>Data:")]],
                colWidths=[W / 3] * 3)
    sig.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, GRID), ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
                             ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    s.append(KeepTogether(sig))

    def canvas_maker(*a, **k):
        c = _Numbered(*a, **k)
        c._title = ctx["source"][:60]
        return c

    doc.build(s, canvasmaker=canvas_maker)
    return buf.getvalue()
