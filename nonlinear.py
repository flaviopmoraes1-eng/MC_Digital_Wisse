"""Experimental CalculiX shell analysis with fixed-direction nodal loads."""
from dataclasses import dataclass
from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np

import fea


@dataclass
class NonlinearResult:
    displacements: np.ndarray
    stress_fields: dict[str, np.ndarray]
    input_deck: bytes
    output: bytes
    status: str


def solver_path():
    configured = os.environ.get("PIPE_FEA_CCX")
    if configured:
        path = Path(configured)
        if not path.is_file():
            raise ValueError(f"Executável CalculiX não encontrado: {path}")
        return str(path.resolve())
    found = shutil.which("ccx") or shutil.which("ccx.exe")
    if found:
        return found
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        candidate = Path(os.environ["LOCALAPPDATA"]) / "PIPE-FEA" / "CalculiX-2.22" / "bin" / "ccx.exe"
        if candidate.is_file():
            return str(candidate)
    raise ValueError("CalculiX não encontrado. Instale o ccx e configure PIPE_FEA_CCX com o caminho do executável.")


def input_deck(mesh, mat, bc, gravity, temperature):
    mat.validate()
    if temperature is not None and abs(fea.thermal_strain(temperature)) > 1e-14:
        raise ValueError("Dilatação térmica não está disponível na análise não linear experimental.")
    if bc.get("cracks"):
        raise ValueError("Trincas não são suportadas na análise não linear experimental.")
    nodes = mesh.nodes
    tri = fea.oriented_triangles(nodes, mesh.triangles)
    thickness = fea.element_thickness(mesh, mat.t, bc.get("corrosion", []))
    area = np.linalg.norm(np.cross(nodes[tri[:, 1]] - nodes[tri[:, 0]],
                                   nodes[tri[:, 2]] - nodes[tri[:, 0]]), axis=1) / 2
    loads = fea.pressure_loads(nodes, tri, mesh.tri_surface, bc.get("pressure", {}))
    hydro = bc.get("hydrostatic")
    if hydro:
        h = hydro["height"]
        if not np.isfinite(h) or h < 0:
            raise ValueError("A altura hidrostática deve ser finita e não negativa.")
        _, z = fea.hydrostatic_reference(mesh, bc.get("curves", {}))
        loads += fea.hydrostatic_loads(nodes, tri, z + h, hydro["rho"], hydro["g"])
    if gravity:
        loads += fea.gravity_loads(nodes, tri, area, thickness, {**gravity, "t_nominal": mat.t})
    loads += fea.edge_loads(nodes, mesh.curves, bc.get("edge_forces", {}))
    for load in bc.get("forces", []):
        i = int(load["node"])
        if not 0 <= i < len(nodes):
            raise ValueError("Nó de força inválido.")
        loads[6 * i:6 * i + 3] += load["F"]
    prescribed = fea.prescribed_dofs(mesh, bc.get("curves", {}))
    if not prescribed:
        raise ValueError("Defina ao menos uma restrição em aresta.")
    if not np.any(loads) and not any(prescribed.values()):
        raise ValueError("Defina uma carga ou deslocamento imposto não nulo.")
    eq = fea.check_equilibrium(nodes, loads, prescribed)
    if not eq.ok:
        raise ValueError("Restrições insuficientes: " + "; ".join(eq.messages))

    lines = ["*HEADING", "PIPE-FEA experimental - geometric nonlinearity",
             "*NODE,NSET=NALL"]
    lines.extend(f"{i + 1},{x:.12g},{y:.12g},{z:.12g}"
                 for i, (x, y, z) in enumerate(nodes))
    lines.append("*ELEMENT,TYPE=S3,ELSET=ALLSHELLS")
    lines.extend(f"{i + 1},{a + 1},{b + 1},{c + 1}"
                 for i, (a, b, c) in enumerate(tri))
    lines.extend(["*MATERIAL,NAME=MAT", "*ELASTIC", f"{mat.E:.12g},{mat.nu:.12g}"])
    for j, value in enumerate(np.unique(thickness)):
        lines.append(f"*ELSET,ELSET=THICK{j}")
        ids = np.flatnonzero(thickness == value) + 1
        lines.extend(",".join(map(str, ids[i:i + 16])) for i in range(0, len(ids), 16))
        lines.extend([f"*SHELL SECTION,ELSET=THICK{j},MATERIAL=MAT", f"{value:.12g}"])
    lines.extend(["*STEP,NLGEOM,INC=1000", "*STATIC", "0.1,1.0,1e-6,0.1", "*BOUNDARY"])
    lines.extend(f"{dof // 6 + 1},{dof % 6 + 1},{dof % 6 + 1},{value:.12g}"
                 for dof, value in sorted(prescribed.items()))
    lines.append("*CLOAD")
    nodal = loads.reshape(-1, 6)
    for node, dof in np.argwhere(nodal[:, :3] != 0):
        lines.append(f"{node + 1},{dof + 1},{nodal[node, dof]:.12g}")
    lines.extend(["*NODE PRINT,NSET=NALL,FREQUENCY=1000000", "U",
                  "*EL PRINT,ELSET=ALLSHELLS,FREQUENCY=1000000",
                  "S", "*END STEP"])
    return ("\n".join(lines) + "\n").encode("ascii")


def parse_displacements(text, node_count):
    blocks = list(re.finditer(
        r"displacements\s+\(vx,vy,vz\)\s+for set\s+NALL\s+and time\s+([+\-0-9.Ee]+)\s*\n",
        text, re.IGNORECASE))
    if not blocks or abs(float(blocks[-1].group(1)) - 1.0) > 1e-6:
        raise ValueError("CalculiX não forneceu deslocamentos no fim do carregamento.")
    rows = text[blocks[-1].end():].splitlines()
    u = np.full((node_count, 3), np.nan)
    for row in rows:
        fields = row.split()
        if len(fields) != 4:
            if np.isfinite(u).all():
                break
            continue
        try:
            i = int(fields[0]) - 1
            xyz = [float(v) for v in fields[1:]]
        except ValueError:
            if np.isfinite(u).all():
                break
            continue
        if i < 0 or i >= node_count or np.isfinite(u[i]).any():
            raise ValueError("Dados de deslocamento do CalculiX inconsistentes.")
        u[i] = xyz
    if not np.isfinite(u).all():
        raise ValueError("Dados de deslocamento do CalculiX incompletos.")
    return u


def parse_stresses(text, element_count):
    blocks = list(re.finditer(
        r"stresses\s+\(elem,\s*integ\.pnt\.,\s*sxx,syy,szz,sxy,sxz,syz\)"
        r"\s+for set\s+ALLSHELLS\s+and time\s+([+\-0-9.Ee]+)\s*\n",
        text, re.IGNORECASE))
    if not blocks or abs(float(blocks[-1].group(1)) - 1.0) > 1e-6:
        raise ValueError("CalculiX não forneceu tensões no fim do carregamento.")
    stress = np.full((element_count, 2, 6), np.nan)
    started = False
    for row in text[blocks[-1].end():].splitlines():
        fields = row.split()
        if not fields:
            if started:
                break
            continue
        try:
            element, point = int(fields[0]) - 1, int(fields[1]) - 1
            values = [float(v) for v in fields[2:8]]
        except (ValueError, IndexError):
            raise ValueError("Dados de tensão do CalculiX inconsistentes.") from None
        if (len(values) != 6 or not 0 <= element < element_count or not 0 <= point < 2
                or np.isfinite(stress[element, point]).any() or not np.isfinite(values).all()):
            raise ValueError("Dados de tensão do CalculiX inconsistentes.")
        stress[element, point] = values
        started = True
    if not np.isfinite(stress).all():
        raise ValueError("Dados de tensão do CalculiX incompletos.")
    return stress


def nodal_stress_fields(mesh, stress):
    sxx, syy, szz, sxy, sxz, syz = np.moveaxis(stress, -1, 0)
    vm = np.sqrt(((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2) / 2
                 + 3 * (sxy**2 + sxz**2 + syz**2))
    tensor = np.empty(stress.shape[:2] + (3, 3))
    tensor[..., 0, 0], tensor[..., 1, 1], tensor[..., 2, 2] = sxx, syy, szz
    tensor[..., 0, 1] = tensor[..., 1, 0] = sxy
    tensor[..., 0, 2] = tensor[..., 2, 0] = sxz
    tensor[..., 1, 2] = tensor[..., 2, 1] = syz
    principal = np.linalg.eigvalsh(tensor)
    tri = mesh.triangles
    nodes = mesh.nodes
    area = np.linalg.norm(np.cross(nodes[tri[:, 1]] - nodes[tri[:, 0]],
                                   nodes[tri[:, 2]] - nodes[tri[:, 0]]), axis=1) / 2
    if not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError("Malha contém elemento de área inválida para o mapa de tensões.")
    if len(np.unique(tri)) != len(nodes):
        raise ValueError("Malha contém nós sem elementos para o mapa de tensões.")
    return {
        "vm": fea._nodal_average(len(nodes), tri, area, vm.max(axis=1)),
        "p1": fea._nodal_average(len(nodes), tri, area, principal[..., 2].max(axis=1)),
        "p2": fea._nodal_average(len(nodes), tri, area, principal[..., 0].min(axis=1)),
    }


def run_analysis(mesh, mat, bc, gravity, temperature, executable=None):
    exe = executable or solver_path()
    deck = input_deck(mesh, mat, bc, gravity, temperature)
    try:
        with tempfile.TemporaryDirectory(prefix="pipefea_ccx_") as folder:
            work = Path(folder)
            (work / "model.inp").write_bytes(deck)
            completed = subprocess.run([exe, "model"], cwd=folder, capture_output=True,
                                       text=True, errors="replace", timeout=600)
            log = completed.stdout + "\n" + completed.stderr
            if completed.returncode != 0 or "Job finished" not in log or "*ERROR" in log.upper():
                raise ValueError(f"CalculiX falhou ou não convergiu (código {completed.returncode}):\n"
                                 + log[-4000:])
            dat, sta = work / "model.dat", work / "model.sta"
            if not dat.is_file() or not sta.is_file():
                raise ValueError("CalculiX não produziu os arquivos de resultados esperados.")
            output = dat.read_bytes()
            status = sta.read_text(encoding="utf-8", errors="replace")
            text = output.decode("ascii", errors="replace")
            return NonlinearResult(
                parse_displacements(text, len(mesh.nodes)),
                nodal_stress_fields(mesh, parse_stresses(text, len(mesh.triangles))),
                deck, output, status)
    except subprocess.TimeoutExpired as exc:
        raise ValueError("CalculiX excedeu o limite de 10 minutos; a análise foi interrompida.") from exc
    except OSError as exc:
        raise ValueError(f"Não foi possível executar o CalculiX: {exc}") from exc
