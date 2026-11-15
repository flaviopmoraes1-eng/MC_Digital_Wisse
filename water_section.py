"""Section contours for the visual liquid level, independent of the FEA loads."""
import numpy as np


def section_contours(nodes, triangles, level):
    """Return closed XY contours cut by a horizontal plane, or raise on ambiguous cuts."""
    heights = nodes[:, 2]
    if not np.isfinite(level):
        raise ValueError("O nível do líquido não é finito.")
    if level < heights.min() or level >= heights.max():
        raise ValueError("O nível está fora da seção da malha.")

    sides = heights > level
    crossing = sides[triangles]
    cut = triangles[np.any(crossing, axis=1) & ~np.all(crossing, axis=1)]
    if not len(cut):
        raise ValueError("Não há contorno da malha no nível informado.")

    adjacency = {}
    positions = {}
    for tri in cut:
        edges = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = int(tri[i]), int(tri[j])
            if sides[a] == sides[b]:
                continue
            edge = tuple(sorted((a, b)))
            if edge not in positions:
                start, end = nodes[a], nodes[b]
                fraction = (level - start[2]) / (end[2] - start[2])
                positions[edge] = start[:2] + fraction * (end[:2] - start[:2])
            edges.append(edge)
        if len(edges) != 2 or edges[0] == edges[1]:
            raise ValueError("A seção da malha é ambígua.")
        a, b = edges
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    if any(len(neighbors) != 2 or neighbors[0] == neighbors[1]
           for neighbors in adjacency.values()):
        raise ValueError("O contorno da malha no nível não é fechado ou é ambíguo.")

    unused = set(adjacency)
    contours = []
    tolerance = max(float(np.ptp(nodes, axis=0).max()) * 1e-12, 1e-12)
    while unused:
        start = next(iter(unused))
        current, previous = start, None
        points = []
        while True:
            if current not in unused:
                raise ValueError("O contorno da malha no nível é ambíguo.")
            unused.remove(current)
            point = positions[current]
            if not points or np.linalg.norm(point - points[-1]) > tolerance:
                points.append(point)
            neighbors = adjacency[current]
            following = neighbors[0] if neighbors[0] != previous else neighbors[1]
            previous, current = current, following
            if current == start:
                break
        if len(points) > 1 and np.linalg.norm(points[0] - points[-1]) <= tolerance:
            points.pop()
        if len(points) < 3:
            raise ValueError("O contorno da malha no nível é degenerado.")
        polygon = np.asarray(points)
        area = abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
                   - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))) / 2
        if area <= tolerance ** 2:
            raise ValueError("O contorno da malha no nível é degenerado.")
        contours.append(polygon.tolist())
    return contours
