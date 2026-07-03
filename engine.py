"""
engine.py — Site rasterization + CP-SAT facility-layout engine for a biogas
/ CBG plant, driven by process flow rather than pure area-packing.

Pipeline:
  1. rasterize(): boundary polygon (with holes/obstacles) -> boolean buildable
     grid, after applying a clearance/setback buffer. Individual obstacles can
     carry an extra hazard-specific buffer on top of the base setback (used
     when an already-existing piece of equipment from a DXF import has a
     safety-clearance requirement against whatever gets placed near it).
  2. decompose_blocked(): the *unbuildable* cells are merged into a small set
     of axis-aligned rectangles for CP-SAT.
  3. generate_layouts(): the scientific part. This is a Facility Layout
     Problem (FLP), not a bin-packing problem — the objective is driven by
     how gas/material actually flows through the plant, not by how densely
     rectangles pack. Concretely, for every process-flow edge (e.g.
     "digester -> biogas holder") the model creates a Manhattan-distance term
     between every instance of the two connected equipment types (weighted by
     how flow-critical that connection is — a raw-gas transfer line matters
     far more than "security hut near gate"), and for every safety rule (e.g.
     "flare stack >= 15 m from digester") the model adds a HARD rectangle-
     separation constraint that a layout can never violate.

     Because there's no single "correct" way to trade off total pipe run vs.
     footprint vs. worst-case single run, generate_layouts() solves the model
     several times with genuinely different secondary objectives (after first
     locking in the maximum achievable placed area/count so no strategy
     sacrifices fitting equipment just to shorten pipes) and returns each as
     a distinct, named, comparable layout.

All internal geometry is done in *grid cells*; callers convert to/from
real-world metres via `cell_size`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box as shapely_box
from shapely.ops import unary_union
from shapely.prepared import prep
from ortools.sat.python import cp_model


# ──────────────────────────────────────────────────────────────────────────
# Rasterization
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class Grid:
    mask: np.ndarray          # bool[h, w], True = buildable
    cell_size: float          # metres per cell
    origin_x: float           # world-space x of grid cell (0,0)'s min corner
    origin_y: float
    width: int                # cols
    height: int                # rows


def rasterize(
    boundary: Polygon,
    obstacles: List[Polygon],
    cell_size: float,
    setback: float,
    obstacle_extra_buffers: Optional[List[float]] = None,
) -> Grid:
    """Conservative rasterization: a cell is buildable only if it is FULLY
    inside (boundary shrunk by setback) AND FULLY outside every obstacle,
    each grown by (setback + its own extra hazard buffer if any). The extra
    per-obstacle buffer lets an existing hazardous piece of equipment (e.g.
    an already-built flare stack found in the imported DXF) demand more
    clearance than the generic setback, without over-buffering every
    ordinary obstacle.
    """
    shrunk = boundary.buffer(-setback, join_style=2) if setback > 0 else boundary
    if shrunk.is_empty:
        raise ValueError("Setback/clearance is too large — no buildable area remains.")

    if obstacle_extra_buffers is None:
        obstacle_extra_buffers = [0.0] * len(obstacles)

    grown_obstacles = []
    for o, extra in zip(obstacles, obstacle_extra_buffers):
        if o.is_empty:
            continue
        buf = setback + max(0.0, extra)
        grown_obstacles.append(o.buffer(buf, join_style=2))
    blocked_union = unary_union(grown_obstacles) if grown_obstacles else None

    minx, miny, maxx, maxy = boundary.bounds
    width = max(1, int(np.ceil((maxx - minx) / cell_size)))
    height = max(1, int(np.ceil((maxy - miny) / cell_size)))

    prepared_shrunk = prep(shrunk)
    prepared_blocked = prep(blocked_union) if blocked_union is not None and not blocked_union.is_empty else None

    mask = np.zeros((height, width), dtype=bool)
    for row in range(height):
        cy0 = miny + row * cell_size
        cy1 = cy0 + cell_size
        for col in range(width):
            cx0 = minx + col * cell_size
            cx1 = cx0 + cell_size
            cell = shapely_box(cx0, cy0, cx1, cy1)
            if not prepared_shrunk.covers(cell):
                continue
            if prepared_blocked is not None and prepared_blocked.intersects(cell):
                continue
            mask[row, col] = True

    return Grid(mask=mask, cell_size=cell_size, origin_x=minx, origin_y=miny, width=width, height=height)


# ──────────────────────────────────────────────────────────────────────────
# Blocked-area rectangle decomposition
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int


def decompose_blocked(mask: np.ndarray) -> List[Rect]:
    """Greedy row-merge decomposition of the *blocked* (False) cells of `mask`
    into a small set of non-overlapping rectangles.
    """
    h, w = mask.shape
    blocked = ~mask

    row_spans: List[List[Tuple[int, int]]] = []
    for r in range(h):
        spans = []
        c = 0
        while c < w:
            if blocked[r, c]:
                c0 = c
                while c < w and blocked[r, c]:
                    c += 1
                spans.append((c0, c))
            else:
                c += 1
        row_spans.append(spans)

    rects: List[Rect] = []
    active: dict[Tuple[int, int], int] = {}

    def flush(span, start_row, end_row):
        c0, c1 = span
        rects.append(Rect(x=c0, y=start_row, w=c1 - c0, h=end_row - start_row))

    for r in range(h):
        current_spans = set(row_spans[r])
        for span in list(active.keys()):
            if span not in current_spans:
                flush(span, active[span], r)
                del active[span]
        for span in current_spans:
            if span not in active:
                active[span] = r
    for span, start_row in active.items():
        flush(span, start_row, h)

    return rects


# ──────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class BuildingSpec:
    id: str             # unique key for this spec (used for placement bookkeeping)
    name: str
    width: float        # metres
    height: float       # metres
    count: int = 1
    allow_rotate: bool = True
    color: str = "#4A90D9"
    flow_stage_id: Optional[str] = None   # biogas_config stage id for flow/safety matching,
                                            # defaults to `id` if not given (set this explicitly
                                            # when `id` had to be made unique, e.g. two rows both
                                            # representing "Digester")

    def stage_key(self) -> str:
        return self.flow_stage_id or self.id


@dataclass
class FlowEdge:
    from_id: str
    to_id: str
    weight: float


@dataclass
class SafetyRule:
    from_id: str
    to_id: str
    min_distance: float   # metres
    reason: str = ""


@dataclass
class Placement:
    spec_id: str
    instance: int
    name: str
    color: str
    x: float             # metres, world space (already includes grid origin)
    y: float
    w: float              # placed width (post-rotation)
    h: float               # placed height
    stage_id: Optional[str] = None   # biogas_config stage id, for flow/safety matching


@dataclass
class LayoutResult:
    name: str
    description: str
    placements: List[Placement]
    unplaced: List[Tuple[str, int]]   # (spec_id, count_unplaced)
    status: str
    metrics: Dict[str, float]
    grid: Grid


# ──────────────────────────────────────────────────────────────────────────
# CP-SAT model builder (shared by every strategy)
# ──────────────────────────────────────────────────────────────────────────

class _ModelBuild:
    """Builds the hard-constraint part of the model once; each strategy adds
    its own secondary objective on top and solves a fresh copy of it (CP-SAT
    models aren't reusable across independent Solve() calls with different
    objectives, so we rebuild per strategy — cheap relative to solve time).
    """

    def __init__(self, grid: Grid, specs: List[BuildingSpec], spacing: float,
                 safety_rules: List[SafetyRule], locked: Optional[List[Placement]]):
        self.grid = grid
        self.specs = specs
        self.cell = grid.cell_size
        self.spacing_cells = max(0, int(round(spacing / self.cell)))
        self.model = cp_model.CpModel()
        self.locked = locked or []

        W, H = grid.width, grid.height
        self.W, self.H = W, H

        x_intervals, y_intervals = [], []

        blocked_rects = decompose_blocked(grid.mask)
        for br in blocked_rects:
            xi = self.model.NewIntervalVar(br.x, br.w, br.x + br.w, f"blockedX_{br.x}_{br.y}")
            yi = self.model.NewIntervalVar(br.y, br.h, br.y + br.h, f"blockedY_{br.x}_{br.y}")
            x_intervals.append(xi)
            y_intervals.append(yi)

        # Locked (existing/imported, must-keep) equipment -> fixed intervals.
        self.locked_boxes = []   # (stage_id, x0, y0, x1, y1) in cells, for safety-rule checks
        for lp in self.locked:
            lx0 = int(round((lp.x - grid.origin_x) / self.cell))
            ly0 = int(round((lp.y - grid.origin_y) / self.cell))
            lw = max(1, int(round(lp.w / self.cell)))
            lh = max(1, int(round(lp.h / self.cell)))
            xi = self.model.NewIntervalVar(lx0, lw, lx0 + lw, f"lockedX_{lp.spec_id}_{lp.instance}")
            yi = self.model.NewIntervalVar(ly0, lh, ly0 + lh, f"lockedY_{lp.spec_id}_{lp.instance}")
            x_intervals.append(xi)
            y_intervals.append(yi)

        # Movable building instances.
        self.presence: Dict[Tuple[str, int], cp_model.IntVar] = {}
        self.x_start: Dict[Tuple[str, int], cp_model.IntVar] = {}
        self.y_start: Dict[Tuple[str, int], cp_model.IntVar] = {}
        self.x_end: Dict[Tuple[str, int], cp_model.IntVar] = {}
        self.y_end: Dict[Tuple[str, int], cp_model.IntVar] = {}
        self.bw: Dict[Tuple[str, int], cp_model.IntVar] = {}
        self.bh: Dict[Tuple[str, int], cp_model.IntVar] = {}
        self.spec_by_key: Dict[Tuple[str, int], BuildingSpec] = {}

        for spec in specs:
            w_cells = max(1, int(round(spec.width / self.cell)))
            h_cells = max(1, int(round(spec.height / self.cell)))
            for inst in range(spec.count):
                key = (spec.id, inst)
                presence = self.model.NewBoolVar(f"present_{spec.id}_{inst}")

                if spec.allow_rotate and w_cells != h_cells:
                    rotated = self.model.NewBoolVar(f"rot_{spec.id}_{inst}")
                    bw = self.model.NewIntVar(min(w_cells, h_cells), max(w_cells, h_cells), f"bw_{spec.id}_{inst}")
                    bh = self.model.NewIntVar(min(w_cells, h_cells), max(w_cells, h_cells), f"bh_{spec.id}_{inst}")
                    self.model.Add(bw == w_cells).OnlyEnforceIf(rotated.Not())
                    self.model.Add(bh == h_cells).OnlyEnforceIf(rotated.Not())
                    self.model.Add(bw == h_cells).OnlyEnforceIf(rotated)
                    self.model.Add(bh == w_cells).OnlyEnforceIf(rotated)
                else:
                    bw = self.model.NewConstant(w_cells)
                    bh = self.model.NewConstant(h_cells)

                max_dim = max(w_cells, h_cells)
                max_x_start = max(0, W - 1)
                max_y_start = max(0, H - 1)
                x_start = self.model.NewIntVar(0, max_x_start, f"x_{spec.id}_{inst}")
                y_start = self.model.NewIntVar(0, max_y_start, f"y_{spec.id}_{inst}")
                x_end = self.model.NewIntVar(0, max_x_start + max_dim, f"xe_{spec.id}_{inst}")
                y_end = self.model.NewIntVar(0, max_y_start + max_dim, f"ye_{spec.id}_{inst}")
                self.model.Add(x_end == x_start + bw)
                self.model.Add(y_end == y_start + bh)
                self.model.Add(x_end <= W).OnlyEnforceIf(presence)
                self.model.Add(y_end <= H).OnlyEnforceIf(presence)

                # Collision-purpose intervals: symmetric spacing inflation
                # (same rationale as before — fixed objects are left
                # un-inflated, movable objects carry the full padding so any
                # movable-vs-anything gap is guaranteed >= spacing).
                x_start_i = self.model.NewIntVar(-self.spacing_cells, max_x_start, f"xsi0_{spec.id}_{inst}")
                y_start_i = self.model.NewIntVar(-self.spacing_cells, max_y_start, f"ysi0_{spec.id}_{inst}")
                self.model.Add(x_start_i == x_start - self.spacing_cells)
                self.model.Add(y_start_i == y_start - self.spacing_cells)
                x_size_i = self.model.NewIntVar(0, max_dim + 2 * self.spacing_cells, f"xsi_{spec.id}_{inst}")
                y_size_i = self.model.NewIntVar(0, max_dim + 2 * self.spacing_cells, f"ysi_{spec.id}_{inst}")
                self.model.Add(x_size_i == bw + 2 * self.spacing_cells)
                self.model.Add(y_size_i == bh + 2 * self.spacing_cells)
                x_end_i = self.model.NewIntVar(-self.spacing_cells, max_x_start + max_dim + self.spacing_cells, f"xei_{spec.id}_{inst}")
                y_end_i = self.model.NewIntVar(-self.spacing_cells, max_y_start + max_dim + self.spacing_cells, f"yei_{spec.id}_{inst}")
                self.model.Add(x_end_i == x_start_i + x_size_i)
                self.model.Add(y_end_i == y_start_i + y_size_i)

                xi = self.model.NewOptionalIntervalVar(x_start_i, x_size_i, x_end_i, presence, f"bX_{spec.id}_{inst}")
                yi = self.model.NewOptionalIntervalVar(y_start_i, y_size_i, y_end_i, presence, f"bY_{spec.id}_{inst}")
                x_intervals.append(xi)
                y_intervals.append(yi)

                self.presence[key] = presence
                self.x_start[key] = x_start
                self.y_start[key] = y_start
                self.x_end[key] = x_end
                self.y_end[key] = y_end
                self.bw[key] = bw
                self.bh[key] = bh
                self.spec_by_key[key] = spec

        self.model.AddNoOverlap2D(x_intervals, y_intervals)

        # Hard safety-clearance constraints between movable instances of
        # different (or the same) hazard-linked stage types.
        keys = list(self.presence.keys())
        for rule in safety_rules:
            d_cells = int(np.ceil(rule.min_distance / self.cell))
            if d_cells <= 0:
                continue
            for ka in keys:
                sa = self.spec_by_key[ka]
                if sa.stage_key() != rule.from_id:
                    continue
                for kb in keys:
                    sb = self.spec_by_key[kb]
                    if sb.stage_key() != rule.to_id or ka == kb:
                        continue
                    self._add_pair_clearance(ka, kb, d_cells, both_movable=True)

            # Also enforce against fixed/locked equipment carrying a matching stage_id.
            for lp in self.locked:
                lock_stage = getattr(lp, "stage_id", None)
                if lock_stage is None:
                    continue
                lx0 = int(round((lp.x - grid.origin_x) / self.cell))
                ly0 = int(round((lp.y - grid.origin_y) / self.cell))
                lw = max(1, int(round(lp.w / self.cell)))
                lh = max(1, int(round(lp.h / self.cell)))
                pairs = []
                if lock_stage == rule.from_id:
                    pairs = [(k, "to") for k in keys if self.spec_by_key[k].stage_key() == rule.to_id]
                elif lock_stage == rule.to_id:
                    pairs = [(k, "from") for k in keys if self.spec_by_key[k].stage_key() == rule.from_id]
                for k, _ in pairs:
                    self._add_fixed_pair_clearance(k, (lx0, ly0, lx0 + lw, ly0 + lh), d_cells)

    def _add_pair_clearance(self, ka, kb, d_cells, both_movable):
        m = self.model
        xA0, yA0, xA1, yA1 = self.x_start[ka], self.y_start[ka], self.x_end[ka], self.y_end[ka]
        xB0, yB0, xB1, yB1 = self.x_start[kb], self.y_start[kb], self.x_end[kb], self.y_end[kb]
        pA, pB = self.presence[ka], self.presence[kb]

        sep_r = m.NewBoolVar(f"sepR_{ka}_{kb}")
        sep_l = m.NewBoolVar(f"sepL_{ka}_{kb}")
        sep_u = m.NewBoolVar(f"sepU_{ka}_{kb}")
        sep_d = m.NewBoolVar(f"sepD_{ka}_{kb}")
        m.Add(xB0 >= xA1 + d_cells).OnlyEnforceIf(sep_r)
        m.Add(xA0 >= xB1 + d_cells).OnlyEnforceIf(sep_l)
        m.Add(yB0 >= yA1 + d_cells).OnlyEnforceIf(sep_u)
        m.Add(yA0 >= yB1 + d_cells).OnlyEnforceIf(sep_d)
        m.AddBoolOr([sep_r, sep_l, sep_u, sep_d]).OnlyEnforceIf([pA, pB])

    def _add_fixed_pair_clearance(self, ka, fixed_box, d_cells):
        m = self.model
        xA0, yA0, xA1, yA1 = self.x_start[ka], self.y_start[ka], self.x_end[ka], self.y_end[ka]
        fx0, fy0, fx1, fy1 = fixed_box
        pA = self.presence[ka]

        sep_r = m.NewBoolVar(f"sepFR_{ka}")
        sep_l = m.NewBoolVar(f"sepFL_{ka}")
        sep_u = m.NewBoolVar(f"sepFU_{ka}")
        sep_d = m.NewBoolVar(f"sepFD_{ka}")
        m.Add(fx0 >= xA1 + d_cells).OnlyEnforceIf(sep_r)
        m.Add(xA0 >= fx1 + d_cells).OnlyEnforceIf(sep_l)
        m.Add(fy0 >= yA1 + d_cells).OnlyEnforceIf(sep_u)
        m.Add(yA0 >= fy1 + d_cells).OnlyEnforceIf(sep_d)
        m.AddBoolOr([sep_r, sep_l, sep_u, sep_d]).OnlyEnforceIf(pA)

    def area_expr(self):
        return sum(
            self.presence[k] * int(round(self.spec_by_key[k].width * self.spec_by_key[k].height * 100))
            for k in self.presence
        )

    def center2(self, key):
        """Doubled center coordinates (2*center), kept integer exactly."""
        cx2 = self.x_start[key] * 2 + self.bw[key]
        cy2 = self.y_start[key] * 2 + self.bh[key]
        return cx2, cy2

    def flow_terms(self, flow_edges: List[FlowEdge]):
        """Returns list of (weight, term_var, term_max_bound) for every
        instance-pair implied by each flow edge. term_var == 0 whenever
        either endpoint instance isn't present.
        """
        m = self.model
        terms = []
        keys_by_stage: Dict[str, List[Tuple[str, int]]] = {}
        for k in self.presence:
            keys_by_stage.setdefault(self.spec_by_key[k].stage_key(), []).append(k)

        max_bound = 2 * (self.W + self.H)  # generous bound on doubled Manhattan distance
        for edge in flow_edges:
            from_keys = keys_by_stage.get(edge.from_id, [])
            to_keys = keys_by_stage.get(edge.to_id, [])
            for ka in from_keys:
                for kb in to_keys:
                    if ka == kb:
                        continue
                    cxA, cyA = self.center2(ka)
                    cxB, cyB = self.center2(kb)
                    dx = m.NewIntVar(-max_bound, max_bound, f"dx_{ka}_{kb}")
                    dy = m.NewIntVar(-max_bound, max_bound, f"dy_{ka}_{kb}")
                    m.Add(dx == cxA - cxB)
                    m.Add(dy == cyA - cyB)
                    adx = m.NewIntVar(0, max_bound, f"adx_{ka}_{kb}")
                    ady = m.NewIntVar(0, max_bound, f"ady_{ka}_{kb}")
                    m.AddAbsEquality(adx, dx)
                    m.AddAbsEquality(ady, dy)
                    dist = m.NewIntVar(0, 2 * max_bound, f"dist_{ka}_{kb}")
                    m.Add(dist == adx + ady)

                    pA, pB = self.presence[ka], self.presence[kb]
                    term = m.NewIntVar(0, 2 * max_bound, f"term_{ka}_{kb}")
                    m.Add(term == dist).OnlyEnforceIf([pA, pB])
                    m.Add(term == 0).OnlyEnforceIf(pA.Not())
                    m.Add(term == 0).OnlyEnforceIf(pB.Not())
                    terms.append((edge.weight, term, 2 * max_bound))
        return terms

    def pairwise_spread_terms(self):
        """All-pairs distance terms among movable instances (any type),
        used for the 'compact' clustering objective. Same presence-gated
        construction as flow_terms but over every pair rather than only
        flow-connected ones.
        """
        m = self.model
        keys = list(self.presence.keys())
        max_bound = 2 * (self.W + self.H)
        terms = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                ka, kb = keys[i], keys[j]
                cxA, cyA = self.center2(ka)
                cxB, cyB = self.center2(kb)
                dx = m.NewIntVar(-max_bound, max_bound, f"sdx_{ka}_{kb}")
                dy = m.NewIntVar(-max_bound, max_bound, f"sdy_{ka}_{kb}")
                m.Add(dx == cxA - cxB)
                m.Add(dy == cyA - cyB)
                adx = m.NewIntVar(0, max_bound, f"sadx_{ka}_{kb}")
                ady = m.NewIntVar(0, max_bound, f"sady_{ka}_{kb}")
                m.AddAbsEquality(adx, dx)
                m.AddAbsEquality(ady, dy)
                dist = m.NewIntVar(0, 2 * max_bound, f"sdist_{ka}_{kb}")
                m.Add(dist == adx + ady)
                pA, pB = self.presence[ka], self.presence[kb]
                term = m.NewIntVar(0, 2 * max_bound, f"sterm_{ka}_{kb}")
                m.Add(term == dist).OnlyEnforceIf([pA, pB])
                m.Add(term == 0).OnlyEnforceIf(pA.Not())
                m.Add(term == 0).OnlyEnforceIf(pB.Not())
                terms.append(term)
        return terms

    def extract(self, solver: cp_model.CpSolver) -> Tuple[List[Placement], Dict[str, int]]:
        placements: List[Placement] = []
        unplaced_counts: Dict[str, int] = {s.id: 0 for s in self.specs}
        for key, presence in self.presence.items():
            spec = self.spec_by_key[key]
            if solver.Value(presence):
                gx = solver.Value(self.x_start[key])
                gy = solver.Value(self.y_start[key])
                gw = solver.Value(self.bw[key])
                gh = solver.Value(self.bh[key])
                placements.append(Placement(
                    spec_id=spec.id, instance=key[1], name=spec.name, color=spec.color,
                    x=self.grid.origin_x + gx * self.cell,
                    y=self.grid.origin_y + gy * self.cell,
                    w=gw * self.cell, h=gh * self.cell,
                    stage_id=spec.stage_key(),
                ))
            else:
                unplaced_counts[spec.id] += 1
        return placements, unplaced_counts


def _solve(model: cp_model.CpModel, time_limit_s: float, seed: int) -> Tuple[cp_model.CpSolver, int]:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = seed
    status = solver.Solve(model)
    return solver, status


def _placement_metrics(build: "_ModelBuild", solver, placements: List[Placement],
                        flow_edges: List[FlowEdge]) -> Dict[str, float]:
    """Recompute human-readable metrics (in metres) from a solved model."""
    by_stage: Dict[str, List[Placement]] = {}
    for p in placements:
        by_stage.setdefault(p.stage_id or p.spec_id, []).append(p)

    total_flow = 0.0
    longest_run = 0.0
    for edge in flow_edges:
        for a in by_stage.get(edge.from_id, []):
            for b in by_stage.get(edge.to_id, []):
                cxa, cya = a.x + a.w / 2, a.y + a.h / 2
                cxb, cyb = b.x + b.w / 2, b.y + b.h / 2
                d = abs(cxa - cxb) + abs(cya - cyb)
                total_flow += edge.weight * d
                longest_run = max(longest_run, d)

    if placements:
        minx = min(p.x for p in placements)
        miny = min(p.y for p in placements)
        maxx = max(p.x + p.w for p in placements)
        maxy = max(p.y + p.h for p in placements)
        footprint = (maxx - minx) + (maxy - miny)
    else:
        footprint = 0.0

    return {
        "total_flow_distance_m": round(total_flow, 1),
        "longest_single_run_m": round(longest_run, 1),
        "footprint_extent_m": round(footprint, 1),
        "placed_area_m2": round(sum(p.w * p.h for p in placements), 1),
    }


def generate_layouts(
    grid: Grid,
    specs: List[BuildingSpec],
    flow_edges: List[FlowEdge],
    safety_rules: List[SafetyRule],
    spacing: float,
    time_limit_s: float = 20.0,
    locked: Optional[List[Placement]] = None,
) -> List[LayoutResult]:
    """Generate several genuinely different, named layout alternatives.

    Phase 1: maximize total placed area/count under all hard constraints
    (boundary + obstacle avoidance, spacing, hard safety clearances). This
    fixes the best achievable A* — no later strategy is allowed to place
    less equipment just to improve its secondary objective.

    Phase 2 (per strategy): rebuild the model, add `placed_area >= A*`, and
    optimize a strategy-specific secondary objective:
      - Flow-Optimized:      minimize total flow-weighted pipe/conveyance distance
      - Compact:             minimize overall clustering spread of all equipment
      - Balanced Pipe Runs:  minimize the single longest flow-connected distance
    """
    strategies = [
        ("Flow-Optimized", "Minimizes total flow-weighted distance along the process "
                            "sequence (shortest total gas/material piping)."),
        ("Compact / Land-Efficient", "Minimizes overall spread of all equipment, favoring "
                                      "the smallest used footprint and shortest internal roads."),
        ("Balanced Pipe Runs", "Minimizes the single longest process-connected run, avoiding "
                                "any one pipeline dominating the layout."),
    ]

    # ── Phase 1: establish A* ──────────────────────────────────────────
    build0 = _ModelBuild(grid, specs, spacing, safety_rules, locked)
    build0.model.Maximize(build0.area_expr())
    solver0, status0 = _solve(build0.model, time_limit_s, seed=1)

    if status0 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Infeasible even for max-area — return one failed result explaining why.
        return [LayoutResult(
            name="Infeasible",
            description="No feasible layout found under the current constraints.",
            placements=[], unplaced=[(s.id, s.count) for s in specs],
            status=solver0.StatusName(status0), metrics={}, grid=grid,
        )]

    a_star = solver0.Value(build0.area_expr())

    results: List[LayoutResult] = []
    for i, (name, desc) in enumerate(strategies):
        build = _ModelBuild(grid, specs, spacing, safety_rules, locked)
        build.model.Add(build.area_expr() >= a_star)

        if name == "Flow-Optimized":
            terms = build.flow_terms(flow_edges)
            if terms:
                build.model.Minimize(sum(w_int_term(w) * t for w, t, _ in terms))
            else:
                build.model.Minimize(0)
        elif name == "Compact / Land-Efficient":
            terms = build.pairwise_spread_terms()
            if terms:
                build.model.Minimize(sum(terms))
            else:
                build.model.Minimize(0)
        else:  # Balanced Pipe Runs
            terms = build.flow_terms(flow_edges)
            if terms:
                max_term = build.model.NewIntVar(0, max(b for _, _, b in terms), "max_flow_term")
                build.model.AddMaxEquality(max_term, [t for _, t, _ in terms])
                build.model.Minimize(max_term)
            else:
                build.model.Minimize(0)

        solver, status = _solve(build.model, time_limit_s, seed=10 + i)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            placements, unplaced_counts = build.extract(solver)
            metrics = _placement_metrics(build, solver, placements, flow_edges)
            unplaced = [(sid, n) for sid, n in unplaced_counts.items() if n > 0]
            results.append(LayoutResult(
                name=name, description=desc, placements=placements, unplaced=unplaced,
                status=solver.StatusName(status), metrics=metrics, grid=grid,
            ))
        else:
            # Shouldn't normally happen since Phase 1 proved a_star is
            # achievable, but guard against solver time-outs on Phase 2.
            results.append(LayoutResult(
                name=name, description=desc + " (solver could not complete in time limit)",
                placements=[], unplaced=[(s.id, s.count) for s in specs],
                status=solver.StatusName(status), metrics={}, grid=grid,
            ))

    return results


def w_int_term(w: float) -> int:
    """Scale a flow weight to an integer multiplier for the CP-SAT objective."""
    return int(round(w * 100))
