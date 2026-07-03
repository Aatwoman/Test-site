"""
dxf_import.py — Extract a site boundary polygon and classified equipment/
obstacle polygons from a DXF (CAD interchange) file, in metres, for a biogas
/ CBG (compressed biogas) plant layout.

Real-world plant DXF exports are messy in three specific ways this module
handles:
  1. Boundaries and equipment outlines are often disconnected LINE/ARC
     segments rather than a single closed polyline.
  2. Units vary (mm/cm/m/inches/feet, or unset entirely).
  3. Equipment symbols are frequently drawn once as a BLOCK and placed many
     times via INSERT (e.g. a "DIGESTER" block placed 3x), and boundary/fill
     areas are sometimes drawn as HATCH regions rather than closed polylines.
     Earlier versions of this importer skipped both of those — this version
     expands INSERT block references (including nested blocks) into their
     underlying geometry, and reads HATCH boundary paths (polyline paths
     fully; line/arc edge paths on a best-effort basis — spline/ellipse hatch
     edges are still out of scope).

Known remaining gaps (not attempted here): XREF-based drawings where the
boundary or equipment live in an externally referenced file, and SPLINE /
ELLIPSE entities used directly (not inside a hatch boundary). If your plant
DXF uses those, that geometry will be silently absent from the import and
the resulting boundary/obstacle set may be incomplete — check the returned
`warnings` and the imported area against your drawing before trusting it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import glob
import io
import math
import os
import platform
import shutil
import tempfile

import ezdxf
from ezdxf.entities import Arc, Circle
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union

import biogas_config

try:
    from ezdxf.addons import odafc
except Exception:      # pragma: no cover - odafc ships with ezdxf itself
    odafc = None


# ── ODA File Converter discovery ─────────────────────────────────────────
# DWG is a closed, proprietary binary format that `ezdxf` cannot parse
# directly. The free ODA File Converter is a separate, non-Python binary
# that does the actual DWG -> DXF conversion; `ezdxf.addons.odafc` just
# shells out to it. Its own `is_installed()` only checks one configured
# path (or PATH, on Linux/macOS) — a real install is easy to miss purely
# because it lives in a version-numbered folder name or somewhere not on
# PATH. `configure_odafc()` below probes the usual install locations per
# OS and, if it finds one, registers it with ezdxf's own config so
# `odafc.readfile()` picks it up with no other change needed.

_WIN_GLOBS = [
    r"C:\Program Files\ODA\ODAFileConverter*\ODAFileConverter.exe",
    r"C:\Program Files (x86)\ODA\ODAFileConverter*\ODAFileConverter.exe",
]
_MAC_GLOBS = [
    "/Applications/ODAFileConverter*.app/Contents/MacOS/ODAFileConverter",
]
_LINUX_GLOBS = [
    "/usr/bin/ODAFileConverter",
    "/usr/local/bin/ODAFileConverter",
    "/opt/ODAFileConverter*/ODAFileConverter",
    "/opt/oda/ODAFileConverter",
    os.path.expanduser("~/Apps/ODAFileConverter*.AppImage"),
    os.path.expanduser("~/*ODAFileConverter*.AppImage"),
]


def _probe_common_odafc_paths() -> Optional[str]:
    """Search the usual per-OS install locations for the ODA File Converter
    binary. Returns the first existing match, or None. Doesn't touch PATH
    or ezdxf config — just finds candidates for configure_odafc() to use.
    """
    system = platform.system()
    patterns = {"Windows": _WIN_GLOBS, "Darwin": _MAC_GLOBS}.get(system, _LINUX_GLOBS)
    for pattern in patterns:
        for hit in sorted(glob.glob(pattern), reverse=True):  # prefer newer version folders
            if os.path.isfile(hit):
                return hit
    return None


def configure_odafc(custom_path: Optional[str] = None) -> Tuple[bool, Optional[str], str]:
    """Make sure `ezdxf.addons.odafc` can find the ODA File Converter.

    Args:
        custom_path: an explicit path to try first (e.g. supplied by the
            user in the UI for a non-standard install). If given and
            invalid, this reports failure rather than falling back to
            auto-detection, so the user gets accurate feedback on the
            path they typed.

    Returns:
        (available, resolved_path, detail) — `detail` is a short
        human-readable explanation suitable for direct display in the UI.
    """
    if odafc is None:
        return False, None, "the ezdxf 'odafc' add-on isn't available in this install"

    system = platform.system()
    opt_key = "win_exec_path" if system == "Windows" else "unix_exec_path"

    if custom_path:
        custom_path = custom_path.strip().strip('"')
        if os.path.isfile(custom_path):
            ezdxf.options.set("odafc-addon", opt_key, custom_path)
            return True, custom_path, "using the path you provided"
        return False, None, f"that path doesn't exist or isn't a file: {custom_path}"

    if odafc.is_installed():
        if system == "Windows":
            found = odafc.get_win_exec_path()
        else:
            found = odafc.get_unix_exec_path() or shutil.which("ODAFileConverter") or "ODAFileConverter (on PATH)"
        return True, found, "found (already configured / on PATH)"

    probed = _probe_common_odafc_paths()
    if probed:
        ezdxf.options.set("odafc-addon", opt_key, probed)
        if odafc.is_installed():
            return True, probed, "found at a standard install location"

    return False, None, "not found"


def odafc_install_guide() -> str:
    """Short, platform-appropriate install instructions for the ODA File
    Converter, reused in both the sidebar status panel and error messages.
    """
    system = platform.system()
    if system == "Windows":
        return (
            "Download and run the Windows installer from "
            "https://www.opendesign.com/guestfiles/oda_file_converter — installs to "
            r"`C:\Program Files\ODA\ODAFileConverter <version>\` by default, which this app "
            "auto-detects. If you installed it somewhere else, paste the full path to "
            "`ODAFileConverter.exe` below."
        )
    if system == "Darwin":
        return (
            "Download the macOS build from "
            "https://www.opendesign.com/guestfiles/oda_file_converter and drag it into "
            "`/Applications`, which this app auto-detects. If it's elsewhere, paste the full "
            "path to the `ODAFileConverter` binary inside the `.app` bundle "
            "(`.../ODAFileConverter.app/Contents/MacOS/ODAFileConverter`) below."
        )
    return (
        "Download the Linux `.deb`/`.rpm` or AppImage from "
        "https://www.opendesign.com/guestfiles/oda_file_converter. A package install "
        "(`ODAFileConverter` on PATH, or in `/usr/bin`, `/usr/local/bin`, `/opt`) is "
        "auto-detected; for an AppImage, `chmod +x` it and paste its full path below."
    )


# DXF $INSUNITS codes -> metres-per-unit
_UNIT_TO_METRES = {
    0: 1.0,      # unitless -> assume metres (most common for site plans)
    1: 0.0254,   # inches
    2: 0.3048,   # feet
    3: 1609.34,  # miles
    4: 0.001,    # millimetres
    5: 0.01,     # centimetres
    6: 1.0,      # metres
    8: 0.000001, # microns
    9: 0.0000254,# mils
    10: 0.9144,  # yards
}

BOUNDARY_LAYER_HINTS = ("boundary", "site", "property", "plot", "land", "parcel")

# Generic "this is some kind of existing structure" fallback hints, used only
# when a shape doesn't match any specific biogas equipment layer hint either.
GENERIC_OBSTACLE_LAYER_HINTS = ("obstacle", "existing", "structure", "building", "no-build", "nobuild", "exclusion")

_MAX_INSERT_DEPTH = 6  # guard against pathological/cyclic block nesting


@dataclass
class EquipmentZone:
    """A closed shape found in the DXF that lies inside the boundary and is
    therefore either an already-built piece of equipment/infrastructure to
    treat as fixed, or an unclassified obstacle.
    """
    polygon: Polygon
    stage_id: Optional[str]   # biogas_config.PROCESS_STAGES id, or None if unmatched
    stage_name: Optional[str]
    layer: str


@dataclass
class DxfImportResult:
    boundary: Polygon
    equipment_zones: List[EquipmentZone]
    detected_unit: str
    unit_scale: float           # metres per DXF drawing unit, as applied
    warnings: List[str]

    @property
    def obstacles(self) -> List[Polygon]:
        """Plain polygon list for callers (e.g. the rasterizer) that only
        need geometry, not equipment classification.
        """
        return [z.polygon for z in self.equipment_zones]


class DxfImportError(ValueError):
    pass


def _arc_to_points(cx: float, cy: float, r: float, a0_deg: float, a1_deg: float,
                    segments_per_360: int = 64) -> List[Tuple[float, float]]:
    a0 = math.radians(a0_deg)
    a1 = math.radians(a1_deg)
    if a1 <= a0:
        a1 += 2 * math.pi
    n = max(2, int(segments_per_360 * (a1 - a0) / (2 * math.pi)) + 1)
    return [(cx + r * math.cos(a0 + (a1 - a0) * i / n), cy + r * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]


def _circle_to_polygon(cx: float, cy: float, r: float, segments: int = 64) -> Polygon:
    pts = [(cx + r * math.cos(2 * math.pi * i / segments), cy + r * math.sin(2 * math.pi * i / segments)) for i in range(segments)]
    return Polygon(pts)


def _iter_entities_flattened(msp, warnings: List[str]):
    """Yield (layer, entity) pairs from modelspace, expanding INSERT block
    references (including nested INSERTs, up to _MAX_INSERT_DEPTH) into
    their transformed underlying geometry via ezdxf's virtual_entities(),
    which already applies the insertion point, scale, and rotation.
    """
    def _walk(entities, depth):
        for e in entities:
            etype = e.dxftype()
            if etype == "INSERT":
                if depth >= _MAX_INSERT_DEPTH:
                    warnings.append("Skipped a deeply nested block insert (depth limit reached).")
                    continue
                try:
                    children = list(e.virtual_entities())
                except Exception as ex:
                    warnings.append(f"Couldn't expand block insert '{e.dxf.name}': {ex}")
                    continue
                # Block-internal entities inherit the INSERT's layer if they
                # were drawn on layer "0" (the standard DXF "by-block" layer
                # convention) — otherwise keep their own layer name.
                insert_layer = (e.dxf.layer or "").lower()
                for c in children:
                    if (c.dxf.layer or "0") == "0" and insert_layer:
                        try:
                            c.dxf.layer = e.dxf.layer
                        except Exception:
                            pass
                yield from _walk(children, depth + 1)
            else:
                yield e

    yield from _walk(msp, 0)


def _hatch_boundary_polygons(hatch, warnings: List[str]) -> List[Polygon]:
    """Best-effort extraction of closed polygons from a HATCH entity's
    boundary paths. Handles polyline paths fully and line/arc edge paths;
    spline/ellipse edges within an edge path are skipped with a warning
    (the path is dropped rather than producing an incorrect polygon).
    """
    polys: List[Polygon] = []
    try:
        paths = hatch.paths.paths
    except Exception:
        return polys

    from ezdxf.entities.boundary_paths import BoundaryPathType

    for path in paths:
        pts: List[Tuple[float, float]] = []
        ok = True
        if path.type == BoundaryPathType.POLYLINE:
            pts = [(v[0], v[1]) for v in path.vertices]
        elif path.type == BoundaryPathType.EDGE:
            for edge in path.edges:
                et = edge.EDGE_TYPE
                if et == "LineEdge":
                    pts.append((edge.start_point[0], edge.start_point[1]))
                elif et == "ArcEdge":
                    arc_pts = _arc_to_points(edge.center[0], edge.center[1], edge.radius,
                                              edge.start_angle, edge.end_angle)
                    pts.extend(arc_pts)
                else:
                    ok = False
                    break
        else:
            ok = False

        if ok and len(pts) >= 3:
            poly = Polygon(pts)
            if poly.is_valid and poly.area > 0:
                polys.append(poly)
        elif not ok:
            warnings.append("Skipped a HATCH boundary path with a spline/ellipse edge (unsupported).")

    return polys


def _collect_linework(msp, warnings: List[str]) -> Tuple[List[LineString], List[Tuple[str, object]]]:
    """Walk modelspace entities (with INSERT blocks expanded), returning:
      - segments: open/closed line segments as LineStrings, fed to
        polygonize() to reconstruct closed faces even from disconnected
        boundary linework.
      - closed_by_layer: (layer, shapely_polygon) for every individually
        closed polyline/circle/hatch region, used to match layer-name hints
        directly (both generic boundary hints and biogas equipment hints).
    """
    segments: List[LineString] = []
    closed_by_layer: List[Tuple[str, object]] = []

    for e in _iter_entities_flattened(msp, warnings):
        etype = e.dxftype()
        layer = (e.dxf.layer or "").lower()

        if etype == "LWPOLYLINE":
            pts = [(p[0], p[1]) for p in e.get_points()]
            if len(pts) < 2:
                continue
            if e.closed and len(pts) >= 3:
                poly = Polygon(pts)
                if poly.is_valid and poly.area > 0:
                    closed_by_layer.append((layer, poly))
                    segments.append(LineString(pts + [pts[0]]))
                    continue
            segments.append(LineString(pts))

        elif etype == "POLYLINE":
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
            if len(pts) < 2:
                continue
            is_closed = bool(e.is_closed)
            if is_closed and len(pts) >= 3:
                poly = Polygon(pts)
                if poly.is_valid and poly.area > 0:
                    closed_by_layer.append((layer, poly))
                    segments.append(LineString(pts + [pts[0]]))
                    continue
            segments.append(LineString(pts))

        elif etype == "LINE":
            p0 = (e.dxf.start.x, e.dxf.start.y)
            p1 = (e.dxf.end.x, e.dxf.end.y)
            if p0 != p1:
                segments.append(LineString([p0, p1]))

        elif etype == "ARC":
            pts = _arc_to_points(e.dxf.center.x, e.dxf.center.y, e.dxf.radius,
                                  e.dxf.start_angle, e.dxf.end_angle)
            if len(pts) >= 2:
                segments.append(LineString(pts))

        elif etype == "CIRCLE":
            poly = _circle_to_polygon(e.dxf.center.x, e.dxf.center.y, e.dxf.radius)
            closed_by_layer.append((layer, poly))
            segments.append(LineString(list(poly.exterior.coords)))

        elif etype == "HATCH":
            for poly in _hatch_boundary_polygons(e, warnings):
                closed_by_layer.append((layer, poly))
                segments.append(LineString(list(poly.exterior.coords)))

        # SPLINE / ELLIPSE drawn directly (outside a HATCH boundary) are
        # still intentionally out of scope for v1 — flagged in the module
        # docstring and worth a warning if the drawing has none of the
        # entity types above at all (checked by the caller).

    return segments, closed_by_layer


def _detect_unit(doc) -> Tuple[str, float]:
    code = doc.header.get("$INSUNITS", 0)
    names = {0: "unitless (assumed metres)", 1: "inches", 2: "feet", 3: "miles", 4: "millimetres",
             5: "centimetres", 6: "metres", 8: "microns", 9: "mils", 10: "yards"}
    scale = _UNIT_TO_METRES.get(code, 1.0)
    return names.get(code, f"unknown (code {code}, assumed metres)"), scale


def import_dxf(
    file_bytes: bytes,
    boundary_layer: Optional[str] = None,
    manual_unit_scale: Optional[float] = None,
) -> DxfImportResult:
    """Parse a plant DXF file's bytes into a boundary polygon + classified
    equipment/obstacle zones, in metres.

    boundary_layer: optional exact/substring layer name to force as the
        boundary (case-insensitive). If None, auto-detects via common layer
        name hints, falling back to "largest closed shape wins".
    manual_unit_scale: if provided, overrides DXF header unit detection
        (metres per drawing unit).
    """
    try:
        doc = ezdxf.read(io.StringIO(file_bytes.decode("utf-8", errors="replace")))
    except Exception:
        try:
            with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tf:
                tf.write(file_bytes)
                tmp_path = tf.name
            try:
                doc = ezdxf.readfile(tmp_path)
            finally:
                os.unlink(tmp_path)
        except Exception as e:
            raise DxfImportError(f"Couldn't parse this file as DXF: {e}")

    return _process_doc(doc, boundary_layer, manual_unit_scale)


def import_dwg(
    file_bytes: bytes,
    boundary_layer: Optional[str] = None,
    manual_unit_scale: Optional[float] = None,
    custom_odafc_path: Optional[str] = None,
) -> DxfImportResult:
    """Parse a plant DWG file's bytes the same way `import_dxf` parses DXF —
    identical boundary/equipment-classification logic, identical return
    type. DWG is a closed, proprietary binary format that `ezdxf` cannot
    read directly, so this shells out to the free ODA File Converter (via
    `ezdxf.addons.odafc`) to convert to DXF first, then runs that DXF
    through the exact same `_process_doc` pipeline as a native DXF upload.

    Requires the ODA File Converter (https://www.opendesign.com/guestfiles/oda_file_converter)
    to be installed wherever this code runs — it's free, but not a Python
    package, so it isn't pulled in automatically by requirements.txt.
    `configure_odafc()` is tried first, which auto-detects standard install
    locations (and accepts `custom_odafc_path` for non-standard ones) before
    falling back to raising a DxfImportError with clear instructions rather
    than crashing or silently producing wrong geometry.
    """
    available, _resolved, _detail = configure_odafc(custom_odafc_path)
    if not available:
        raise DxfImportError(
            "DWG files need the free ODA File Converter installed on this machine "
            "to convert them (ezdxf itself can only read DXF). Two options: "
            "1) Install the ODA File Converter (https://www.opendesign.com/guestfiles/oda_file_converter), "
            "free, ~2 minutes, then re-upload this DWG — this app auto-detects standard install "
            "locations, or you can point it at a custom path in the sidebar. "
            "2) In your CAD software, use File → Save As / Export → DXF and upload "
            "that instead — this works immediately with no extra install."
        )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".dwg", delete=False) as tf:
            tf.write(file_bytes)
            tmp_path = tf.name
        try:
            doc = odafc.readfile(tmp_path)
        except odafc.ODAFCNotInstalledError:
            raise DxfImportError(
                "DWG files need the free ODA File Converter installed on this machine. "
                "See https://www.opendesign.com/guestfiles/oda_file_converter, or "
                "re-export this drawing as DXF from your CAD software and upload that instead."
            )
        except odafc.UnsupportedVersion as e:
            raise DxfImportError(f"Unsupported DWG version: {e}")
        except odafc.UnsupportedFileFormat as e:
            raise DxfImportError(f"Unsupported file format: {e}")
        except Exception as e:
            raise DxfImportError(
                f"Couldn't convert this DWG file ({e}). Try re-exporting it as DXF "
                "from your CAD software and uploading that instead."
            )
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return _process_doc(doc, boundary_layer, manual_unit_scale)


def import_cad_file(
    file_bytes: bytes,
    filename: str,
    boundary_layer: Optional[str] = None,
    manual_unit_scale: Optional[float] = None,
    custom_odafc_path: Optional[str] = None,
) -> DxfImportResult:
    """Dispatches to import_dwg() or import_dxf() based on `filename`'s
    extension. This is the function app.py should call for any upload —
    the two paths converge on the identical boundary/equipment-
    classification logic either way.

    custom_odafc_path is only used for DWG uploads (ignored for DXF).
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".dwg":
        return import_dwg(file_bytes, boundary_layer, manual_unit_scale, custom_odafc_path)
    return import_dxf(file_bytes, boundary_layer, manual_unit_scale)


def _process_doc(
    doc,
    boundary_layer: Optional[str] = None,
    manual_unit_scale: Optional[float] = None,
) -> DxfImportResult:
    """Everything downstream of "we have a parsed ezdxf Drawing" — shared
    by both the native-DXF path and the DWG-via-ODA path so the two file
    types get byte-for-byte identical boundary/equipment classification.
    """
    warnings: List[str] = []

    msp = doc.modelspace()
    segments, closed_by_layer = _collect_linework(msp, warnings)

    if not segments and not closed_by_layer:
        raise DxfImportError(
            "No usable geometry found (no LWPOLYLINE, POLYLINE, LINE, ARC, CIRCLE, HATCH, "
            "or block-inserted equivalents in modelspace). Check the file was exported with "
            "geometry on visible layers, and that any XREFs were bound before export."
        )

    detected_unit, auto_scale = _detect_unit(doc)
    unit_scale = manual_unit_scale if manual_unit_scale is not None else auto_scale

    # Build the planar face arrangement from all linework (handles
    # disconnected-segment boundaries and equipment outlines commonly seen in
    # real exports). Faces from polygonize() are the sole geometry source of
    # truth — a fully nested closed shape naturally becomes an interior ring
    # (hole) of its enclosing face, which is exactly the "boundary minus
    # equipment footprint" semantics we want. We deliberately do NOT also add
    # each closed entity's own solid polygon as a separate candidate: that
    # would duplicate/compete with the (more correct) hole-aware face.
    candidate_faces: List[Polygon] = []
    if segments:
        try:
            faces = list(polygonize(unary_union(segments)))
            for f in faces:
                if f.is_valid and f.area > 0:
                    candidate_faces.append(f)
        except Exception as e:
            warnings.append(f"Polygon assembly from line segments partially failed: {e}")

    if not candidate_faces:
        raise DxfImportError(
            "Found drawing entities, but none form a closed shape. The boundary "
            "must be a closed polyline, a HATCH region, or a set of lines/arcs that "
            "fully connect into a loop."
        )

    def _face_layer(face: Polygon) -> Optional[str]:
        face_ext = Polygon(face.exterior.coords)
        for layer, entity_poly in closed_by_layer:
            if face_ext.equals(entity_poly) or face_ext.symmetric_difference(entity_poly).area < 1e-6 * max(face_ext.area, 1):
                return layer
        return None

    # Choose the boundary: explicit layer hint > layer-name heuristic > largest area.
    boundary_poly = None
    boundary_source = None

    def _match_by_layer(predicate) -> Optional[Polygon]:
        matches = [f for f in candidate_faces if (lyr := _face_layer(f)) and predicate(lyr)]
        return max(matches, key=lambda p: p.area) if matches else None

    if boundary_layer:
        boundary_poly = _match_by_layer(lambda lyr: boundary_layer.lower() in lyr)
        if boundary_poly is not None:
            boundary_source = f"layer match '{boundary_layer}'"

    if boundary_poly is None:
        for hint in BOUNDARY_LAYER_HINTS:
            boundary_poly = _match_by_layer(lambda lyr, h=hint: h in lyr)
            if boundary_poly is not None:
                boundary_source = f"layer name hint '{hint}'"
                break

    if boundary_poly is None:
        boundary_poly = max(candidate_faces, key=lambda p: p.area)
        boundary_source = "largest closed shape (no boundary layer detected)"

    warnings.append(f"Boundary selected via: {boundary_source}.")

    # Equipment/obstacle zones: every interior ring of the chosen boundary
    # face, plus any other candidate face that sits inside the boundary's
    # footprint but wasn't already absorbed as a hole (e.g. a closed loop
    # sharing no topology with the boundary ring, drawn as a fully separate
    # sub-region — common for equipment symbols placed via block insert).
    raw_obstacle_polys: List[Polygon] = []
    boundary_exterior = Polygon(boundary_poly.exterior.coords)

    for interior in boundary_poly.interiors:
        raw_obstacle_polys.append(Polygon(interior.coords))

    for f in candidate_faces:
        if f.equals(boundary_poly):
            continue
        if f.area <= 0 or f.area >= boundary_exterior.area * 0.98:
            continue
        if boundary_exterior.contains(f.representative_point()):
            already = any(abs(f.area - o.area) < 1e-6 and f.centroid.distance(o.centroid) < 1e-6 for o in raw_obstacle_polys)
            if not already:
                raw_obstacle_polys.append(f)

    boundary_m = _scale_polygon(boundary_exterior, unit_scale)

    equipment_zones: List[EquipmentZone] = []
    classified_counts: dict = {}
    for o in raw_obstacle_polys:
        o_m = _scale_polygon(o, unit_scale)
        layer = _face_layer(o) or ""
        stage_id = biogas_config.classify_layer(layer)
        stage_name = biogas_config.stage(stage_id).name if stage_id else None
        equipment_zones.append(EquipmentZone(polygon=o_m, stage_id=stage_id, stage_name=stage_name, layer=layer))
        key = stage_name or "Unclassified existing structure"
        classified_counts[key] = classified_counts.get(key, 0) + 1

    if classified_counts:
        summary = ", ".join(f"{v}x {k}" for k, v in classified_counts.items())
        warnings.append(f"Existing equipment/obstacles found: {summary}.")

    if not boundary_m.is_valid or boundary_m.area <= 0:
        raise DxfImportError("The detected boundary polygon is invalid or has zero area.")

    return DxfImportResult(
        boundary=boundary_m,
        equipment_zones=equipment_zones,
        detected_unit=detected_unit,
        unit_scale=unit_scale,
        warnings=warnings,
    )


def _scale_polygon(poly: Polygon, scale: float) -> Polygon:
    from shapely.affinity import scale as shapely_scale
    return shapely_scale(poly, xfact=scale, yfact=scale, origin=(0, 0))
