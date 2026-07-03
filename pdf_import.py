"""
pdf_import.py — Extract a site boundary polygon and classified equipment/
road/obstacle zones from an uploaded PDF plant layout drawing, for the same
biogas/CBG plant layout tool that dxf_import.py serves.

Returns the exact same `dxf_import.DxfImportResult` / `EquipmentZone` types
DXF import returns, so app.py's existing DXF-handling code (equipment ->
locked buildings, unclassified zones -> obstacles, requirement-count
auto-reduction, etc.) needs no changes at all — only the upload dispatch
does.

Two extraction paths, auto-selected per page (or forced via `force_raster`):

  1. VECTOR path (preferred, high fidelity): most "export/print to PDF"
     output from CAD software embeds real vector drawing commands (lines,
     bezier curves, rects) and real text runs in the PDF content stream —
     not a picture of the drawing. When present, we read those directly via
     PyMuPDF: exact coordinates, no OCR error. Lines/rects/beziers become
     shapely LineStrings, unioned and polygonized exactly like DXF's
     LINE/ARC/LWPOLYLINE handling, and any text whose bounding box falls
     inside a candidate shape is used to classify it (reusing
     `biogas_config.classify_layer()` against the extracted words, the same
     function DXF layer-name matching uses).

  2. RASTER path (fallback, best-effort): used when a page has little/no
     vector content — i.e. it's essentially a scanned image. The page is
     rasterized at high DPI, OpenCV finds closed contours (candidate
     boundary/building/road shapes), and Tesseract OCR reads text near each
     contour for classification. This is inherently approximate — always
     review the extracted shapes on the canvas (drag/resize/delete as
     needed) before running the optimizer.

SCALE / CALIBRATION — READ THIS: unlike DXF, PDF has no embedded
real-world-units header. A page is just points (1/72 inch) of *paper*, and
a plan is normally plotted at an engineering scale (1:100, 1:200, ...) that
isn't machine-readable from the PDF alone. So `import_pdf()` always returns
geometry in raw, uncalibrated PDF points (`unit_scale=1.0`,
`detected_unit="uncalibrated — set a real-world scale"`). Call
`rescale_result()` once you know the boundary's true width or height (or a
known plot scale) to convert it to metres — app.py's sidebar does this as a
one-time calibration step right after import, the same way it exposes a
manual units override for DXF.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import fitz  # PyMuPDF
import cv2
import pytesseract
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import polygonize, unary_union
from shapely.affinity import scale as shapely_scale

import biogas_config
from dxf_import import EquipmentZone, DxfImportResult, DxfImportError, BOUNDARY_LAYER_HINTS


class PdfImportError(DxfImportError):
    """Subclasses DxfImportError so app.py's existing `except
    DxfImportError` handling around the import call catches this too,
    with no changes needed there.
    """
    pass


@dataclass
class PdfImportInfo:
    extraction_method: str      # "vector" | "raster"
    raw_width: float            # PDF points, pre-calibration
    raw_height: float
    page_number: int
    page_count: int


ROAD_KEYWORDS = ("road", "access road", "driveway", "access way", "route", "carriageway", "internal road")

_MIN_VECTOR_ITEMS = 8      # fewer drawing items than this -> treat page as a scan
_BEZIER_SEGMENTS = 12
_RASTER_DPI = 300


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────

def import_pdf(
    file_bytes: bytes,
    page_number: int = 0,
    boundary_hint: Optional[str] = None,
    force_raster: bool = False,
    ocr_lang: str = "eng",
) -> Tuple[DxfImportResult, PdfImportInfo]:
    """Parse a plant PDF's bytes into a boundary polygon + classified
    equipment/road/obstacle zones, in RAW PDF POINTS (see module docstring
    re: calibration — call rescale_result() before treating anything here
    as metres).

    boundary_hint: optional substring matched (case-insensitive) against
        text found on/near a candidate shape to force it as the boundary,
        same idea as dxf_import's boundary_layer.
    force_raster: skip the vector-content check and always use the
        OCR/contour path (useful if a PDF has stray vector junk that
        confuses the vector path but is fundamentally a scanned drawing).
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise PdfImportError(f"Couldn't open this file as a PDF: {e}")

    if doc.page_count == 0:
        raise PdfImportError("This PDF has no pages.")
    if page_number >= doc.page_count:
        raise PdfImportError(f"Page {page_number + 1} doesn't exist — this PDF has {doc.page_count} page(s).")

    page = doc[page_number]
    warnings: List[str] = []
    if doc.page_count > 1:
        warnings.append(
            f"This PDF has {doc.page_count} pages — only page {page_number + 1} was imported."
        )

    drawings = page.get_drawings()
    vector_item_count = sum(len(d["items"]) for d in drawings)

    if not force_raster and vector_item_count >= _MIN_VECTOR_ITEMS:
        result = _extract_vector(page, boundary_hint, warnings)
        method = "vector"
    else:
        if not force_raster:
            warnings.append(
                f"Only {vector_item_count} vector drawing element(s) found on this page — "
                "treating it as a scanned/rasterized drawing and using OCR + shape detection "
                "instead. This is best-effort: check every extracted shape on the canvas "
                "before optimizing."
            )
        result = _extract_raster(page, boundary_hint, warnings, ocr_lang)
        method = "raster"

    info = PdfImportInfo(
        extraction_method=method,
        raw_width=page.rect.width,
        raw_height=page.rect.height,
        page_number=page_number,
        page_count=doc.page_count,
    )
    return result, info


def rescale_result(result: DxfImportResult, factor: float, unit_label: str = "calibrated metres") -> DxfImportResult:
    """Return a copy of `result` with every polygon scaled by `factor`
    (e.g. metres per raw PDF point) about the origin. Use once you know the
    boundary's real-world width/height: factor = true_width_m / raw_width.
    """
    if factor <= 0:
        raise PdfImportError("Scale factor must be positive.")
    new_boundary = _scale_polygon(result.boundary, factor)
    new_zones = [
        EquipmentZone(
            polygon=_scale_polygon(z.polygon, factor),
            stage_id=z.stage_id, stage_name=z.stage_name, layer=z.layer,
        )
        for z in result.equipment_zones
    ]
    return DxfImportResult(
        boundary=new_boundary,
        equipment_zones=new_zones,
        detected_unit=unit_label,
        unit_scale=result.unit_scale * factor,
        warnings=result.warnings,
    )


# ──────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────

def _scale_polygon(poly: Polygon, scale: float) -> Polygon:
    return shapely_scale(poly, xfact=scale, yfact=scale, origin=(0, 0))


def _flip_y(poly: Polygon, page_height: float) -> Polygon:
    """PyMuPDF reports coordinates top-left-origin, y increasing downward
    (image convention). dxf_import.py's DXF output is bottom-left-origin,
    y increasing upward (CAD convention), and everything downstream (the
    canvas, the optimizer) was built/tested against that. Flip once, at
    the very end, so PDF-derived geometry matches DXF-derived geometry's
    orientation exactly rather than importing upside-down.
    """
    coords = [(x, page_height - y) for x, y in poly.exterior.coords]
    holes = [[(x, page_height - y) for x, y in ring.coords] for ring in poly.interiors]
    return Polygon(coords, holes)


def _classify_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Match extracted text against biogas equipment keywords, then a
    generic road-keyword list. Returns (stage_id, stage_name) or
    (None, "Road / Access Way") or (None, None).
    """
    stage_id = biogas_config.classify_layer(text)
    if stage_id:
        return stage_id, biogas_config.stage(stage_id).name
    lower = text.lower()
    if any(kw in lower for kw in ROAD_KEYWORDS):
        return None, "Road / Access Way"
    return None, None


def _candidate_faces_from_segments(segments: List[LineString], warnings: List[str]) -> List[Polygon]:
    faces: List[Polygon] = []
    if not segments:
        return faces
    try:
        for f in polygonize(unary_union(segments)):
            if f.is_valid and f.area > 0:
                faces.append(f)
    except Exception as e:
        warnings.append(f"Polygon assembly from extracted lines partially failed: {e}")
    return faces


def _label_for_polygon(poly: Polygon, words: List[Tuple[float, float, str]]) -> str:
    """words: (center_x, center_y, text) in the same coordinate frame as
    poly. Small buffer so a label sitting just outside a thin-stroked
    outline (common — labels are often placed just above/beside a shape,
    not strictly inside it) still counts.
    """
    matched = [text for cx, cy, text in words if poly.buffer(3).contains(_pt(cx, cy))]
    return " ".join(matched)


def _pick_boundary(candidate_faces: List[Polygon], words: List[Tuple[float, float, str]],
                    boundary_hint: Optional[str], warnings: List[str]) -> Polygon:
    def _match(predicate):
        matches = [f for f in candidate_faces if predicate(_label_for_polygon(f, words).lower())]
        return max(matches, key=lambda p: p.area) if matches else None

    boundary = None
    if boundary_hint:
        boundary = _match(lambda t: boundary_hint.lower() in t)
        if boundary is not None:
            warnings.append(f"Boundary selected via: text match '{boundary_hint}'.")

    if boundary is None:
        for hint in BOUNDARY_LAYER_HINTS:
            boundary = _match(lambda t, h=hint: h in t)
            if boundary is not None:
                warnings.append(f"Boundary selected via: text hint '{hint}'.")
                break

    if boundary is None:
        boundary = max(candidate_faces, key=lambda p: p.area)
        warnings.append("Boundary selected via: largest closed shape (no boundary text label found).")

    return boundary


def _build_result(candidate_faces: List[Polygon], words: List[Tuple[float, float, str]],
                   boundary_hint: Optional[str], page_height: float, warnings: List[str]) -> DxfImportResult:
    if not candidate_faces:
        raise PdfImportError(
            "Found drawing content, but no closed shapes could be assembled from it. "
            "The boundary needs to be a closed outline (rectangle, polyline, or lines/curves "
            "that fully connect into a loop)."
        )

    boundary_poly = _pick_boundary(candidate_faces, words, boundary_hint, warnings)
    boundary_ext = Polygon(boundary_poly.exterior.coords)

    # NOTE: interior rings get reconstructed as fresh Polygon objects here,
    # so any labelling MUST happen after this point, by spatial containment
    # against these final shapes — not against `candidate_faces` by object
    # identity, which silently breaks the moment a shape's label lives on
    # an interior-ring-derived duplicate rather than the original face.
    raw_zones: List[Polygon] = [Polygon(r.coords) for r in boundary_poly.interiors]
    for f in candidate_faces:
        if f.equals(boundary_poly):
            continue
        if f.area <= 0 or f.area >= boundary_ext.area * 0.98:
            continue
        if boundary_ext.contains(f.representative_point()):
            dup = any(abs(f.area - o.area) < 1e-6 and f.centroid.distance(o.centroid) < 1e-6 for o in raw_zones)
            if not dup:
                raw_zones.append(f)

    equipment_zones: List[EquipmentZone] = []
    counts: dict = {}
    for z in raw_zones:
        text = _label_for_polygon(z, words)
        stage_id, stage_name = _classify_text(text)
        equipment_zones.append(EquipmentZone(
            polygon=_flip_y(z, page_height), stage_id=stage_id, stage_name=stage_name, layer=text,
        ))
        key = stage_name or "Unclassified existing structure"
        counts[key] = counts.get(key, 0) + 1

    if counts:
        summary = ", ".join(f"{v}x {k}" for k, v in counts.items())
        warnings.append(f"Existing equipment/roads/obstacles found: {summary}.")

    boundary_flipped = _flip_y(boundary_ext, page_height)
    if not boundary_flipped.is_valid or boundary_flipped.area <= 0:
        raise PdfImportError("The detected boundary shape is invalid or has zero area.")

    return DxfImportResult(
        boundary=boundary_flipped,
        equipment_zones=equipment_zones,
        detected_unit="uncalibrated PDF points — set a real-world scale before optimizing",
        unit_scale=1.0,
        warnings=warnings,
    )


# ──────────────────────────────────────────────────────────────────────────
# Vector path
# ──────────────────────────────────────────────────────────────────────────

def _tessellate_bezier(p0, p1, p2, p3, n: int = _BEZIER_SEGMENTS) -> List[Tuple[float, float]]:
    pts = []
    for i in range(n + 1):
        t = i / n
        mt = 1 - t
        x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
        pts.append((x, y))
    return pts


def _drawing_to_segments(drawing) -> List[LineString]:
    segments: List[LineString] = []
    for item in drawing["items"]:
        op = item[0]
        if op == "l":
            p0, p1 = item[1], item[2]
            if (p0.x, p0.y) != (p1.x, p1.y):
                segments.append(LineString([(p0.x, p0.y), (p1.x, p1.y)]))
        elif op == "c":
            p0, p1, p2, p3 = item[1], item[2], item[3], item[4]
            pts = _tessellate_bezier((p0.x, p0.y), (p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y))
            segments.append(LineString(pts))
        elif op == "re":
            r = item[1]
            pts = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1), (r.x0, r.y0)]
            segments.append(LineString(pts))
        elif op == "qu":
            q = item[1]
            pts = [(q.ul.x, q.ul.y), (q.ur.x, q.ur.y), (q.lr.x, q.lr.y), (q.ll.x, q.ll.y), (q.ul.x, q.ul.y)]
            segments.append(LineString(pts))
        # other ops (rare) are silently skipped rather than raising — a
        # missing decorative element shouldn't fail the whole import.
    return segments


def _extract_vector(page, boundary_hint: Optional[str], warnings: List[str]) -> DxfImportResult:
    drawings = page.get_drawings()
    segments: List[LineString] = []
    for d in drawings:
        segments.extend(_drawing_to_segments(d))

    candidate_faces = _candidate_faces_from_segments(segments, warnings)

    words = [
        ((x0 + x1) / 2, (y0 + y1) / 2, text)
        for x0, y0, x1, y1, text, *_ in page.get_text("words")
    ]

    return _build_result(candidate_faces, words, boundary_hint, page.rect.height, warnings)


def _pt(x, y):
    return Point(x, y)


# ──────────────────────────────────────────────────────────────────────────
# Raster / OCR path
# ──────────────────────────────────────────────────────────────────────────

def _extract_raster(page, boundary_hint: Optional[str], warnings: List[str], ocr_lang: str) -> DxfImportResult:
    pix = page.get_pixmap(dpi=_RASTER_DPI)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    px_per_pt = _RASTER_DPI / 72.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu binarization: works well for clean black-line-on-white-paper
    # scans; a photographed/uneven-lighting scan may need the person to
    # pre-process (crop, deskew, contrast) before uploading for good results.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Close small gaps in hand-drawn or slightly-broken lines so outlines
    # form fully connected loops before contour tracing.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # RETR_TREE (not EXTERNAL): we need genuinely nested contours (e.g. a
    # digester drawn inside the boundary) to survive. But a *stroked*
    # (unfilled) rectangle is a thin ring, which produces an outer-edge and
    # inner-edge contour a few pixels apart for the SAME shape — RETR_TREE
    # returns both. Deduplicate those pairs explicitly by bounding-box IoU
    # (near-identical box = same stroke's two edges, not two real shapes)
    # rather than by suppressing nesting altogether.
    contours, _hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    page_area_px = pix.width * pix.height
    min_area_px = page_area_px * 0.0005   # drop noise/text-glyph-sized specks
    max_area_px = page_area_px * 0.98     # drop a contour that's basically the whole page/border

    def _bbox_iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix0, iy0 = max(ax, bx), max(ay, by)
        ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    # Largest first, so we keep an outer stroke edge and discard its inner
    # edge (near-identical bbox) rather than the other way around.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    kept_contours: List = []
    kept_boxes: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px or area > max_area_px:
            continue
        bbox = cv2.boundingRect(cnt)
        if any(_bbox_iou(bbox, kb) > 0.85 for kb in kept_boxes):
            continue  # inner edge of an already-kept shape's stroke
        kept_contours.append(cnt)
        kept_boxes.append(bbox)

    candidate_faces: List[Polygon] = []
    for cnt in kept_contours:
        eps = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) < 3:
            continue
        pts_pt = [(pt[0][0] / px_per_pt, pt[0][1] / px_per_pt) for pt in approx]
        poly = Polygon(pts_pt)
        if poly.is_valid and poly.area > 0:
            candidate_faces.append(poly)
        elif not poly.is_valid:
            fixed = poly.buffer(0)
            if fixed.area > 0 and fixed.geom_type == "Polygon":
                candidate_faces.append(fixed)

    if not candidate_faces:
        raise PdfImportError(
            "OCR/shape detection found text but no usable closed outlines on this page. "
            "If this is a scanned drawing, try a higher-contrast scan (dark clean lines on "
            "plain white background) — faint, dashed, or hand-sketched outlines are the most "
            "common cause of this."
        )

    # OCR the whole page once (much faster than per-contour crops), then
    # hand every (center-point, text) word through to _build_result, which
    # does the actual shape association against the final zone geometry.
    # PSM 12 ("sparse text + orientation/script detection") rather than
    # Tesseract's default automatic-layout mode: in testing, the default
    # mode reliably missed isolated scattered labels that don't form
    # paragraph-like blocks — exactly the layout a CAD drawing has.
    # Note: labels rotated to fit inside a narrow shape (e.g. running along
    # a road) are a known gap here — Tesseract doesn't read rotated text at
    # 0 degrees, and multi-angle OCR passes were tried and dropped because
    # misreads of upright text from the "wrong" angle occasionally produced
    # garbage that could contaminate a shape's label. The vector path
    # doesn't have this limitation (PDF text carries its own rotation), so
    # for a drawing with load-bearing rotated labels, prefer a PDF that
    # still has real vector content over a flattened scan.
    ocr = pytesseract.image_to_data(img, lang=ocr_lang, config="--psm 12", output_type=pytesseract.Output.DICT)
    words: List[Tuple[float, float, str]] = []
    n = len(ocr["text"])
    for i in range(n):
        text = ocr["text"][i].strip()
        conf = float(ocr["conf"][i]) if ocr["conf"][i] not in ("", "-1") else -1
        if not text or conf < 40:
            continue
        x, y, w, h = ocr["left"][i], ocr["top"][i], ocr["width"][i], ocr["height"][i]
        words.append(((x + w / 2) / px_per_pt, (y + h / 2) / px_per_pt, text))

    return _build_result(candidate_faces, words, boundary_hint, page.rect.height, warnings)
