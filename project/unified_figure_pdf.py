"""Export an ordered figure-only PDF with uniform page width and margins."""
from pathlib import Path
from io import BytesIO

from rineng_content_id import content_identity
import json

PAGE_WIDTH_PT = 841.8897637795277  # A3 width; individual figure exports remain native.
SIDE_MARGIN_PT = 24.0
TOP_MARGIN_PT = 32.0
BOTTOM_MARGIN_PT = 25.0


def write_figures(ordered, destination, *, root, mode="smoke"):
    """Preserve every vector plate and its aspect ratio, one figure per page.

    Page height follows the source aspect ratio so tall appendix plates keep
    their readable width. The only added text is the figure ID and page number.
    """
    from pypdf import PdfReader, PdfWriter, Transformation
    from reportlab.pdfgen import canvas
    ordered = list(ordered)
    if len(ordered) != 31 or len({identifier for identifier, _ in ordered}) != 31:
        raise ValueError("Expected 31 distinct scientific figures in the accepted order")
    root = Path(root).resolve()
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    records = []
    for index, (identifier, path) in enumerate(ordered, 1):
        path = Path(path).resolve()
        reader = PdfReader(path)
        if len(reader.pages) != 1:
            raise ValueError(f"Expected one scientific plate: {path}")
        source = reader.pages[0]
        source.transfer_rotation_to_content()
        box = source.cropbox
        width, height = float(box.width), float(box.height)
        scale = (PAGE_WIDTH_PT - 2 * SIDE_MARGIN_PT) / width
        page_height = height * scale + TOP_MARGIN_PT + BOTTOM_MARGIN_PT
        page = writer.add_blank_page(PAGE_WIDTH_PT, page_height)
        dx = SIDE_MARGIN_PT - float(box.left) * scale
        dy = BOTTOM_MARGIN_PT - float(box.bottom) * scale
        page.merge_transformed_page(source, Transformation().scale(scale).translate(dx, dy))
        overlay = BytesIO()
        c = canvas.Canvas(overlay, pagesize=(PAGE_WIDTH_PT, page_height))
        c.setFillColorRGB(.20, .25, .29)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(SIDE_MARGIN_PT, page_height - 19, "Figure " + identifier)
        c.setFont("Helvetica", 8)
        c.drawRightString(PAGE_WIDTH_PT - SIDE_MARGIN_PT, 10, f"{index} / {len(ordered)}")
        c.save()
        page.merge_page(PdfReader(overlay).pages[0])
        writer.add_outline_item("Figure " + identifier, index - 1)
        records.append(dict(page=index, figure=identifier,
            source_pdf=path.relative_to(root).as_posix(),
            source_content_id=content_identity(path.read_bytes()).hexdigest(),
            source_dimensions_pt=[width, height],
            page_dimensions_pt=[PAGE_WIDTH_PT, page_height],
            vector_scale=scale, translation_pt=[dx, dy]))
    writer.add_metadata({"/Title": "GFE - smoke figures in order" if mode == "smoke"
                         else "GFE - full figures in order",
                         "/Subject": "31 scientific figures; uniform width; native vector artwork"})
    temporary = destination.with_suffix(".pdf.writing")
    with temporary.open("wb") as stream:
        writer.write(stream)
    temporary.replace(destination)
    manifest = dict(mode=mode, scientific_figures=31, pages=31,
        page_width_pt=PAGE_WIDTH_PT, side_margin_pt=SIDE_MARGIN_PT,
        top_margin_pt=TOP_MARGIN_PT, bottom_margin_pt=BOTTOM_MARGIN_PT,
        layout="One figure per page; uniform width; height follows source aspect ratio",
        scientific_source_modified=False, figures=records)
    destination.with_suffix(".figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return str(destination)
