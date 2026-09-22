"""Remove editorial review text while retaining scientific plot labels."""
from matplotlib.text import Text


def remove_review_text(figure):
    """Keep evidence status in captions/data rather than plot footers."""
    title = getattr(figure, "_suptitle", None)
    if title is not None:
        title.set_text("")
        title.set_visible(False)
    for artist in figure.findobj(Text):
        content = artist.get_text().strip()
        if ("SMOKE" in content or "PROTOTYPE" in content
                or content.startswith("Nine paired targets")
                or content.startswith("Both roots:")):
            artist.set_text("")
            artist.set_visible(False)
