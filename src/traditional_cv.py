import warnings
from src.core.traditional.base_processor import (
    sort_points,
    get_perspective_transform
)
from src.core.traditional.plate_locator import locate_license_plate
from src.core.traditional.document_scanner import scan_document

warnings.warn(
    "src/traditional_cv.py is deprecated and will be removed in a future release. "
    "Please import from src.core.traditional instead.",
    DeprecationWarning,
    stacklevel=2
)
