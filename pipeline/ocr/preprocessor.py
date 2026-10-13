"""Image preprocessing pipeline for improved OCR accuracy on scanned documents."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _noise_filter(gray: np.ndarray) -> np.ndarray:
    """
    Remove scan noise using Non-Local Means denoising.

    Non-Local Means averages pixels based on patch similarity across the
    entire image, which preserves text edges better than Gaussian or median
    blurring while eliminating speckle noise typical in scanner output.

    Args:
        gray: Single-channel uint8 numpy array.

    Returns:
        Denoised single-channel uint8 numpy array.
    """
    return cv2.fastNlMeansDenoising(
        gray,
        h=10,                # filter strength — higher removes more noise but softens text
        templateWindowSize=7,
        searchWindowSize=21,
    )


def _contrast_enhancement(gray: np.ndarray) -> np.ndarray:
    """
    Equalise contrast using CLAHE (Contrast Limited Adaptive Histogram Equalisation).

    Unlike global histogram equalisation, CLAHE divides the image into small
    tiles and equalises each independently, which handles documents where some
    regions are darker than others (e.g. book-spine shadows, uneven lighting).
    The clip limit prevents over-amplifying noise in near-uniform regions.

    Args:
        gray: Single-channel uint8 numpy array.

    Returns:
        Contrast-enhanced single-channel uint8 numpy array.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _edge_enhancement(gray: np.ndarray) -> np.ndarray:
    """
    Sharpen text edges using an unsharp mask.

    An unsharp mask subtracts a blurred copy from the original, amplifying
    high-frequency detail (letter strokes, serifs). This makes OCR easier on
    slightly out-of-focus or low-resolution scans.

    Args:
        gray: Single-channel uint8 numpy array.

    Returns:
        Sharpened single-channel uint8 numpy array.
    """
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
    # addWeighted(src1, alpha, src2, beta, gamma): result = alpha*src1 + beta*src2 + gamma
    return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)


def preprocess(image: Image.Image) -> Image.Image:
    """
    Apply a three-step preprocessing pipeline to a scanned page image.

    Steps applied in order:
        1. **Noise filtering** — Non-Local Means denoising removes speckle
           and scanner noise without blurring text strokes.
        2. **Contrast enhancement** — CLAHE normalises brightness across the
           page so dark or washed-out regions become legible.
        3. **Edge enhancement** — Unsharp mask sharpens character edges,
           improving recognition of thin strokes and small fonts.

    The image is converted to greyscale for processing (colour carries no
    useful information for OCR) and returned as RGB so it is compatible with
    all three OCR engines (Tesseract, docTR, Claude).

    Args:
        image: Input PIL Image (any mode).

    Returns:
        Preprocessed PIL Image in RGB mode, same dimensions as input.
    """
    gray = np.array(image.convert("L"), dtype=np.uint8)

    gray = _noise_filter(gray)
    gray = _contrast_enhancement(gray)
    gray = _edge_enhancement(gray)

    return Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
