"""
Lane Detection & Drivable Area Highlighting
--------------------------------------------
Processes every image in INPUT_DIR, detects the two lane boundaries,
overlays a filled polygon for the drivable area between them, and
saves each result into OUTPUT_DIR.

Usage:
    python lane_detection.py
    (edit INPUT_DIR / OUTPUT_DIR below, or pass them as CLI args)

    python lane_detection.py path/to/input path/to/output
"""

import cv2
import numpy as np
import os
import sys


# ----------------------------- Core CV steps -----------------------------

def canny(image):
    """Grayscale -> blur -> Canny edge map."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges


def region_of_interest(image):
    """
    Mask out the sky/trees/dashboard and keep a full-width horizontal
    band across the lower part of the image where the road surface
    and lane markings live. A full-width band (rather than a narrow
    triangle) is used because the input images come from many
    different cameras/angles/curvatures, and a narrow triangle tends
    to clip one of the two lane lines on curves. Defined as FRACTIONS
    of the image size so it works for any resolution.
    """
    height, width = image.shape[:2]
    polygon = np.array([[
        (0, height),
        (width, height),
        (width, int(0.45 * height)),
        (int(0.6 * width), int(0.35 * height)),
        (int(0.4 * width), int(0.35 * height)),
        (0, int(0.45 * height)),
    ]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(image, mask)


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    height, width = image.shape[:2]
    y1 = height
    y2 = int(y1 * 0.6)
    # guard against a vertical/near-zero slope blowing up the x coord
    if abs(slope) < 1e-6:
        slope = 1e-6
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    # clamp to image bounds -- a shallow slope can otherwise extrapolate
    # far off-canvas (e.g. x = 1291 in a 495px-wide image), which makes
    # the line invisible since both endpoints fall outside the frame
    x1 = int(np.clip(x1, 0, width - 1))
    x2 = int(np.clip(x2, 0, width - 1))
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    """
    Separate Hough lines into 'left lane' (negative slope) and
    'right lane' (positive slope) groups and average each into
    a single representative line. Returns None for a side if no
    lines were found on that side (instead of crashing).
    """
    left_fit, right_fit = [], []
    if lines is None:
        return None, None

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x1 == x2:
            continue  # perfectly vertical, polyfit would fail
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope, intercept = parameters
        # filter out near-horizontal noise (not real lane lines)
        if abs(slope) < 0.3:
            continue
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_line = None
    right_line = None
    if left_fit:
        left_line = make_coordinates(image, np.average(left_fit, axis=0))
    if right_fit:
        right_line = make_coordinates(image, np.average(right_fit, axis=0))

    return left_line, right_line


def draw_lane_overlay(image, left_line, right_line):
    """
    Draws the two lane boundary lines (in red) and fills the
    drivable area between them (in translucent green).
    """
    overlay = np.zeros_like(image)

    # Filled drivable-area polygon (only if both lines were found)
    if left_line is not None and right_line is not None:
        lx1, ly1, lx2, ly2 = left_line
        rx1, ry1, rx2, ry2 = right_line
        polygon = np.array([[
            (lx1, ly1), (lx2, ly2), (rx2, ry2), (rx1, ry1)
        ]], dtype=np.int32)
        cv2.fillPoly(overlay, polygon, (0, 255, 0))  # green fill

    # Lane boundary lines (drawn on top, in red)
    for line in (left_line, right_line):
        if line is not None:
            x1, y1, x2, y2 = line
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 10)

    # Blend: mostly original image + overlay
    combo = cv2.addWeighted(image, 1.0, overlay, 0.4, 0)
    return combo


def process_image(image):
    """Run the full pipeline on a single BGR image and return the result."""
    canny_image = canny(image)
    cropped = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(
        cropped, 2, np.pi / 180, 100,
        np.array([]), minLineLength=40, maxLineGap=5
    )
    left_line, right_line = average_slope_intercept(image, lines)
    result = draw_lane_overlay(image, left_line, right_line)
    return result


# ----------------------------- Batch driver -----------------------------

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(VALID_EXTS))

    if not files:
        print(f"No images found in {input_dir}")
        return

    for filename in files:
        in_path = os.path.join(input_dir, filename)
        image = cv2.imread(in_path)
        if image is None:
            print(f"  [skip] could not read {filename}")
            continue

        try:
            result = process_image(image)
        except Exception as e:
            print(f"  [warn] {filename}: {e} -- saving original with no overlay")
            result = image

        out_name = os.path.splitext(filename)[0] + "_lanes.jpg"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, result)
        print(f"  [ok] {filename} -> {out_name}")

    print(f"\nDone. {len(files)} image(s) processed. Output saved to: {output_dir}")


if __name__ == "__main__":
    # Default folders (relative to this script) -- override via CLI args
    default_input = os.path.join(os.path.dirname(__file__), "input")
    default_output = os.path.join(os.path.dirname(__file__), "output")

    input_dir = sys.argv[1] if len(sys.argv) > 1 else default_input
    output_dir = sys.argv[2] if len(sys.argv) > 2 else default_output

    process_folder(input_dir, output_dir)
