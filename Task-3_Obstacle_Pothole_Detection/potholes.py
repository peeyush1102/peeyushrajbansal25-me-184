
import cv2
import numpy as np
import sys
import os


def detect_potholes(img, hsv):
    h, w = img.shape[:2]

    lower_white = np.array([0, 0, 190])
    upper_white = np.array([180, 60, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 255
    params.thresholdStep = 10

    params.filterByColor = True
    params.blobColor = 255

    params.filterByArea = True
    params.minArea = 150
    params.maxArea = 0.02 * h * w

    params.filterByCircularity = True
    params.minCircularity = 0.55

    params.filterByInertia = True
    params.minInertiaRatio = 0.05

    params.filterByConvexity = True
    params.minConvexity = 0.8

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(white_mask)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    potholes = []
    for kp in keypoints:
        cx, cy = kp.pt
        best = None
        for c in contours:
            if cv2.pointPolygonTest(c, (cx, cy), False) >= 0:
                best = c
                break
        if best is None:
            # fall back to the keypoint's own circular extent
            r = kp.size / 2.0
            x1, y1 = max(0, int(cx - r)), max(0, int(cy - r))
            x2, y2 = min(w - 1, int(cx + r)), min(h - 1, int(cy + r))
            bbox = (x1, y1, x2 - x1, y2 - y1)
        else:
            bbox = cv2.boundingRect(best)

        potholes.append({
            "bbox": bbox,
            "center": (int(cx), int(cy)),
        })
    return potholes, white_mask


def detect_obstacles(img, hsv):
    h, w = img.shape[:2]

    # Saturated / colored pixels = obstacles (cylinders, crates, cones...)
    lower_color = np.array([0, 80, 40])
    upper_color = np.array([180, 255, 255])
    color_mask = cv2.inRange(hsv, lower_color, upper_color)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    obstacles = []
    min_area = 0.001 * h * w
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        obstacles.append({
            "bbox": (x, y, bw, bh),
            "center": (x + bw // 2, y + bh // 2),
        })
    return obstacles, color_mask


def annotate(img, obstacles, potholes):
    out = img.copy()

    for i, obs in enumerate(obstacles, start=1):
        x, y, w, h = obs["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"Obstacle {i} ({x},{y})"
        cv2.putText(out, label, (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for i, ph in enumerate(potholes, start=1):
        x, y, w, h = ph["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
        label = f"Pothole {i} ({x},{y})"
        cv2.putText(out, label, (x, min(out.shape[0] - 5, y + h + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    total = len(obstacles) + len(potholes)
    summary = f"Obstacles: {len(obstacles)}  Potholes: {len(potholes)}  Total: {total}"
    (tw, th), _ = cv2.getTextSize(summary, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(out, (5, 5), (15 + tw, 15 + th + 10), (0, 0, 0), -1)
    cv2.putText(out, summary, (10, 10 + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return out


def process_image(path, out_path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    obstacles, _ = detect_obstacles(img, hsv)
    potholes, _ = detect_potholes(img, hsv)

    out = annotate(img, obstacles, potholes)
    cv2.imwrite(out_path, out)

    print(f"\n=== {os.path.basename(path)} ===")
    for i, o in enumerate(obstacles, start=1):
        x, y, w, h = o["bbox"]
        print(f"  Obstacle {i}: bbox=(x={x}, y={y}, w={w}, h={h}) center={o['center']}")
    for i, p in enumerate(potholes, start=1):
        x, y, w, h = p["bbox"]
        print(f"  Pothole {i}: bbox=(x={x}, y={y}, w={w}, h={h}) center={p['center']}")
    print(f"  TOTAL obstacles = {len(obstacles)}, TOTAL potholes = {len(potholes)}, "
          f"GRAND TOTAL = {len(obstacles) + len(potholes)}")

    return len(obstacles), len(potholes)


def process_folder(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = sorted(f for f in os.listdir(in_dir) if f.lower().endswith(exts))
    grand_obstacles = grand_potholes = 0
    for f in files:
        n_obs, n_ph = process_image(os.path.join(in_dir, f),
                                     os.path.join(out_dir, f"detected_{f}"))
        grand_obstacles += n_obs
        grand_potholes += n_ph
    print(f"\n============================================")
    print(f"Processed {len(files)} images.")
    print(f"Grand total obstacles = {grand_obstacles}, potholes = {grand_potholes}, "
          f"overall total = {grand_obstacles + grand_potholes}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:")
        print("  python3 detect.py <input_image> <output_image>       # single image")
        print("  python3 detect.py <input_folder> <output_folder>     # batch mode")
        sys.exit(1)

    src, dst = sys.argv[1], sys.argv[2]
    if os.path.isdir(src):
        process_folder(src, dst)
    else:
        process_image(src, dst)