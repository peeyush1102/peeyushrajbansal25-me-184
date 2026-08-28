import cv2
import numpy as np
import heapq
from collections import deque

# ---------------- helper: build masks ----------------
def get_masks(img, safety_margin=17):
    """Return (road_mask, free_mask) as uint8 0/255 images."""
    # 1) find the road: median-blur removes the grainy noise texture,
    #    leaving the road as a clean, slightly-brighter gray band.
    blur = cv2.medianBlur(img, 15)
    gray_b = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    road_raw = cv2.inRange(gray_b, 90, 130)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    road_raw = cv2.morphologyEx(road_raw, cv2.MORPH_CLOSE, k, iterations=2)

    # keep only the largest blob -> the road loop itself
    n, labels, stats, _ = cv2.connectedComponentsWithStats(road_raw, 8)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    road_mask = np.uint8(labels == largest) * 255

    # 2) obstacles = colorful markers or dark pothole rings, on the road
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    colorful = (hsv[:, :, 1] > 55).astype(np.uint8) * 255
    dark = (gray < 60).astype(np.uint8) * 255
    obstacles = cv2.bitwise_and(cv2.bitwise_or(colorful, dark), road_mask)

    # inflate obstacles by the vehicle's safety margin
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (safety_margin, safety_margin))
    obstacles = cv2.dilate(obstacles, k2, iterations=1)

    free_mask = cv2.bitwise_and(road_mask, cv2.bitwise_not(obstacles))
    free_mask = largest_component(free_mask)   # drop any pockets sealed off by obstacles
    return road_mask, free_mask


# ---------------- helper: keep only the main drivable loop ----------------
def largest_component(mask):
    """Small pockets can get sealed off by obstacle inflation (e.g. under the
    START arrow icon). Keep only the single largest connected free blob so every
    point we plan with is guaranteed reachable from every other."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.uint8(labels == largest) * 255


# ---------------- helper: snap a point onto free space ----------------
def snap_to_free(free_mask, p, max_r=60):
    h, w = free_mask.shape
    if 0 <= p[0] < w and 0 <= p[1] < h and free_mask[p[1], p[0]] > 0:
        return p
    for r in range(1, max_r):
        for dx in range(-r, r + 1):
            for dy in (-r, r):
                x, y = p[0] + dx, p[1] + dy
                if 0 <= x < w and 0 <= y < h and free_mask[y, x] > 0:
                    return (x, y)
            for dy in range(-r, r + 1):
                for dx in (-r, r):
                    x, y = p[0] + dx, p[1] + dy
                    if 0 <= x < w and 0 <= y < h and free_mask[y, x] > 0:
                        return (x, y)
    return p


# ---------------- checkpoint method: sample points evenly around the loop ----------------
def get_checkpoints(road_mask, free_mask, start, n_checkpoints=10):
    """Bucket road pixels by angle around the track's centroid, one checkpoint per
    bucket, then order them starting next to 'start' so the route runs one full lap."""
    ys, xs = np.nonzero(road_mask)
    cx, cy = xs.mean(), ys.mean()
    ang = np.arctan2(ys - cy, xs - cx)

    bins = np.linspace(-np.pi, np.pi, n_checkpoints + 1)
    checkpoints = []
    for i in range(n_checkpoints):
        m = (ang >= bins[i]) & (ang < bins[i + 1])
        if m.sum() == 0:
            continue
        px, py = int(xs[m].mean()), int(ys[m].mean())
        checkpoints.append(snap_to_free(free_mask, (px, py)))

    start_ang = np.arctan2(start[1] - cy, start[0] - cx)
    checkpoints.sort(key=lambda p: (np.arctan2(p[1] - cy, p[0] - cx) - start_ang) % (2 * np.pi))
    return checkpoints


# ---------------- A* on a pixel grid ----------------
def astar(free_mask, start, goal, step=6):
    h, w = free_mask.shape

    def free(p):
        x, y = p
        return 0 <= x < w and 0 <= y < h and free_mask[y, x] > 0

    moves = [(-step, 0), (step, 0), (0, -step), (0, step),
             (-step, -step), (-step, step), (step, -step), (step, step)]

    def hcost(p):
        return np.hypot(p[0] - goal[0], p[1] - goal[1])

    open_set = [(hcost(start), start)]
    came_from = {}
    g = {start: 0}
    visited = set()

    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur in visited:
            continue
        visited.add(cur)
        if hcost(cur) < step:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            return path[::-1]
        for dx, dy in moves:
            nxt = (cur[0] + dx, cur[1] + dy)
            if not free(nxt):
                continue
            ng = g[cur] + np.hypot(dx, dy)
            if nxt not in g or ng < g[nxt]:
                g[nxt] = ng
                came_from[nxt] = cur
                heapq.heappush(open_set, (ng + hcost(nxt), nxt))
    return None


# ---------------- main per-image routine ----------------
def plan_path_for_image(in_path, out_path, start_point):
    img = cv2.imread(in_path)
    road_mask, free_mask = get_masks(img)

    start_point = snap_to_free(free_mask, start_point)
    checkpoints = get_checkpoints(road_mask, free_mask, start_point)
    waypoints = [start_point] + checkpoints + [start_point]   # one full lap, back to start

    full_path = []
    ok = True
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        leg = astar(free_mask, a, b)
        if leg is None:
            ok = False
            break
        full_path += leg if not full_path else leg[1:]

    result = img.copy()
    if full_path:
        for i in range(len(full_path) - 1):
            cv2.line(result, full_path[i], full_path[i + 1], (0, 255, 0), 4)
    for cp in checkpoints:
        cv2.circle(result, cp, 6, (0, 0, 255), -1)          # red  = checkpoints
    cv2.circle(result, start_point, 9, (255, 0, 0), -1)     # blue = start/finish
    cv2.imwrite(out_path, result)
    print(f"{in_path} -> {out_path}  |  full lap found: {ok}")
    return ok


if __name__ == "__main__":
    import os
    INPUT_FOLDER = "images"
    OUTPUT_FOLDER = "output"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # (x, y) pixel near the START arrow in each image - update these to match your files
    # (tip: open the image, hover your mouse over the arrow, and read the pixel coords)
    start_points = {
        "1.jpeg":  (1033, 664),
        "2.jpeg":  (1107, 672),
        "3.jpeg":  (915, 672),
        "4.jpeg":  (970, 683),
        "5.jpeg":  (970, 683),
        "6.jpeg":  (915, 536),
        "7.jpeg":  (235, 595),
        "8.jpeg":  (631, 1000),
        "9.jpeg":  (875, 736),
        "10.jpeg": (292, 834),
    }

    for filename, start_point in start_points.items():
        plan_path_for_image(os.path.join(INPUT_FOLDER, filename),
                             os.path.join(OUTPUT_FOLDER, f"path_{filename}"),
                             start_point)