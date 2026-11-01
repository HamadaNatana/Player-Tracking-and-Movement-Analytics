"""Pitch geometry and landmark extraction shared by the keypoint scripts.

Pitch coordinates are meters with the origin at the pitch center, x along the length
(left goal at negative x) and y across the width (top touchline at negative y), as
inferred from SoccerNetGS bbox_pitch values. check_pitch_labels.py tests this
assumption against the data before any labels are trusted.

A landmark is the crossing of two named pitch lines, for example the top touchline
and the halfway line. SoccerNetGS stores each visible line as a polyline of points
given as fractions of the image width and height.
"""

import numpy as np

HALF_LENGTH = 52.5  # pitch is 105 m long
HALF_WIDTH = 34.0  # and 68 m wide
BOX_DEPTH = 16.5  # penalty box depth from the goal line
BOX_HALF_WIDTH = 20.16  # penalty box is 40.32 m wide
SIX_DEPTH = 5.5  # goal area depth
SIX_HALF_WIDTH = 9.16  # goal area is 18.32 m wide


def _build_landmarks():
    """Ordered {name: (line_a, line_b, (x, y))} for every line crossing we label."""
    landmarks = {}
    for side, sign in (("left", -1), ("right", 1)):
        goal_line = f"Side line {side}"
        goal_x = sign * HALF_LENGTH
        for end, y_sign in (("top", -1), ("bottom", 1)):
            touchline = f"Side line {end}"
            landmarks[f"corner {side} {end}"] = (touchline, goal_line, (goal_x, y_sign * HALF_WIDTH))
            for rect, depth, half_width in (("Big rect.", BOX_DEPTH, BOX_HALF_WIDTH),
                                            ("Small rect.", SIX_DEPTH, SIX_HALF_WIDTH)):
                name = rect.lower().rstrip(".")
                side_line = f"{rect} {side} {end}"
                landmarks[f"{name} {side} {end} at goal line"] = (
                    side_line, goal_line, (goal_x, y_sign * half_width))
                landmarks[f"{name} {side} {end} at front"] = (
                    side_line, f"{rect} {side} main", (sign * (HALF_LENGTH - depth), y_sign * half_width))
    landmarks["halfway top"] = ("Side line top", "Middle line", (0.0, -HALF_WIDTH))
    landmarks["halfway bottom"] = ("Side line bottom", "Middle line", (0.0, HALF_WIDTH))
    return landmarks


LANDMARKS = _build_landmarks()
LANDMARK_NAMES = list(LANDMARKS)  # index order used in the keypoint labels


def pitch_xy(name):
    """Template position of a landmark in meters."""
    return LANDMARKS[name][2]


def pitch_lines_px(annotation, width, height):
    """Pixel polylines {line name: N x 2 array} from one pitch annotation."""
    lines = {}
    for name, points in (annotation.get("lines") or {}).items():
        if len(points) >= 2:
            lines[name] = np.array([[p["x"] * width, p["y"] * height] for p in points], dtype=float)
    return lines


def point_segment_distance(point, start, end):
    """Distance in pixels from a point to the segment start-end."""
    segment = end - start
    length_sq = float(segment @ segment)
    if length_sq == 0:
        return float(np.linalg.norm(point - start))
    t = min(1.0, max(0.0, float((point - start) @ segment) / length_sq))
    return float(np.linalg.norm(point - (start + t * segment)))


def polyline_intersection(a, b, tolerance=3.0):
    """Where polyline a crosses polyline b, or None.

    Annotated lines that meet at a corner often end a pixel or two apart, so a crossing
    counts if the intersection of the two segments' lines lies within `tolerance` pixels
    of both segments. Near-parallel segments are skipped.
    """
    for i in range(len(a) - 1):
        p, r = a[i], a[i + 1] - a[i]
        for j in range(len(b) - 1):
            q, s = b[j], b[j + 1] - b[j]
            denominator = r[0] * s[1] - r[1] * s[0]
            if abs(denominator) < 0.05 * np.linalg.norm(r) * np.linalg.norm(s):
                continue
            t = ((q[0] - p[0]) * s[1] - (q[1] - p[1]) * s[0]) / denominator
            point = p + t * r
            if (point_segment_distance(point, a[i], a[i + 1]) <= tolerance
                    and point_segment_distance(point, b[j], b[j + 1]) <= tolerance):
                return point
    return None


def find_landmarks(lines, tolerance=3.0):
    """Pixel positions {landmark name: (x, y)} for every landmark whose two lines are visible."""
    found = {}
    for name, (line_a, line_b, _) in LANDMARKS.items():
        if line_a in lines and line_b in lines:
            point = polyline_intersection(lines[line_a], lines[line_b], tolerance)
            if point is not None:
                found[name] = (float(point[0]), float(point[1]))
    return found