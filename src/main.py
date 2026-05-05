import cv2
import mediapipe as mp
import numpy as np
import argparse
import json
import os

# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_pose     = mp.solutions.pose
mp_drawing  = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles

# ── Landmark indices we care about for shooting form ────────────────────────
LANDMARKS = {
    "left_shoulder":  mp_pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "left_elbow":     mp_pose.PoseLandmark.LEFT_ELBOW,
    "right_elbow":    mp_pose.PoseLandmark.RIGHT_ELBOW,
    "left_wrist":     mp_pose.PoseLandmark.LEFT_WRIST,
    "right_wrist":    mp_pose.PoseLandmark.RIGHT_WRIST,
    "left_hip":       mp_pose.PoseLandmark.LEFT_HIP,
    "right_hip":      mp_pose.PoseLandmark.RIGHT_HIP,
    "left_knee":      mp_pose.PoseLandmark.LEFT_KNEE,
    "right_knee":     mp_pose.PoseLandmark.RIGHT_KNEE,
    "left_ankle":     mp_pose.PoseLandmark.LEFT_ANKLE,
    "right_ankle":    mp_pose.PoseLandmark.RIGHT_ANKLE,
}

# ── Ideal angle ranges for a correct shooting form ──────────────────────────
# These are biomechanically validated ranges used by NBA coaches.
# We'll use these in Phase 2 for feedback generation.
IDEAL_RANGES = {
    "shooting_elbow_at_set":  (85,  95),   # elbow ~90° at set point
    "shooting_elbow_release": (160, 180),  # elbow nearly fully extended at release
    "knee_bend_at_jump":      (100, 140),  # slight knee bend before jumping
    "wrist_follow_through":   (55,  75),   # wrist snap downward after release
}


# ── Geometry helpers ─────────────────────────────────────────────────────────
def get_coords(landmarks, name, w, h):
    """Return (x, y) pixel coordinates for a named landmark."""
    lm = landmarks[LANDMARKS[name].value]
    return int(lm.x * w), int(lm.y * h)


def compute_angle(a, b, c):
    """
    Compute the angle (degrees) at point B formed by the vectors B->A and B->C.
    a, b, c are (x, y) tuples.
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def extract_shooting_angles(landmarks, w, h, shooting_side="right"):
    """
    Extract the four key angles for shooting form analysis.
    Returns a dict of angle_name -> degrees (float).
    shooting_side: 'right' or 'left' depending on player's shooting hand.
    """
    s = shooting_side  # shorthand

    shoulder = get_coords(landmarks, f"{s}_shoulder", w, h)
    elbow    = get_coords(landmarks, f"{s}_elbow",    w, h)
    wrist    = get_coords(landmarks, f"{s}_wrist",    w, h)
    hip      = get_coords(landmarks, f"{s}_hip",      w, h)
    knee     = get_coords(landmarks, f"{s}_knee",     w, h)
    ankle    = get_coords(landmarks, f"{s}_ankle",    w, h)

    return {
        "elbow_angle":  round(compute_angle(shoulder, elbow, wrist), 1),
        "knee_angle":   round(compute_angle(hip, knee, ankle), 1),
        "shoulder_angle": round(compute_angle(elbow, shoulder, hip), 1),
        # wrist angle requires more points; approximated via elbow-wrist vertical
        "wrist_elevation": round(abs(wrist[1] - elbow[1]), 1),  # pixel diff, used in Phase 2
    }


# ── Overlay helpers ──────────────────────────────────────────────────────────
def draw_angle_label(frame, point, angle, label, color=(0, 255, 150)):
    """Draw angle value next to a joint on the frame."""
    x, y = point
    text = f"{label}: {angle:.1f}"
    cv2.putText(frame, text, (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_hud(frame, angles, frame_num, fps):
    """Draw a HUD panel with all angle values in the top-left corner."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent dark box
    cv2.rectangle(overlay, (10, 10), (310, 140), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "SHOOTING FORM ANALYZER", (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)

    y = 55
    for name, val in angles.items():
        cv2.putText(frame, f"{name.replace('_', ' ').title()}: {val}", (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1)
        y += 20

    cv2.putText(frame, f"Frame: {frame_num}  |  {fps:.1f} fps", (18, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)


# ── Core processing function ─────────────────────────────────────────────────
def process_video(source, output_path=None, shooting_side="right", save_json=True):
    """
    Process a video file or webcam feed.
    source: path string OR 0 for webcam.
    Returns list of per-frame angle dicts (saved to JSON if save_json=True).
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Cannot open source: {source}")

    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_src, (w, h))

    all_frames_data = []
    frame_num = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,          # 0=fast, 1=balanced, 2=accurate
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            angles = {}
            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark

                # Draw full skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )

                # Extract angles
                angles = extract_shooting_angles(lms, w, h, shooting_side)

                # Annotate key joints with their angles
                elbow_pt = get_coords(lms, f"{shooting_side}_elbow", w, h)
                knee_pt  = get_coords(lms, f"{shooting_side}_knee",  w, h)
                draw_angle_label(frame, elbow_pt, angles["elbow_angle"],  "Elbow")
                draw_angle_label(frame, knee_pt,  angles["knee_angle"],   "Knee", color=(0, 200, 255))

            draw_hud(frame, angles, frame_num, fps_src)
            all_frames_data.append({"frame": frame_num, "angles": angles})

            if writer:
                writer.write(frame)

            # Show live (press Q to quit)
            cv2.imshow("Basketball Form Analyzer - Phase 1", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Save angle data as JSON for Phase 2 ingestion
    if save_json and all_frames_data:
        json_path = (output_path or "output").replace(".mp4", "") + "_angles.json"
        with open(json_path, "w") as f:
            json.dump(all_frames_data, f, indent=2)
        print(f"\n✅ Angle data saved to: {json_path}")

    print(f"\n📊 Processed {frame_num} frames.")
    return all_frames_data


# ── Quick stats on extracted data ────────────────────────────────────────────
def summarize_angles(frames_data):
    """Print average angles across all detected frames."""
    keys = ["elbow_angle", "knee_angle", "shoulder_angle", "wrist_elevation"]
    collected = {k: [] for k in keys}

    for fd in frames_data:
        for k in keys:
            v = fd["angles"].get(k)
            if v is not None:
                collected[k].append(v)

    print("\n── Average Angles Across Session ──────────────────")
    for k, vals in collected.items():
        if vals:
            print(f"  {k.replace('_', ' ').title():25s}: {np.mean(vals):6.1f}°  "
                  f"(min {np.min(vals):.1f}° / max {np.max(vals):.1f}°)")
    print("────────────────────────────────────────────────────\n")


# ── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basketball Shooting Form - Phase 1 Pose Extractor")
    parser.add_argument("--input",  type=str, help="Path to input video file")
    parser.add_argument("--output", type=str, default=None, help="Path to save annotated output video")
    parser.add_argument("--webcam", action="store_true", help="Use webcam as input (overrides --input)")
    parser.add_argument("--side",   type=str, default="right", choices=["left", "right"],
                        help="Shooting hand side (default: right)")
    args = parser.parse_args()

    source = 0 if args.webcam else args.input
    if source is None:
        parser.error("Provide --input <video> or use --webcam")

    print(f"\n🏀 Basketball Form Analyzer — Phase 1")
    print(f"   Source : {'webcam' if args.webcam else args.input}")
    print(f"   Side   : {args.side} hand\n")

    data = process_video(source, args.output, shooting_side=args.side)
    summarize_angles(data)