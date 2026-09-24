import cv2
import numpy as np
import os

os.makedirs('sample_data', exist_ok=True)
output_path = 'sample_data/sample_surveillance.mp4'

width, height = 640, 480
fps = 20
duration_s = 5
total_frames = fps * duration_s

# Use mp4v codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

for i in range(total_frames):
    # Create hallway/corridor background
    frame = np.full((height, width, 3), 45, dtype=np.uint8)
    
    # Perspective corridor lines
    cv2.line(frame, (0, 0), (220, 180), (70, 70, 70), 2)
    cv2.line(frame, (width, 0), (420, 180), (70, 70, 70), 2)
    cv2.line(frame, (0, height), (220, 360), (70, 70, 70), 2)
    cv2.line(frame, (width, height), (420, 360), (70, 70, 70), 2)
    cv2.rectangle(frame, (220, 180), (420, 360), (60, 60, 60), 2)
    
    # Floor tile grid
    for y_floor in [380, 410, 445]:
        cv2.line(frame, (50, y_floor), (590, y_floor), (55, 55, 55), 1)

    # Simulated person / moving subject walking from left to right
    progress = i / total_frames
    person_x = int(120 + progress * 380)
    person_y = int(220 + np.sin(progress * 10) * 4)
    person_w, person_h = 36, 110

    # Person silhouette
    cv2.rectangle(frame, (person_x, person_y + 30), (person_x + person_w, person_y + person_h), (30, 30, 30), -1)
    cv2.circle(frame, (person_x + person_w // 2, person_y + 15), 14, (30, 30, 30), -1)
    
    # Surveillance bounding box around target
    cv2.rectangle(frame, (person_x - 8, person_y - 6), (person_x + person_w + 8, person_y + person_h + 6), (0, 220, 130), 2)
    cv2.putText(frame, "TARGET: PERSON [0.94]", (person_x - 8, person_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 130), 1, cv2.LINE_AA)

    # Surveillance OSD (On Screen Display)
    cv2.putText(frame, "ARGUS SEC-NET // CAM-02 NORTH CORRIDOR", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 180), 1, cv2.LINE_AA)
    
    time_str = f"2026-09-24  01:{10 + i // (fps * 60):02d}:{(i // fps) % 60:02d}.{int((i % fps) * 50):03d} UTC"
    cv2.putText(frame, time_str, (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    
    cv2.putText(frame, f"REC [LIVE]  FRAME: {i+1:04d}/{total_frames}  FPS: {fps}", (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1, cv2.LINE_AA)

    # Frame scanlines / noise simulation
    noise = np.random.randint(0, 8, (height, width, 3), dtype=np.uint8)
    frame = cv2.add(frame, noise)

    out.write(frame)

out.release()
print(f"Generated sample surveillance video at {output_path} ({os.path.getsize(output_path)} bytes).")
