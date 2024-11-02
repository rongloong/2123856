import cv2

import numpy as np


def save_video(
    frames: np.array,
    output_path: str,
    fps: float = 30.0,
    codec: str = 'mp4v'
) -> None:
    assert frames is not None and len(frames) > 0, "The frame list is empty, cannot save the video."

    # Check if the frames are grayscale (single channel)
    if len(frames[0].shape) == 2 or frames[0].shape[2] == 1:
        is_color = False
        height, width = frames[0].shape[:2]
    else:
        is_color = True
        height, width = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*codec)  # noqa
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), is_color)

    for frame in frames:
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()
    print(f"Video has been successfully saved to {output_path}")
