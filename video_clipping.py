import cv2
import os


def get_video_shape(video_path):
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return n, w, h  # TODO: return channels?


def video_writer_like(src_path, dest_path):
    cap = cv2.VideoCapture(src_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    shape = get_video_shape(src_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(dest_path, fourcc, fps, shape[1:])


# write a function that takes in a video and creates a new file with the first 10 seconds
def clip_video(src_path, dest_dir, frames_per_clip: int):
    # TODO: docstring

    cap = cv2.VideoCapture(src_path)
    shape = get_video_shape(src_path)

    success, frame = cap.read()
    frame_count = 0
    clip_count = 0
    while success:
        if frame_count % frames_per_clip == 0:
            print(f"creating clip {clip_count} (frame {frame_count}/{shape[0]})...")
            dest_path = os.path.join(dest_dir, 'clip_{}.mp4'.format(clip_count))
            out = video_writer_like(src_path, dest_path)
            clip_count += 1
            # if clip_count > 3:
                # break

        frame_count += 1

        out.write(frame)
        success, frame = cap.read()
    cap.release()


print(get_video_shape("nasa.mp4"))
clip_video("nasa.mp4", "OUTPUT_CLIPS", 500)