import cv2


def get_video_shape(video_path):
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    c = None  # TODO
    return n, h, w, c


# write a function that takes in a video and creates a new file with the first 10 seconds
def video_clipping(video_file, new_file):
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(new_file, fourcc, fps, size)
    success, frame = cap.read()
    count = 0
    while success:
        out.write(frame)
        success, frame = cap.read()
        count += 1
        if count > 10:
            break
    cap.release()


print(get_video_shape("nasa.mp4"))
video_clipping("nasa.mp4", "nasa_CLIP.mp4")