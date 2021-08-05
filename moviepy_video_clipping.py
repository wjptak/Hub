from moviepy.editor import *


def clip(src_path, dest_dir, seconds_per_clip: int):
    """Saves the video at `src_path` as clips of max length `seconds_per_clip` inside `dest_dir`."""

    video = VideoFileClip(src_path)
    nclips = int(video.duration / seconds_per_clip)

    print(f"breaking into {nclips} clips")
    for i in range(nclips):
        print(f"creating clip {i}/{nclips}...")
        dest_path = os.path.join(dest_dir, 'clip_{}.mp4'.format(i))

        start = i * seconds_per_clip
        end =  min((i + 1) * seconds_per_clip, video.duration)

        sub = video.subclip(start, end)
        sub.write_videofile(dest_path)


def images(src_path, dest_dir):
    """Saves the video at `src_path` as single-frame images inside `dest_dir`."""

    video = VideoFileClip(src_path)
    dest_path = os.path.join(dest_dir, "frame%04d.jpeg")
    video.write_images_sequence(dest_path)

# clip("nasa.mp4", "OUTPUT_CLIPS", 90)
# images("nasa.mp4", "OUTPUT_FRAMES")