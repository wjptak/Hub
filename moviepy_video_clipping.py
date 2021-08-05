from moviepy.editor import *


def clip(src_path, dest_dir, seconds_per_clip: int):
    video = VideoFileClip(src_path)
    nclips = int(video.duration / seconds_per_clip)

    print(f"breaking into {nclips} clips")
    for i in range(nclips):
        print(f"creating clip {i}/{nclips}...")
        dest_path = os.path.join(dest_dir, 'clip_{}.mp4'.format(i))
        sub = video.subclip(i * seconds_per_clip, (i + 1) * seconds_per_clip)
        sub.write_videofile(dest_path)

clip("nasa.mp4", "OUTPUT_CLIPS", 10)