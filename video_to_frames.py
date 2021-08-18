import cv2


# write a function that takes in a video path and saves each frame of the video as an image
def video_to_images(video_path):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        success, image = vidcap.read()
        cv2.imwrite(
            "./OUTPUT_FRAMES/frame%d.jpg" % count, image
        )  # save frame as JPEG file
        count += 1


# video_to_images("nasa.mp4")
video_to_clips("nasa.mp4")
