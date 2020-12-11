from vidgear.gears import WriteGear
import cv2
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
# define (Codec,CRF,preset) FFmpeg tweak parameters for writer
output_params = {"-vcodec": "libx264", "-crf": 0, "-preset": "fast"}

# Open live webcam video stream on first index(i.e. 0) device
stream = cv2.VideoCapture('rtsp://192.168.1.130:5554/playlist.m3u')

writer = WriteGear(output_filename='test/output.m3u8', compression_mode=True, logging=True,
                   **output_params)  # Define writer with output filename 'Output.mp4'

# infinite loop
while True:

    (grabbed, frame) = stream.read()
    # read frames

    # check if frame empty

    # {do something with frame here}
    if grabbed:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('dsd',frame)
    # write a modified frame to writer
        writer.write(gray)
    else:
        break

        # Show output window
    # cv2.imshow("Output Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    # check for 'q' key-press
    if key == ord("q"):
        # if 'q' key-pressed break out
        break

cv2.destroyAllWindows()
# close output window

stream.release()
# safely close video stream
writer.close()
# safely close writer