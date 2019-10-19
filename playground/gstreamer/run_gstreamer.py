import argparse
from functools import partial
import re
import time

import numpy as np
from PIL import Image
import svgwrite
import gstreamer

def shadow_text(dwg, x, y, text, font_size=16):
    dwg.add(dwg.text(text, insert=(x + 1, y + 1), fill='black',
                     font_size=font_size, style='font-family:sans-serif'))
    dwg.add(dwg.text(text, insert=(x, y), fill='white',
                     font_size=font_size, style='font-family:sans-serif'))

def draw_pose(dwg, pose, color='yellow', threshold=0.2):
    """
    :param dwg: svgwrite.drawing.Drawing
    :param pose:
    :param color:
    :param threshold:
    :return:
    """
    pass

def run(callback, use_appsrc=False):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mirror', help='flip video horizontally', action='store_true')
    parser.add_argument('--model', help='.tflite model path.', required=False)
    parser.add_argument('--res', help='Resolution', default='640x480',
                        choices=['480x360', '640x480', '1280x720'])
    parser.add_argument('--videosrc', help='Which video source to use', default='/dev/video0')
    parser.add_argument('--h264', help='Use video/x-h264 input', action='store_true')
    args = parser.parse_args()

    default_model = 'models/posenet_mobilenet_v1_075_%d_%d_quant_decoder_edgetpu.tflite'
    if args.res == '480x360':
        src_size = (640, 480)
        appsink_size = (480, 360)
        model = args.model or default_model % (353, 481)
    elif args.res == '640x480':
        src_size = (640, 480)
        appsink_size = (640, 480)
        model = args.model or default_model % (481, 641)
    elif args.res == '1280x720':
        src_size = (1280, 720)
        appsink_size = (1280, 720)
        model = args.model or default_model % (721, 1281)

#    print('Loading model: ', model)
#    engine = PoseEngine(model, mirror=args.mirror)
    gstreamer.run_pipeline(callback,
                           src_size, appsink_size,
                           use_appsrc=use_appsrc, mirror=args.mirror,
                           videosrc=args.videosrc, h264input=args.h264)

def main():
    last_time = time.monotonic()
    n = 0
    sum_fps = 0
    sum_process_time = 0
    sum_inference_time = 0

    def render_overlay(image, svg_canvas):
        nonlocal n, sum_fps, sum_process_time, sum_inference_time, last_time
        start_time = time.monotonic()

        # outputs, inference_time = engine.DetectPosesInImage(image)
        inference_time = 100
        outputs = []

        end_time = time.monotonic()
        n += 1
        sum_fps += 1.0 / (end_time - last_time)
        sum_process_time += 1000 * (end_time - start_time) - inference_time
        sum_inference_time += inference_time
        last_time = end_time
        text_line = 'PoseNet: %.1fms Frame IO: %.2fms TrueFPS: %.2f Nposes %d' % (
            sum_inference_time / n, sum_process_time / n, sum_fps / n, len(outputs)
        )
        print(text_line)

        shadow_text(svg_canvas, 10, 20, text_line)
        for pose in outputs:
            draw_pose(svg_canvas, pose)

    run(render_overlay)


if __name__ == '__main__':
    main()