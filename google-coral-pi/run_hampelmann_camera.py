# Original Copyright 2019 Google LLC
# Adapted pose_camera.py at Saar Hackathon November 2019 https://www.what-the-hack.saarland/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from functools import partial
import time

import svgwrite
from svgwrite.image import Image as SVGImage
import gstreamer

from pose_engine import PoseEngine

EDGES = (
    ('nose', 'left eye'),
    ('nose', 'right eye'),
    ('nose', 'left ear'),
    ('nose', 'right ear'),
    ('left ear', 'left eye'),
    ('right ear', 'right eye'),
    ('left eye', 'right eye'),
    ('left shoulder', 'right shoulder'),
    ('left shoulder', 'left elbow'),
    ('left shoulder', 'left hip'),
    ('right shoulder', 'right elbow'),
    ('right shoulder', 'right hip'),
    ('left elbow', 'left wrist'),
    ('right elbow', 'right wrist'),
    ('left hip', 'right hip'),
    ('left hip', 'left knee'),
    ('right hip', 'right knee'),
    ('left knee', 'left ankle'),
    ('right knee', 'right ankle'),
)

def shadow_text(dwg, x, y, text, font_size=16):
    dwg.add(dwg.text(text, insert=(x + 1, y + 1), fill='black',
                     font_size=font_size, style='font-family:sans-serif'))
    dwg.add(dwg.text(text, insert=(x, y), fill='white',
                     font_size=font_size, style='font-family:sans-serif'))

# dwg: svg_canvas, pose
def draw_pose(dwg, pose, color='yellow', threshold=0.2):
    xys = {}
    for label, keypoint in pose.keypoints.items():
        if keypoint.score < threshold: continue
        xys[label] = (int(keypoint.yx[1]), int(keypoint.yx[0]))
        dwg.add(dwg.circle(center=(int(keypoint.yx[1]), int(keypoint.yx[0])), r=5,
                           fill='cyan', fill_opacity=keypoint.score, stroke=color))

    xysNose = xys.get("nose")
    xysLeftEar = xys.get("left ear")
    xysRightEar = xys.get("right ear")
    if not (xysNose is None or xysLeftEar is None or xysRightEar is None):
        xLeftEar = xysLeftEar[1]
        xRightEar = xysRightEar[1]
        dxEars = abs(xLeftEar - xRightEar)
        dwg.add(dwg.circle(center=(xysNose[0], xysNose[1]), r=int(dxEars/1.4),
                               fill='red', fill_opacity=1.0, stroke=color))

#        dwg.add(dwg.ellipse(center=(xysLeftEar[0], xysLeftEar[1]), r=(int(dxEars), int(dxEars)*1.5),
#                           fill='brown', fill_opacity=1.0, stroke=color))

#        dwg.add(dwg.ellipse(center=(xysRightEar[0], xysRightEar[1]), r=(int(dxEars), int(dxEars)*1.5),
#                            fill='brown', fill_opacity=1.0, stroke=color))

    for a, b in EDGES:
        if a not in xys or b not in xys: continue

        if a == "nose" or b == "nose": continue

        ax, ay = xys[a]
        bx, by = xys[b]
        dwg.add(dwg.line(start=(ax, ay), end=(bx, by), stroke=color, stroke_width=2))

#    if not xysNose is None:
#        noseImage = SVGImage("Nose.png", insert=(xysNose[0], xysNose[1]), size=(64, 64))
#        dwg.add(noseImage)

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

    print('Loading model: ', model)
    engine = PoseEngine(model, mirror=args.mirror)
    gstreamer.run_pipeline(partial(callback, engine),
                           src_size, appsink_size,
                           use_appsrc=use_appsrc, mirror=args.mirror,
                           videosrc=args.videosrc, h264input=args.h264)


def main():
    last_time = time.monotonic()
    n = 0
    sum_fps = 0
    sum_process_time = 0
    sum_inference_time = 0

    def render_overlay(engine, image, svg_canvas):
        nonlocal n, sum_fps, sum_process_time, sum_inference_time, last_time
        start_time = time.monotonic()
        outputs, inference_time = engine.DetectPosesInImage(image)
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