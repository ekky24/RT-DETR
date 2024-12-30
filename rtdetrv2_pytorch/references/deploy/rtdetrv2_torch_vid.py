"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import sys
sys.path.append('./')

import torch
import torch.nn as nn 
import torchvision.transforms as T

import numpy as np 
from PIL import Image, ImageDraw
import cv2
import os
import time

from src.core import YAMLConfig
import deployment_config


def draw(images, labels, boxes, scores, pred_type, fps, thrh = 0.6):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]

        for j,b in enumerate(box):
            lab_name = deployment_config.CLASS_NAME[lab[j].item()]
            if lab_name == pred_type:
                draw.rectangle(list(b), outline='red', width=2)
                draw.text((b[0], b[1]), text=f"{lab_name} {round(scrs[j].item(),2)}", fill='blue', )

        image_rgb = np.array(im)
        image_cv2 = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Add a white box and FPS text to the frame
        text = f"FPS: {fps:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        color = (0, 255, 0)  # Green color for text
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        box_coords = (10, 10, 10 + text_size[0] + 10, 10 + text_size[1] + 10)  # (x1, y1, x2, y2)
        cv2.rectangle(image_cv2, (box_coords[0], box_coords[1]), (box_coords[2], box_coords[3]), (255, 255, 255), -1)
        text_position = (box_coords[0] + 5, box_coords[3] - 5)
        cv2.putText(image_cv2, text, text_position, font, font_scale, color, thickness, cv2.LINE_AA)

        return image_cv2


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)

    cap = cv2.VideoCapture(args.vid_file)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    save_path = f'output/{os.path.basename(args.vid_file)}'
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (int(w), int(h)))

    fps = 0
    prev_time = time.time()
        
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            # image preprocessing
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(frame)

            w, h = im_pil.size
            orig_size = torch.tensor([w, h])[None].to(args.device)

            transforms = T.Compose([
                T.Resize((640, 640)),
                T.ToTensor(),
            ])
            im_data = transforms(im_pil)[None].to(args.device)

            output = model(im_data, orig_size)
            labels, boxes, scores = output

            # Calculate FPS
            current_time = time.time()
            elapsed_time = current_time - prev_time
            prev_time = current_time
            if elapsed_time > 0:
                fps = 1 / elapsed_time

            result_frame = draw([im_pil], labels, boxes, scores, pred_type=args.pred_type, fps=fps)
            vid_writer.write(result_frame)

            cv2.namedWindow("DAMO-YOLO", cv2.WINDOW_NORMAL)
            cv2.imshow("DAMO-YOLO", result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                vid_writer.release()
                break

        else:
            vid_writer.release()
            break


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, )
    parser.add_argument('-r', '--resume', type=str, )
    parser.add_argument('-f', '--vid-file', type=str, )
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-t', '--pred-type', type=str, default='car')
    args = parser.parse_args()
    main(args)
