#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import math
import time
import copy
import argparse
from collections import deque

from typing import Any, Tuple, List, Deque, Optional

import cv2
import vidstab  # type: ignore
import numpy as np


def get_args() -> Any:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--movie",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=str,
        default='output.mp4',
    )
    parser.add_argument(
        "--output_frame_width",
        type=int,
        default=1920,
    )
    parser.add_argument(
        "--smoothing_window",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    return args


def calc_table_size(length: int) -> Tuple[int, int]:
    sqrt_number = math.sqrt(length)

    if sqrt_number.is_integer():
        column_num = int(sqrt_number)
        row_num = int(sqrt_number)
    else:
        column_num = int(sqrt_number) + 1

        fractional_part = sqrt_number - int(sqrt_number)
        if fractional_part > 0.5:
            row_num = int(sqrt_number) + 1
        else:
            row_num = int(sqrt_number)

    return column_num, row_num


def main(
    movie_path: str,
    output_path: str,
    smoothing_window: int,
    kp_method_list: List[str],
    output_frame_width: int,
) -> None:
    # 各手法向けのデータ保持用のキューを生成
    stabilizer_list: List[Any] = []
    stabilized_frame_list: List[np.ndarray] = []
    elapsed_time_list: List[float] = []
    for kp_method in kp_method_list:
        stabilizer_list.append(vidstab.VidStab(kp_method=kp_method))
        elapsed_time_list.append(0)
        stabilized_frame_list.append(np.array([]))

    # 動画ファイルを準備
    video_capture = cv2.VideoCapture(movie_path)
    video_writer = None

    frame_count: int = 0
    frame_queue: Deque[Any] = deque(maxlen=smoothing_window)
    while True:
        # フレーム読み込み
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1

        # ブレ補正を実施
        for index, stabilizer in enumerate(stabilizer_list):
            start_time = time.time()
            stabilized_frame = stabilizer.stabilize_frame(
                input_frame=frame,
                smoothing_window=smoothing_window,
            )
            elapsed_time = time.time() - start_time

            stabilized_frame_list[index] = stabilized_frame
            elapsed_time_list[index] = elapsed_time

        frame_queue.append(frame)
        if frame_count <= smoothing_window:
            continue

        # 描画
        debug_image = draw_debug_info(
            frame_queue[0],
            kp_method_list,
            stabilized_frame_list,
            elapsed_time_list,
            smoothing_window,
            output_frame_width,
        )

        # 動画書き込み
        if video_writer is None and debug_image is not None:
            # VideoWriter生成
            debug_width = debug_image.shape[1]
            debug_height = debug_image.shape[0]
            video_writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
                video_capture.get(cv2.CAP_PROP_FPS),
                (debug_width, debug_height),
            )
        if video_writer is not None:
            video_writer.write(debug_image)

        # デバッグ表示
        cv2.imshow('VidStab', debug_image)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    video_capture.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()


def draw_debug_info(
    frame: np.ndarray,
    kp_method_list: List[str],
    stabilized_frame_list: List[np.ndarray],
    elapsed_time_list: List[float],
    smoothing_window: int,
    output_frame_width: int,
):
    debug_image: Optional[np.ndarray] = None

    column_num, row_num = calc_table_size(len(kp_method_list) + 1)
    resize_width = int(output_frame_width / column_num)
    image_width, image_height = frame.shape[1], frame.shape[0]
    resize_height = int(image_height * (resize_width / image_width))

    row_image: Any = None
    for row_index in range(row_num):
        column_image: Any = None
        for column_index in range(column_num):
            index = (column_index + (row_index * column_num))
            if index <= len(kp_method_list):
                if column_image is None:
                    if index == 0:
                        column_image = draw_analysis_info(
                            frame,
                            resize_width,
                            resize_height,
                            'Original',
                            None,
                            None,
                        )
                    else:
                        column_image = draw_analysis_info(
                            stabilized_frame_list[index - 1],
                            resize_width,
                            resize_height,
                            kp_method_list[index - 1],
                            elapsed_time_list[index - 1],
                            smoothing_window,
                        )
                else:
                    temp_image = draw_analysis_info(
                        stabilized_frame_list[index - 1],
                        resize_width,
                        resize_height,
                        kp_method_list[index - 1],
                        elapsed_time_list[index - 1],
                        smoothing_window,
                    )
                    column_image = cv2.hconcat([column_image, temp_image])
            else:
                black_image = np.zeros(
                    (resize_height, resize_width, 3),
                    np.uint8,
                )
                column_image = cv2.hconcat([column_image, black_image])
        if row_image is None:
            row_image = copy.deepcopy(column_image)
        else:
            row_image = cv2.vconcat([row_image, column_image])
    debug_image = row_image

    return debug_image


def draw_analysis_info(
    image: np.ndarray,
    resize_width: int,
    resize_height: int,
    kp_method: str,
    elapsed_time: float | None,
    smoothing_window: int | None,
):
    # 枠線
    temp_image = cv2.resize(image, (resize_width, resize_height))
    cv2.rectangle(
        temp_image,
        (0, 0),
        (resize_width - 1, resize_height - 1),
        (255, 255, 255),
        1,
    )

    # キーポイント抽出手法、処理時間、平滑窓数
    text = kp_method
    if elapsed_time is not None:
        text += ':' + '{:.1f}'.format(elapsed_time * 1000) + "ms"
    temp_image = cv2.putText(
        temp_image,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        thickness=2,
    )

    if smoothing_window is not None:
        temp_image = cv2.putText(
            temp_image,
            'smoothing_window:' + str(smoothing_window),
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            thickness=2,
        )

    return temp_image


if __name__ == '__main__':
    # 引数解析
    args = get_args()
    movie_path = args.movie
    output_path = args.output
    output_frame_width = args.output_frame_width

    smoothing_window = args.smoothing_window

    # 動画パス未指定時：サンプル動画ダウンロード
    if movie_path is None:
        if not os.path.exists('ostrich.mp4'):
            print("Download : ostrich.mp4")
            vidstab.download_ostrich_video('ostrich.mp4')
        movie_path = 'ostrich.mp4'

    # 比較するキーポイント抽出方法を追加
    kp_method_list = []
    kp_method_list.append('GFTT')
    kp_method_list.append('BRISK')
    kp_method_list.append('DENSE')
    kp_method_list.append('FAST')
    # kp_method_list.append('HARRIS')
    kp_method_list.append('MSER')
    kp_method_list.append('ORB')
    kp_method_list.append('STAR')
    # # kp_method_list.append('SURF')
    # # kp_method_list.append('SIFT')

    main(
        movie_path,
        output_path,
        smoothing_window,
        kp_method_list,
        output_frame_width,
    )
