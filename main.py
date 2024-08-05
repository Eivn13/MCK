import time
import multiprocessing
import os
import re
import pandas as pd
import numpy as np
import cv2 as cv
import demucs.api
import torchaudio
from deepface import DeepFace

from moviepy.video.io.VideoFileClip import VideoFileClip


def natural_sort_key(s):
    """Key function for natural sorting."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def find_data_paths(data_path, formats):
    """
    Gets path and finds all .mp4, .mkv and .mov files recursively, whilst sorting these files naturally.
    Returns DataFrame with two columns, path and name.
    """
    path = []
    name = []
    for dirpath, _, filenames in sorted(os.walk(data_path), key=lambda x: natural_sort_key(x[0])):
        for filename in sorted(filenames, key=natural_sort_key):
            if filename.lower().endswith(formats):
                whole_path = os.path.join(dirpath, filename)
                path.append(whole_path)
                name.append(filename[:-4])

    df = pd.DataFrame({'Path': path, 'Name': name})
    return df


def subclip_task(df, worker_id):
    """
    This function generates 2.4 second subclips from each clip, then saves the subclips into a new folder named
    chunks_dataset. It also saves the first image of the subclip into folder named preprocessed_data.
    """
    chunk_duration = 2.4
    len_df = len(df)
    progress_file_path = f"outputs/progress_file_{worker_id}.txt"

    if not os.path.exists(progress_file_path):
        with open(progress_file_path, "w") as progress_file:
            progress_file.write("")
    try:
        progress_file = open(progress_file_path, 'r')
        list_of_done_file_paths = progress_file.read().splitlines()
        progress_file.close()
    except FileExistsError:
        return "Progress file could not be created."

    progress_file = open(progress_file_path, 'a')

    for x, (index, row) in enumerate(df.iterrows()):
        uri = row['Path']
        name = row['Name']

        if uri in list_of_done_file_paths:
            continue

        save_dir_end_list = uri.split("/")[1:-1]
        save_dir_end = "/".join(save_dir_end_list)

        save_path = os.path.join("chunks_dataset",
                                 save_dir_end)
        os.makedirs(save_path, exist_ok=True)

        preprocess_path = os.path.join("preprocessed_data",
                                       save_dir_end)
        os.makedirs(preprocess_path, exist_ok=True)

        video_clip = VideoFileClip(uri)
        num_of_full_chunks = int(video_clip.duration / chunk_duration)
        video_chunks = [video_clip.subclip(i * chunk_duration, (i + 1) * chunk_duration) for i in
                        range(num_of_full_chunks)]

        for i, chunk in enumerate(video_chunks):
            chunk.write_videofile(f"{save_path}/{name}_chunk_{i + 1}.mp4", logger=None)
            video_frame = chunk.get_frame(0)
            cv.imwrite(f"{preprocess_path}/{name}_chunk_{i + 1}_visual.png",
                       cv.cvtColor(video_frame, cv.COLOR_BGR2RGB))
            print(f"Worker {worker_id} has preprocessed {i + 1} out of {num_of_full_chunks} video chunks.")

        progress_file.write(uri + "\n")
        progress_file.flush()
        print(f"Worker {worker_id} has preprocessed {x + 1} out of {len_df} videos.")


def create_subclips_and_raw_images(data):
    start_time_all = time.time()
    num_workers = 10

    rows_per_part, remainder = divmod(len(data), num_workers)
    split_df = np.array_split(data, num_workers)
    if remainder != 0:
        data_remainder = data.iloc[-remainder:]
        split_df[-1] = pd.concat([split_df[-1], data_remainder])

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = [pool.apply_async(subclip_task, args=(df, worker_id)) for worker_id, df in enumerate(split_df)]

        final_results = [result.get() for result in results]
    print(f"Subclips and raw images function took {time.time() - start_time_all} seconds.")
    print("Final results: ", final_results)


def audio_task(df, worker_id):
    separator = demucs.api.Separator(model="mdx_extra")
    len_df = len(df)
    progress_file_path = f"outputs/audio_progress_file_{worker_id}.txt"

    if not os.path.exists(progress_file_path):
        with open(progress_file_path, "w") as progress_file:
            progress_file.write("")
    try:
        progress_file = open(progress_file_path, 'r')
        list_of_done_file_paths = progress_file.read().splitlines()
        progress_file.close()
    except FileExistsError:
        return "Progress file could not be created."

    progress_file = open(progress_file_path, 'a')

    for x, (index, row) in enumerate(df.iterrows()):
        uri = row['Path']

        if uri in list_of_done_file_paths:
            continue

        save_dir_end_list = uri.split("/")[1:]
        name = save_dir_end_list[-1][:-4]
        del save_dir_end_list[-1]

        save_dir_end = "/".join(save_dir_end_list)

        preprocess_path = os.path.join("preprocessed_data",
                                       save_dir_end)
        os.makedirs(preprocess_path, exist_ok=True)

        video_clip = VideoFileClip(uri)
        temp_audio_path = f"outputs/temp_audio_file_{worker_id}.wav"
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(temp_audio_path, logger=None)
        waveform, sample_rate = torchaudio.load(f"outputs/temp_audio_file_{worker_id}.wav")
        origin, separated = separator.separate_tensor(waveform)
        torchaudio.save(uri=f'{preprocess_path}/{name}_audio.wav', src=separated['vocals'], sample_rate=44100)

        progress_file.write(uri + "\n")
        progress_file.flush()
        print(f"Worker {worker_id} has preprocessed {x + 1} out of {len_df} chunks.")


def preprocess_audio(data):
    start_time_all = time.time()
    num_workers = 4

    rows_per_part, remainder = divmod(len(data), num_workers)
    split_df = np.array_split(data, num_workers)
    if remainder != 0:
        data_remainder = data.iloc[-remainder:]
        split_df[-1] = pd.concat([split_df[-1], data_remainder])

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = [pool.apply_async(audio_task, args=(df, worker_id)) for worker_id, df in enumerate(split_df)]

        final_results = [result.get() for result in results]
    print(f"Preprocess audio function took {time.time() - start_time_all} seconds.")
    print("Final results: ", final_results)


def image_task(df, worker_id):
    """
    This function gets "raw" image and detects face. Then does preprocessing with deepface library. Images without
    detected face are saved in outputs/failed_detections.txt
    """
    failed_detections = open(f"outputs/failed_detections_{worker_id}.txt", 'w')

    for img_path in df.Path:
        try:
            detection = DeepFace.extract_faces(img_path=img_path,
                                               detector_backend="yolov8",
                                               target_size=(224, 224))
            detection[0]['face'] *= 225
            # cv.imwrite(img_path, cv.cvtColor(detection[0]["face"], cv.COLOR_RGB2BGR))
        except ValueError:
            failed_detections.write(img_path + "\n")


def preprocess_images(data):
    start_time_all = time.time()
    num_workers = 10

    rows_per_part, remainder = divmod(len(data), num_workers)
    split_df = np.array_split(data, num_workers)
    if remainder != 0:
        data_remainder = data.iloc[-remainder:]
        split_df[-1] = pd.concat([split_df[-1], data_remainder])

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = [pool.apply_async(image_task, args=(df, worker_id)) for worker_id, df in enumerate(split_df)]

        final_results = [result.get() for result in results]
    print(f"Preprocess images function took {time.time() - start_time_all} seconds.")
    print("Final results: ", final_results)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    raw_data = find_data_paths("raw", ('.mp4', '.mkv', '.mov'))
    create_subclips_and_raw_images(raw_data)
    subclips = find_data_paths("chunks_dataset", ('.mp4', '.mkv', '.mov'))
    preprocess_audio(subclips)
    images = find_data_paths("/home/domino/dd/preprocessed_data", '.png')
    preprocess_images(images)
