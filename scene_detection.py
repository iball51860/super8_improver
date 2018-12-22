import os
import time
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print('Starting project')
base_dir = '/Volumes/Samsung_T5/Super-8/suedfrankreich'
print(f'Project root is {base_dir}')

scans = [subdir.path for subdir in os.scandir(base_dir) if subdir.is_dir()]
print(f'Found {len(scans)} scans.')
print('\n'.join(scans))

out_dir = os.path.join(base_dir, 'output')
os.makedirs(out_dir)
print(f'writing output to {out_dir}')
out_dir_scan_by_scene = os.path.join(out_dir, 'scanned_scenes')

info_dicts = []

for scan_index in range(0, len(scans)):
    scan = scans[scan_index]
    scan_out_folder = os.path.join(out_dir_scan_by_scene, f'scan_{scan_index}')
    print(f'Rearranging {scan}')
    frames = []

    before = time.time()
    for dirpath, dirnames, filenames in os.walk(scan):
        frames += [os.path.join(dirpath, filename) for filename in [f for f in filenames if f.endswith(".jpg")]]
    after = time.time()

    print(f'Found {len(frames)} frames in {after - before} seconds.')

    before = time.time()
    frames.sort()
    after = time.time()
    print(f'Sorted {len(frames)} frames in {after - before} seconds.')

    scene_no = 0
    scene_frame_counter = -1
    cumulated_duplicates = 0
    cumulated_erroneous = 0
    input_start_frame = 0
    current_scene_folder = os.path.join(scan_out_folder, f'scene_{str(scene_no).zfill(3)}')
    os.makedirs(current_scene_folder)
    scene_dict = {
        'scan': scan_index,
        'scene': scene_no,
        'start_frame': 0,
        'rmse': np.nan,
        'rolling_mean_rmse': np.nan}

    lastDifferences = []
    lastFrame = np.zeros(cv2.imread(frames[0]).shape)
    diffs = []
    duplicates_paths = []
    erroneous_paths = []
    for i in range(0, len(frames)):
        scene_frame_counter += 1
        frameOK = True
        frameDup = False

        framePath = frames[i]
        frame = cv2.imread(framePath)
        diff = np.mean(np.sqrt((frame - lastFrame) ** 2))
        if diff == 0:
            print(f'duplicate frame {framePath}')
            frameOK = False
            frameDup = True
        elif len(lastDifferences) > 5:
            sensitivity = 6
            diffThreshold = 8.0  # np.mean(lastDifferences) + sensitivity * np.std(lastDifferences)
            significantDiff = diff > diffThreshold
            nextFrameAvailable = i + 1 < len(frames)

            if significantDiff and nextFrameAvailable:
                nextFrame = cv2.imread(frames[i + 1])
                diffNext = np.mean(np.sqrt((nextFrame - lastFrame) ** 2))

                if diffNext > diffThreshold:
                    new_duplicates = len(duplicates_paths) - cumulated_duplicates
                    new_erroneous = len(erroneous_paths) - cumulated_erroneous
                    old_scene_dict = {**scene_dict,
                                      'length': scene_frame_counter,
                                      'duplicates': new_duplicates,
                                      'erroneous': new_erroneous,
                                      'end_frame': i - 1}
                    info_dicts.append(old_scene_dict)

                    scene_no += 1
                    scene_frame_counter = 0
                    cumulated_duplicates = len(duplicates_paths)
                    cumulated_erroneous = len(erroneous_paths)
                    current_scene_folder = os.path.join(scan_out_folder, f'scene_{str(scene_no).zfill(3)}')
                    os.makedirs(current_scene_folder)
                    scene_dict = {
                        'scan': scan_index,
                        'scene': scene_no,
                        'start_frame': i,
                        'rmse': diff,
                        'rolling_mean_rmse': np.mean(lastDifferences),
                        'folder': current_scene_folder
                    }
                    print(f'new scene at frame {framePath}. '
                          f'Diff: {diff}, '
                          f'avg{len(lastDifferences)}: {np.mean(lastDifferences)}, '
                          f'{sensitivity} * stddev{len(lastDifferences)}: {sensitivity * np.std(lastDifferences)}')
                    lastDifferences = []
                else:
                    print(f'something seems wrong with frame {framePath}.')
                    frameOK = False

        if frameOK:
            diffs.append(diff)
            lastDifferences.append(diff)
            if len(lastDifferences) > 36:
                lastDifferences = lastDifferences[1:]
            lastFrame = frame

            frame_out_path = os.path.join(current_scene_folder, f'frame_{str(scene_frame_counter).zfill(5)}.jpg')
        elif frameDup:
            frame_out_path = os.path.join(current_scene_folder, f'frame_{str(scene_frame_counter).zfill(5)}_duplicate.jpg')
            duplicates_paths.append(frame_out_path)
        else:
            frame_out_path = os.path.join(current_scene_folder, f'frame_{str(scene_frame_counter).zfill(5)}_warning.jpg')
            erroneous_paths.append(frame_out_path)
        cv2.imwrite(frame_out_path, frame)

    print(f'duplicates in folder {scan_out_folder}', '\n'.join(duplicates_paths))
    print(f'erroneous in folder {scan_out_folder}', '\n'.join(erroneous_paths))

    plot = sns.lineplot([n for n in range(0, len(diffs) - 1)], diffs[1:])
    plt.show()

(pd.DataFrame(info_dicts)
 .set_index(['scene', 'scan'])
 .sort_index()
 .to_csv(
    os.path.join(out_dir, 'info.csv')))
