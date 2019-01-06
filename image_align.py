import os
import sys
from multiprocessing import Pool
import cv2
import numpy as np
import time


def alignAndStack(projectDir: str):
    inputDirectory = os.path.join(projectDir, 'output/scanned_scenes')
    if not os.path.exists(inputDirectory):
        raise ValueError('Project has no detected scenes yet.')

    scanDirectories = [subdir.path for subdir in os.scandir(inputDirectory) if os.path.isdir(subdir)]
    if len(scanDirectories) < 2:
        raise ValueError('Project has only one scan, image alignment not necessary.')

    sceneLengthInfos = [getSceneLengths(scanDir) for scanDir in scanDirectories]

    for i, sceneLengthInfoDict in enumerate(sceneLengthInfos[1:]):
        if sceneLengthInfoDict != sceneLengthInfos[0]:
            different = []
            for key in sceneLengthInfoDict.keys():
                if sceneLengthInfoDict[key] != sceneLengthInfos[0][key]:
                    different.append(f'scan0[{key}] - frames {sceneLengthInfos[0][key]}  |  {sceneLengthInfoDict[key]} frames - scan{i}[{key}]')
            raise ValueError('Scans seem to have different scenes or scene lengths.\n' + '\n'.join(different))

    sceneNames = list(sceneLengthInfos[0].keys())
    sceneNames.sort()
    for sceneName in sceneNames:
        alignAndStackScene(projectDir, scanDirectories, sceneName)


def getSceneLengths(scanDir):
    sceneSizes = {}
    for scenePath in [subdir.path for subdir in os.scandir(scanDir) if os.path.isdir(subdir.path)]:
        sceneLength = len([imagePath for imagePath in os.scandir(scenePath) if imagePath.name.endswith('.jpg')])
        sceneSizes[os.path.basename(scenePath)] = sceneLength
    return sceneSizes


def alignAndStackScene(projectDir, scanDirectories, sceneName):
    print(f'[{time.strftime("%H:%M:%S")}] aligning {sceneName}')
    framePaths = []
    for scanDir in scanDirectories:
        scanName = os.path.basename(scanDir)
        os.makedirs(os.path.join(projectDir, 'output/aligned', scanName, sceneName))
        os.makedirs(os.path.join(projectDir, 'output/average', sceneName), exist_ok=True)
        os.makedirs(os.path.join(projectDir, 'output/median', sceneName), exist_ok=True)
        sceneInputDir = os.path.join(scanDir, sceneName)
        framePathsInScanDir = [framePath.path for framePath in os.scandir(sceneInputDir) if framePath.path.endswith('.jpg')]
        framePathsInScanDir.sort()
        framePaths.append(framePathsInScanDir)
    framePaths = list(zip(*framePaths))
    try:
        pool = Pool()
        pool.map(process_frame, framePaths)
    finally:
        pool.close()
        pool.join()


def spreadHist(im: np.core.multiarray, grey: np.core.multiarray):
    maskedGray = grey[20:(grey.shape[0]-20), 20:(grey.shape[1]-20)]
    pixels = maskedGray.shape[0] * maskedGray.shape[1]
    leftThreshold = np.floor(pixels * 0.005)
    rightThreshold = pixels - leftThreshold
    hist = np.histogram(maskedGray, range(0, 257))
    cumHist = np.cumsum(hist[0])
    histMask = (cumHist > leftThreshold) & (cumHist < rightThreshold)
    existingValues = hist[1][:-1][histMask]
    lower = np.min(existingValues)
    upper = np.max(existingValues)
    alpha = 255 / (upper - lower)
    imMinBefore = np.min(im)
    imMaxBefore = np.max(im)
    lowerMask = im < lower
    upperMask = im > upper
    im = (im - lower) * alpha
    im[upperMask] = 255
    im[lowerMask] = 0
    im = np.floor(im)
    return im


def process_frame(framePaths: tuple):
    width, height = (1204, 885)
    # Specify the number of iterations.
    number_of_iterations = 50
    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 5e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    stack = np.array([cv2.imread(path) for path in framePaths])
    stack_grey = np.array([cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in stack])
    stack = np.array([spreadHist(im, grey) for im, grey in zip(stack, stack_grey)])

    ccs = []
    for layer_index, im_gray in enumerate(stack_grey[1:], 1):
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC(stack_grey[0], im_gray, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria)
        ccs.append(cc)

        stack[layer_index] = cv2.warpAffine(
            stack[layer_index],
            warp_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    scaled_stack = [cv2.resize(im, (width*2, height*2), interpolation=cv2.INTER_LANCZOS4) for im in stack]

    projectPath, sceneName, frameName = '_'*3
    for i, im in enumerate(stack):
        scenePath, frameName = os.path.split(framePaths[i])
        scanPath, sceneName = os.path.split(scenePath)
        sceneDetectionPath, scanName = os.path.split(scanPath)
        projectPath = os.path.split(os.path.split(sceneDetectionPath)[0])[0]
        cv2.imwrite(os.path.join(projectPath, 'output/aligned', scanName, sceneName, f'aligned_{frameName}'), im)

    median_image = np.median(scaled_stack, axis=0)
    cv2.imwrite(os.path.join(projectPath, 'output/median', sceneName, frameName), median_image)

    average_image = np.mean(scaled_stack, axis=0)
    cv2.imwrite(os.path.join(projectPath, 'output/average', sceneName, frameName), average_image)
