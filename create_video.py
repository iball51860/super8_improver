import cv2
import os


def createVideos(inputScenesDir: str):
    outDirectory = os.path.split(inputScenesDir)[0]
    videoOutDir = os.path.join(outDirectory, 'video')
    os.makedirs(videoOutDir)
    for scene in [subdir for subdir in os.scandir(inputScenesDir) if os.path.isdir(subdir.path)]:
        print('creating video for ', scene.name)
        outPath = os.path.join(videoOutDir, scene.name + '.mp4')
        createVideoFromFrames(outPath, scene.path)


def createVideoFromFrames(outPath: str, framesDirectory: str):
    framePaths = [framePath.path for framePath in os.scandir(framesDirectory) if framePath.path.endswith('.jpg')]

    height, width = 2408, 1770
    writer = cv2.VideoWriter(outPath, 0x21, 18, (height, width), True)
    writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 1)

    framePaths.sort()
    for framePath in framePaths:
        im = cv2.imread(framePath)
        writer.write(im)
