import cv2
import os


def createVideos(inputScenesDir: str):
    outDirectory = os.path.split(inputScenesDir)[0]
    videoOutDir = os.path.join(outDirectory, 'video')
    os.makedirs(videoOutDir)
    scenePaths = [subdir for subdir in os.scandir(inputScenesDir) if os.path.isdir(subdir.path)]
    scenePaths.sort(key=lambda dirE: dirE.name)
    for scene in scenePaths:
        print('creating video for ', scene.name)
        outPath = os.path.join(videoOutDir, scene.name + '.mp4')
        createVideoFromFrames(outPath, scene.path)


def createVideoFromFrames(outPath: str, framesDirectory: str):
    framePaths = [framePath.path for framePath in os.scandir(framesDirectory) if framePath.path.endswith('.jpg')]
    framePaths.sort()
    
    height, width, channels = cv2.imread(framePaths[0]).shape
    
    writer = cv2.VideoWriter(outPath, 0x21, 18, (width, height), True)
    writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 1)

    for framePath in framePaths:
        im = cv2.imread(framePath)
        writer.write(im)
    writer.release()
