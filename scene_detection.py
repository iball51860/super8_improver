import os
import cv2
import numpy as np
import pandas as pd


class SceneDetector:
    baseDirectory: str
    scanInputDirectories: list
    baseOutputDirectory: str
    sceneDetectionOutputDirectory: str
    currentScanOutDirectory: str
    _currentSceneDirectory: str
    _currentScanInputDirectory: str
    _currentFramePath: str

    infoDicts: list = []
    _sceneDict: dict

    _duplicatesPaths: list = []
    _erroneousPaths: list = []

    _scanNo: int = -1
    _sceneNo: int = -1
    _scanFrameNo: int = -1
    _sceneFrameCounter: int = -1
    _accumulatedDuplicates: int = 0
    _accumulatedErroneous: int = 0
    _inputStartFrame: int = 0
    _diffs: list = []
    _lastDiffs: list = []

    def __init__(self, projectDirectory):
        self.baseDirectory = projectDirectory
        self.scanInputDirectories = [subdir.path for subdir in os.scandir(self.baseDirectory) if subdir.is_dir()]
        self.baseOutputDirectory = os.path.join(self.baseDirectory, 'output')
        self.sceneDetectionOutputDirectory = os.path.join(self.baseOutputDirectory, 'scanned_scenes')

    def detect_scenes(self):
        self.printStartInformation()

        os.makedirs(self.baseOutputDirectory)

        for scan_index in range(0, len(self.scanInputDirectories)):
            self._scanNo = scan_index
            self._currentScanInputDirectory = self.scanInputDirectories[scan_index]
            self.currentScanOutDirectory = os.path.join(self.sceneDetectionOutputDirectory, f'scan_{scan_index}')
            print(f'Started scene detection for scan {self._currentScanInputDirectory}')

            frames = self.getFramesForCurrentScan()

            if not len(frames) == 0:
                self._currentFramePath = frames[0]
                self.newScene(np.nan)
                lastFrame = np.zeros(cv2.imread(frames[0]).shape)
            for frame_index in range(0, len(frames)):
                self._scanFrameNo = frame_index
                self._sceneFrameCounter += 1
                self._currentFramePath = frames[self._scanFrameNo]
                frame = cv2.imread(self._currentFramePath)

                diff, frameDup, frameOK = self.analyzeFrame(frame, lastFrame, frames)

                if frameOK:
                    self._diffs.append(diff)
                    self._lastDiffs.append(diff)
                    if len(self._lastDiffs) > 36:
                        self._lastDiffs = self._lastDiffs[1:]
                    lastFrame = frame
                    append = ''
                elif frameDup:
                    append = '_duplicate'
                else:
                    append = '_erroneous'

                frameFileName = f'frame_{str(self._sceneFrameCounter).zfill(5)}{append}.jpg'
                frameOutPath = os.path.join(self._currentSceneDirectory, frameFileName)
                cv2.imwrite(frameOutPath, frame)
                if frameDup:
                    self._duplicatesPaths.append(frameOutPath)
                elif not frameOK:
                    self._erroneousPaths.append(frameOutPath)

        print(f'duplicates:', '\n'.join(self._duplicatesPaths))
        print(f'erroneous:', '\n'.join(self._erroneousPaths))
        self.writeInfoDictsToCSV()

    def printStartInformation(self):
        print('Starting scene detection')
        print(f'Project root is {self.baseDirectory}')
        print(f'Found {len(self.scanInputDirectories)} scans:')
        print('\n'.join(self.scanInputDirectories))
        print(f'writing output to {self.baseOutputDirectory}')

    def getFramesForCurrentScan(self):
        frames = []
        for dirpath, dirnames, filenames in os.walk(self._currentScanInputDirectory):
            frames += [os.path.join(dirpath, filename)
                       for filename in [f for f in filenames if f.endswith(".jpg")]]
        frames.sort()
        print(f'Found and sorted {len(frames)} frames.')
        return frames

    def analyzeFrame(self, frame, lastFrame, frames):
        diff = np.mean(np.sqrt((frame - lastFrame) ** 2))
        frameOK = True
        frameDup = False
        if diff == 0:
            print(f'duplicate frame {self._currentFramePath}')
            frameOK = False
            frameDup = True
        elif len(self._lastDiffs) > 5:
            diffThreshold = 8.0
            significantDiff = diff > diffThreshold
            nextFrameAvailable = self._scanFrameNo + 1 < len(frames)

            if significantDiff and nextFrameAvailable:
                nextFrame = cv2.imread(frames[self._scanFrameNo + 1])
                diffNext = np.mean(np.sqrt((nextFrame - lastFrame) ** 2))

                if diffNext > diffThreshold:
                    self.newScene(diff)
                else:
                    print(f'something seems wrong with frame {self._currentFramePath}.')
                    frameOK = False
        return diff, frameDup, frameOK

    def newScene(self, diff,):
        if self._sceneNo >= 0:
            newDuplicates = len(self._duplicatesPaths) - self._accumulatedDuplicates
            newErroneous = len(self._erroneousPaths) - self._accumulatedErroneous
            oldSceneDict = {**self._sceneDict,
                            'length': self._sceneFrameCounter,
                            'duplicates': newDuplicates,
                            'erroneous': newErroneous,
                            'end_frame': self._scanFrameNo - 1}
            self.infoDicts.append(oldSceneDict)
        self._sceneNo += 1
        self._sceneFrameCounter = 0
        self._accumulatedDuplicates = len(self._duplicatesPaths)
        self._accumulatedErroneous = len(self._erroneousPaths)
        self._currentSceneDirectory = os.path.join(self.currentScanOutDirectory, f'scene_{str(self._sceneNo).zfill(3)}')
        os.makedirs(self._currentSceneDirectory)
        self._sceneDict = {
            'scan': self._scanNo,
            'scene': self._sceneNo,
            'start_frame': self._scanFrameNo,
            'rmse': diff,
            'rolling_mean_rmse': np.mean(self._lastDiffs),
            'folder': self._currentSceneDirectory
        }
        print(f'new scene at frame {self._currentFramePath}. '
              f'Diff: {diff}, '
              f'avg{len(self._lastDiffs)}: {np.mean(self._lastDiffs)}')
        self._lastDiffs = []

    def writeInfoDictsToCSV(self):
        (pd.DataFrame(self.infoDicts)
            .set_index(['scene', 'scan'])
            .sort_index()
            .to_csv(
            os.path.join(self.baseOutputDirectory, 'info.csv')))
