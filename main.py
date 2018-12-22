from scene_detection import SceneDetector


def main():
    sceneDetector = SceneDetector('/Volumes/Samsung_T5/Super-8/suedfrankreich')
    sceneDetector.detect_scenes()


main()
