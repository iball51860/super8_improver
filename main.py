from create_video import createVideos
from image_align import alignAndStack
from scene_detection import SceneDetector
import argparse
import os


parser = argparse.ArgumentParser(description='Putting together reflecta super8'
                                             'scans professionally.')
parser.add_argument('command',
                    nargs=1,
                    choices=['split', 'stack', 'video'],
                    help='Step in processing pipeline to execute.')
parser.add_argument('project_path', type=str, nargs=1, help='Project path (in Docker container!).')

arguments = parser.parse_args()


def main(args):
    projectDir = args.project_path[0]
    if args.command[0] == 'split':
        sceneDetector = SceneDetector(projectDir)
        sceneDetector.detect_scenes()
    elif args.command[0] == 'stack':
        alignAndStack(projectDir)
    elif args.command[0] == 'video':
        createVideos(os.path.join(projectDir, 'output/average'))


main(arguments)
