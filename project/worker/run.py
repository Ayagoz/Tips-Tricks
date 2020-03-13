import cv2

cv2.setNumThreads(0)
import sys
import logging
from logging.handlers import RotatingFileHandler

from worker.state import State
from worker.video_reader import VideoReader
# from worker.ocr_stream import OcrStream
from worker.visualize_stream import VisualizeStream


# from worker.config import CONFIG
# TODO: INITIALIZE YOUR args parser and remove sys.argv[] calls


def setup_logging(path, level='INFO'):
    handlers = [logging.StreamHandler()]
    file_path = path
    if file_path:
        file_handler = RotatingFileHandler(filename=file_path,
                                           maxBytes=10 * 10 * 1024 * 1024,
                                           backupCount=5)
        handlers.append(file_handler)
    logging.basicConfig(
        format='[{asctime}][{levelname}] - {name}: {message}',
        style='{',
        level=logging.getLevelName(level),
        handlers=handlers,
    )


class CNDProject:
    def __init__(self, name, video_path, save_path, fps=60, frame_size=(800, 1600), coord=(500, 500)):
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.state = State()
        self.video_reader = VideoReader("VideoReader", video_path)
        # self.ocr_stream = OcrStream(self.state, self.video_reader)

        self.visualize_stream = VisualizeStream("VisualizeStream", self.video_reader,
                                                self.state, save_path, fps, frame_size, coord)
        self.logger.info("Start Project")

    def start(self):
        self.logger.info("Start project act start")
        try:
            self.video_reader.start()
            # self.ocr_stream.start()
            self.visualize_stream.start()
            self.state.exit_event.wait()
        except Exception as e:
            self.logger.exception(e)
        finally:
            self.stop()

    def stop(self):
        self.logger.info("Stop Project")

        self.video_reader.stop()
        # self.ocr_stream.stop()
        self.visualize_stream.stop()


if __name__ == '__main__':
    setup_logging(sys.argv[1], sys.argv[2])
    logger = logging.getLogger(__name__)
    project = None
    try:
        project = CNDProject("CNDProject", sys.argv[3], sys.argv[4])
        project.start()
    except Exception as e:
        logger.exception(e)
    finally:
        if project is not None:
            project.stop()
