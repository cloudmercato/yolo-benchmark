#!/usr/bin/env python
import os
import argparse
from yolo_benchmark.loggers import logger
from ultralytics.utils import LOGGER as ultralytics_logger
from yolo_benchmark.predict import main as predict

DEFAULT_MODEL = "yolov8n.pt"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', default=DEFAULT_MODEL)
    parser.add_argument('-v', '--verbosity', type=int, default=0, choices=(0, 1, 2))

    subparsers = parser.add_subparsers(dest="action", required=True)
    predict_parser = subparsers.add_parser("predict", help="")

    args, _ = parser.parse_known_args()

    logger.setLevel(40-(10+args.verbosity*10))
    ultralytics_logger.setLevel(60-(10+args.verbosity*10))
    os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
    logger.info('Log level: %s', logger.level)

    if args.action == 'predict':
        predict(predict_parser, parser)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
