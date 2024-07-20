import logging
from ultralytics import YOLO

logger = logging.getLogger('yolo_benchmark')


def main(parser, main_parser):
    parser.add_argument(
        'source',
        type=str,
        help='Specifies the data source for inference. Can be an image path, video file, directory, URL, or device ID for live feeds. Supports a wide range of formats and sources, enabling flexible application across different types of input.'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Sets the minimum confidence threshold for detections. Objects detected with confidence below this threshold will be disregarded. Adjusting this value can help reduce false positives.'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.7,
        help='Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS). Lower values result in fewer detections by eliminating overlapping boxes, useful for reducing duplicates.'
    )
    parser.add_argument(
        '--imgsz',
        type=lambda x: tuple(map(int, x.split(','))) if ',' in x else int(x),
        default=640,
        help='Defines the image size for inference. Can be a single integer 640 for square resizing or a (height, width) tuple. Proper sizing can improve detection accuracy and processing speed.'
    )
    parser.add_argument(
        '--half',
        action='store_true',
        default=False,
        help='Enables half-precision (FP16) inference, which can speed up model inference on supported GPUs with minimal impact on accuracy.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Specifies the device for inference (e.g., cpu, cuda:0 or 0). Allows users to select between CPU, a specific GPU, or other compute devices for model execution.'
    )
    parser.add_argument(
        '--max_det',
        type=int,
        default=300,
        help='Maximum number of detections allowed per image. Limits the total number of objects the model can detect in a single inference, preventing excessive outputs in dense scenes.'
    )
    parser.add_argument(
        '--vid_stride',
        type=int,
        default=1,
        help='Frame stride for video inputs. Allows skipping frames in videos to speed up processing at the cost of temporal resolution. A value of 1 processes every frame, higher values skip frames.'
    )
    parser.add_argument(
        '--stream_buffer',
        action='store_true',
        default=False,
        help='Determines if all frames should be buffered when processing video streams (True), or if the model should return the most recent frame (False). Useful for real-time applications.'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        default=False,
        help='Activates visualization of model features during inference, providing insights into what the model is "seeing". Useful for debugging and model interpretation.'
    )
    parser.add_argument(
        '--augment',
        action='store_true',
        default=False,
        help='Enables test-time augmentation (TTA) for predictions, potentially improving detection robustness at the cost of inference speed.'
    )
    parser.add_argument(
        '--agnostic_nms',
        action='store_true',
        default=False,
        help='Enables class-agnostic Non-Maximum Suppression (NMS), which merges overlapping boxes of different classes. Useful in multi-class detection scenarios where class overlap is common.'
    )
    parser.add_argument(
        '--classes',
        type=lambda s: [int(item) for item in s.split(',')],
        default=None,
        help='Filters predictions to a set of class IDs. Only detections belonging to the specified classes will be returned. Useful for focusing on relevant objects in multi-class detection tasks.'
    )
    parser.add_argument(
        '--retina_masks',
        action='store_true',
        default=False,
        help='Uses high-resolution segmentation masks if available in the model. This can enhance mask quality for segmentation tasks, providing finer detail.'
    )
    parser.add_argument(
        '--embed',
        type=lambda s: [int(item) for item in s.split(',')],
        default=None,
        help='Specifies the layers from which to extract feature vectors or embeddings. Useful for downstream tasks like clustering or similarity search.'
    )

    main_args = main_parser.parse_args()
    args, _ = parser.parse_known_args()

    model = YOLO(
        model=main_args.model,
        verbose=main_args.verbosity > 0,
    )

    kwargs = {
        k: getattr(main_args, k)
        for k in vars(args)
        if hasattr(main_args, k)
    }
    logger.info("Predict conf: %s", kwargs)
    result = model.predict(**kwargs)[0]

    print(f"model : {main_args.model}")
    for key, value in kwargs.items():
        print(f"{key} : {value}")
    for key, value in result.speed.items():
        print(f"{key} : {value}")

    return result
