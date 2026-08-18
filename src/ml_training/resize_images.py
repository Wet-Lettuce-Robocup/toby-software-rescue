import time
from pathlib import Path
import cv2

dir = Path('raw_photos')

out_dir = dir / 'Anew'
out_dir.mkdir(parents=True, exist_ok=True)


def run():
    for image_path in dir.iterdir():
        if (
            not image_path.is_file()
            or image_path.parent != dir
            or image_path.suffix.lower()
            not in [
                '.jpg',
                '.jpeg',
                '.png',
            ]
        ):
            continue

        image = cv2.imread(image_path)

        if image is None:
            print(f'Error: image {image_path.stem} not loading')
            continue

        cropped_frame, top_left, bottom_right = crop_frame(image)
        # display_frame = image.copy()
        # cv2.rectangle(display_frame, top_left, bottom_right, (0, 0, 255), 2)
        # cv2.imshow('Frame', display_frame)
        # cv2.waitKey(1)

        # time.sleep(1)
        success = cv2.imwrite(
            out_dir / (image_path.stem + '_cropped.jpg'),
            cropped_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 100],
        )

        if not success:
            print(f'Image {image_path.stem}failed to save')
    time.sleep(0.01)


def crop_frame(frame):
    height, width = frame.shape[:2]

    bottom_margin = int(height * 0.02)
    start_y = int(height / 3)
    end_y = int(height - bottom_margin)

    side_margin = int(width * 0.05)
    start_x = side_margin
    end_x = width - side_margin

    return (frame[start_y:end_y, start_x:end_x]), (start_x, start_y), (end_x - 1, end_y - 1)


if __name__ == '__main__':
    try:
        run()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
