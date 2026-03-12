import argparse
import cv2
from model import Model, CATEGORY_META


def argument_parser():
    parser = argparse.ArgumentParser(description='Safety Monitor — single image test')
    parser.add_argument('--image-path', type=str, default='./data/7.jpg',
                        help='path to image')
    return parser.parse_args()


if __name__ == '__main__':
    args = argument_parser()
    model = Model()

    image = cv2.imread(args.image_path)
    rgb   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = model.predict(image=rgb)

    print(f"\n── Safety Monitor Result ─────────────────")
    print(f"  Label      : {result['label']}")
    print(f"  Category   : {result['category']}")
    print(f"  Confidence : {result['confidence']:.3f}")
    print(f"  Alert      : {'YES — ' + CATEGORY_META[result['category']]['alert'] if result['is_alert'] else 'No'}")
    print(f"  Headcount  : {result['headcount']} people")
    if result['crowd_alert']:
        print(f"  ⚠ CROWD ALERT: {result['headcount']} people detected")
    print(f"──────────────────────────────────────────\n")

    # Draw results on image
    meta  = CATEGORY_META.get(result['category'], CATEGORY_META['normal'])
    color = meta['color']

    # Draw person bounding boxes
    for (x1, y1, x2, y2) in result['person_boxes']:
        cv2.rectangle(image, (x1, y1), (x2, y2), (80, 220, 80), 2)

    # Draw status banner
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], 55), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.58, image, 0.42, 0, image)

    text = f"{meta['alert']}  ({result['confidence']:.0%})  | People: {result['headcount']}"
    cv2.putText(image, text, (10, 38), cv2.FONT_HERSHEY_DUPLEX, 0.85, color, 2, cv2.LINE_AA)

    if result['is_alert']:
        h, w = image.shape[:2]
        cv2.rectangle(image, (2, 2), (w - 3, h - 3), color, 5)

    cv2.imshow(meta['alert'], image)
    cv2.waitKey(0)