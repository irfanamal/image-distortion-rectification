import cv2
from correction import BBox
from transform import ImageRectification

input_image_file = ''
input_bbox_file = ''
output_image_file = ''
output_bbox_file = ''
k = 0.1

image = cv2.imread(input_image_file)
boxes = []
with open(input_bbox_file, 'r') as f:
    lines = f.readlines()
for line in lines:
    boxes.append(BBox(*[float(x) for x in line.split(',')]))

rect = ImageRectification()
image_rec = rect.rectify(image, k)
cv2.imwrite(output_image_file, image_rec)

with open(output_bbox_file, 'a+') as f:
    for bbox in boxes:
        x, y, w, h = bbox.convert_pixel_2_yolo(bbox.correct(k, image_rec.shape[1], image_rec.shape[0]), image_rec.shape[1], image_rec.shape[0])
        f.write(f'{x}, {y}, {w}, {h}\n')
