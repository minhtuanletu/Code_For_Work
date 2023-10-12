from ultralytics import YOLO
import cv2
import pandas as pd
import torch
def intersection_over_union(boxes_preds, boxes_labels):
    box1_x1 = boxes_preds[0]
    box1_y1 = boxes_preds[1]
    box1_x2 = boxes_preds[2]
    box1_y2 = boxes_preds[3]
    box2_x1 = boxes_labels[0]
    box2_y1 = boxes_labels[1]
    box2_x2 = boxes_labels[2]
    box2_y2 = boxes_labels[3]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def nms(bboxes, iou_threshold, conf_threshold):
    # box format: <xmin> <ymin> <xmax> <ymax> <conf> <class_id>
    assert type(bboxes) == list
    bboxes = [box for box in bboxes if box[4] > conf_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [box for box in bboxes if box[-1] != chosen_box[-1] or
                  intersection_over_union(torch.tensor(chosen_box[:4]), torch.tensor(box[:4])) < iou_threshold]
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def draw_bounding_box(img, boxes):
    for box in boxes:
        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[2])
        ymax = int(box[3])
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)
        # img = cv2.putText(img, f"{float(box[-2]):.2f}", (xmax, ymax), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv2.imwrite("res_before_NMS.jpg", img)
    cv2.imwrite("res_after_NMS.jpg", img)

def convert_bounding_box(result):
    boxes = result[0].boxes
    xyxy = pd.DataFrame(boxes.xyxy.cpu().numpy(), columns=['xmin', 'ymin', 'xmax', 'ymax'])
    conf = pd.DataFrame(boxes.conf.cpu().numpy(), columns=['confidence'])
    cls = pd.DataFrame(boxes.cls.cpu().numpy(), columns=['class'])
    result = pd.concat([xyxy, conf,cls], axis=1)
    return result.values.tolist()

def output(model,image):
    img = cv2.imread(image)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = model.predict(img_rgb, save=False, imgsz=1344, conf=0.1, verbose=False)
    boxes = convert_bounding_box(result)
    boxes_after_nms = nms(boxes, iou_threshold=0.3, conf_threshold=0.1)
    # boxes = list(map(lambda x: [int(y) for y in x], boxes))
    # for i in boxes:
    #     print(f"Value: {i[5]}, coordinates: {i[0:4]}")
    # draw_bounding_box(img_rgb, boxes)
    draw_bounding_box(img_rgb, boxes_after_nms)

model=YOLO('tuan_v2.pt')
output(model,'imagetoexcel_binary.jpg')