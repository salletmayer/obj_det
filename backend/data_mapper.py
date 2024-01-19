import json
import torch
import numpy as np

def __swap__(n1, n2):
    help = n1
    n1 = n2
    n2 = help
    return n1, n2

def __restructure_bbox__(bbox):
    [x1, x2, y1, y2] = bbox

    if x1 > x2:
        x1, x2 = __swap__(x1, x2)
    if y1 > y2:
        y1, y2 = __swap__(y1, y2)

    if y1 == y2:
        y2 += 0.01
    if x1 == x2:
        x2 += 0.01

    return [x1, y1, x2, y2]

def __load_from_disk__(abs_path: str):
    with open(abs_path, 'r') as file:
        data = json.load(file)

    return data

def __combine_file_label__(data):
    result = []

    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        bbox = annotation["bbox"]
        bbox = __restructure_bbox__(bbox)

        category_id = annotation["category_id"]
        
        image_filename = next((image["file_name"] for image in data["images"] if image["id"] == image_id), None)
        
        image_found = False
        for stored_image in result:
            if stored_image['filename'] == image_filename:
                stored_image['labels'].append({
                    "category_id": category_id,
                    "bbox": bbox
                })

                image_found = True

        if image_filename and not image_found:
            result.append({
                "filename": image_filename,
                "labels": [{
                    "category_id": category_id,
                    "bbox": bbox
                }]
            })

    return result

def __map_single__(labels_one_image):
    boxes = []
    labels = []
    for annotation in labels_one_image['labels']:
        labels.append(annotation['category_id'])
        boxes.append(annotation['bbox'])

    return labels, boxes

def __map_to_set__(labels_all_images):
    targets = []
    image_paths = []
    for res in labels_all_images:
        d = {}

        tmp_labels, tmp_boxes = __map_single__(res)
        d['boxes'] = torch.from_numpy(np.array(tmp_boxes))

        d['labels'] = torch.from_numpy(np.array(tmp_labels))

        targets.append(d)
        image_paths.append(res['filename'])
    
    return image_paths, targets

def get_set(abs_path):
    data = __load_from_disk__(abs_path)
    combined = __combine_file_label__(data)
    image_paths, targets = __map_to_set__(combined)

    return image_paths, targets

if __name__ == '__main__':
    import os

    abspath = os.path.abspath("../data/labels.json")
    images, targets = get_set(abspath)

    print(targets[0]['boxes'].shape)