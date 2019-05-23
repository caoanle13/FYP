import cv2
import numpy as np

def random_colors(N):
    """ Generate random colors """
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range (N)]
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """ Apply mask to image """
    for n,c in enumerate(color):
        image[:,:,n] = np.where(
            mask==1,
            image[:,:,n]* (1 - alpha) + alpha * c,
            image[:,:,n]
        )
    return image

def display_instances(image, boxes, masks, ids, names, scores):

    n_instances = boxes.shape[0]

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    colors = random_colors(n_instances)
    height, width = image.shape[:2]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        mask = masks[:,:,i]
        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1,y1), (x2,y2), color, 1)
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 1
        )
    
    return image