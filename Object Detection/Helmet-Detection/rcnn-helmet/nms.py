import numpy as np

def nms(bounding_boxes, confidence_score, labels, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [],[],[]


    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # labels
    labels = np.array(labels)

    
    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    picked_labels = []

    unique_labels = np.unique(labels)

    mask = [labels == i for i in unique_labels]

    iterator = 0

    for m in mask:
      
      sub_boxes = boxes[m]
      sub_score = score[m]

     

      # coordinates of bounding boxes
      start_x = sub_boxes[:, 0]
      start_y = sub_boxes[:, 1]
      end_x = sub_boxes[:, 2]
      end_y = sub_boxes[:, 3]

      


      # Compute areas of bounding boxes
      areas = (end_x - start_x + 1) * (end_y - start_y + 1)
    

      # Sort by confidence score of bounding boxes
      order = np.argsort(sub_score)
      

      # Iterate bounding boxes
      while order.size > 0:
          # The index of largest confidence score
          index = order[-1]

        

          # Pick the bounding box with largest confidence score
          picked_boxes.append(sub_boxes[index])
          picked_score.append(sub_score[index])
          picked_labels.append(unique_labels[iterator])

          

          # Compute ordinates of intersection-over-union(IOU)
          #x1 returns max of n-1 x1 as it brodcast find max one by one
          x1 = np.maximum(start_x[index], start_x[order[:-1]])
          x2 = np.minimum(end_x[index], end_x[order[:-1]])
          y1 = np.maximum(start_y[index], start_y[order[:-1]])
          y2 = np.minimum(end_y[index], end_y[order[:-1]])

          # Compute areas of intersection-over-union
          w = np.maximum(0.0, x2 - x1 + 1)
          h = np.maximum(0.0, y2 - y1 + 1)
          intersection = w * h

          

          # Compute the ratio between intersection and union
          ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

          left = np.where(ratio < threshold)

          # continue with those who are not so close to main frame.
          order = order[left]
      
      # color loop
      iterator+=1

    return picked_boxes, picked_score, picked_labels