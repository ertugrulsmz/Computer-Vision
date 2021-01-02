import torchvision.transforms as transforms
import cv2
import numpy
import numpy as np
from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))
transform = transforms.Compose([
    transforms.ToTensor(),
])

def predict(image, model, device, detection_threshold):
    # transform the image to tensor
    image = transform(image).to(device)
    image = image.unsqueeze(0) # add a batch dimension
    outputs = model(image) # get the predictions on the image
    # print the results individually
    # print(f"BOXES: {outputs[0]['boxes']}")
    # print(f"LABELS: {outputs[0]['labels']}")
    # print(f"SCORES: {outputs[0]['scores']}")
    # get all the predicited class names
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    # get score for all the predicted objects
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # get all the predicted bounding boxes
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # get boxes above the threshold score
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    return boxes, pred_classes, outputs[0]['labels']
	

def draw_boxes(boxes, classes, labels, image):
    # read the image with OpenCV
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
                    lineType=cv2.LINE_AA)
    return image
	

import torchvision
import numpy
import torch
import argparse
import cv2
import detect_utils
from PIL import Image


if __name__ == "__main__":
	
	# construct the argument parser
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', help='path to input image/video')
	parser.add_argument('-m', '--min-size', dest='min_size', default=800, 
						help='minimum input size for the FasterRCNN network')
	args = vars(parser.parse_args())



	# download or load the model from disk
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, 
														min_size=args['min_size'])
														

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


	image = Image.open(args['input'])
	model.eval().to(device)

	boxes, classes, labels = predict(image, model, device, 0.8)
	image = draw_boxes(boxes, classes, labels, image)
	cv2.imshow('Image', image)


	import os
	name =  os.path.splitext(args["input"])[0]+'_detected.jpg'
	print("name : ",name)


	k = cv2.waitKey(0)


	if k == 27: #escape
	  cv2.destroyAllWindows()
	elif k == ord('s'):
	  cv2.imwrite(name, image)
	  cv2.destroyAllWindows()

	











