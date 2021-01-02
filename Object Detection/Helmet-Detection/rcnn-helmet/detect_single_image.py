import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from PIL import Image
from torchvision import datasets, transforms
from nms import nms


prediction_threshold = 0.8
image_size = 415

image_labels = ["head","person","helmet"]

COLORS = np.random.uniform(0, 255, size=(len(image_labels)+1, 3))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = torch.load( "./rcnn_model" )
model.eval()


def predict(model,image,threshold):
  model.eval()
  transform = transforms.Compose([
    transforms.Resize((image_size,image_size)),
    transforms.ToTensor()
    
  ])

  image = transform(image).to(device)
  image = image.unsqueeze(0) # add a batch dimension

  outputs = model(image)
  
  


  # 0 represent nothing, so index starts at 0
  # returns class name and class index as array of tuples
  predict_classes = outputs[0]['labels'].cpu().numpy()

  # get score for all the predicted objects
  #scores are sorted by order, so first n boxes are important for us
  predict_scores = outputs[0]['scores'].detach().cpu().numpy()
    # get all the predicted bounding boxes
  boxes = outputs[0]['boxes'].detach().cpu().numpy()
  
  scores_bigger_than_threshold = predict_scores >= threshold
  predict_boxes = boxes[scores_bigger_than_threshold].astype(np.int32)
  

  return image[0], predict_boxes, predict_classes[:len(predict_boxes)], predict_scores[:len(predict_boxes)]





def draw_bounding_box(image,bounding_boxes,confidence_score,label):


  open_cv_image = np.array(image) 
  # Convert RGB to BGR 
  image = open_cv_image[:, :, ::-1].copy() 

  print("image shape : ",image.shape)

  # Draw parameters
  font = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 0.3
  thickness = 1
  
  for index,((start_x, start_y, end_x, end_y), confidence) in enumerate(zip(bounding_boxes, confidence_score)):

      print("confidence : ",confidence," rectangle :",(start_x,start_y)," ",(end_x, end_y))
      (w, h), baseline = cv2.getTextSize(str(image_labels[label[index]-1]), font, font_scale, thickness)
      cv2.rectangle(image, (start_x, start_y - (2 * baseline + 5)), (start_x + w, start_y), (22,93,0), -1)
      
      cv2.rectangle(image, (start_x, start_y), (end_x, end_y), COLORS[label[index]], 2)
    
      cv2.putText(image, str(image_labels[label[index]-1]), (start_x, start_y), font, font_scale, (0, 0, 0), thickness)
  
  return image
  
  
  
if __name__ == "__main__":

	# construct the argument parser
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', help='path to input image')
	parser.add_argument('-m', '--min-size', dest='min_size', default=800, 
						help='minimum input size for the FasterRCNN network')
						
	args = vars(parser.parse_args())
	
		
	sample_path1 = args['input']

	img = Image.open(sample_path1).convert("RGB") 
	
	image, boxes, classes, scores = predict(model,img, prediction_threshold)
	
	picked_boxes, picked_score, picked_labels = nms(boxes,scores,classes,0.4)
		
	np_predicted_image = image.to("cpu").numpy().transpose((1,2,0))
	
	if(len(picked_boxes) != 0):
		final_cv2_image = draw_bounding_box(np_predicted_image,picked_boxes, picked_score,picked_labels)
	else:
		final_cv2_image = np_predicted_image
	
	cv2.imshow("img",final_cv2_image)
	
	k = cv2.waitKey(0)
	
	
	import os
	path_without_extension = os.path.splitext(sample_path1)[0]
	path_to_save = path_without_extension + "_detected.bmp"
	
	# imwrite expects 0-255 interval not 0-1
	final_cv2_image = final_cv2_image*255
	
	
	if k == 27: #escape
	  cv2.destroyAllWindows()
	elif k == ord('s'):
	  cv2.imwrite(path_to_save, final_cv2_image)
	  cv2.destroyAllWindows()

	#final_plt_image = cv2.cvtColor(final_cv2_image,cv2.COLOR_BGR2RGB)

	#from pylab import rcParams
	#rcParams['figure.figsize'] = 10, 20
	#plt.imshow(final_plt_image)

	
	
	
	
	