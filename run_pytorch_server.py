# encoding: utf-8
import os
import torch
from net import get_model
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
import numpy as np
import io
import json
import base64
import flask
#import torch
#import torch
#import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

#from torch import nn
#from torchvision import transforms as T
#from torchvision.models import resnet50
# anchor configuration
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5
transforms = T.Compose([
    T.Resize(size=(288, 144)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)
id2class = {0: 'Mask', 1: 'NoMask'}
id2chiclass = {0: '您戴了口罩', 1: '您没有戴口罩'}
colors = ((0, 255, 0), (255, 0 , 0))
# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)
# Initialize our Flask application and the PyTorch model.
app = flask.Flask(__name__)
model = None
use_gpu = False
import cv2
with open('imagenet_class.txt', 'r') as f:
    idx2label = eval(f.read())
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
def inference(net, image, conf_thresh=0.5, iou_thresh=0.4, target_shape=(160, 160), draw_result=True, chinese=False):
    height, width, _ = image.shape
    # print(len(image))
    # print(image,height, width)

    if height==0 or width==0 or image is None:
        print(image)
        return 1,[]
            

    
    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=target_shape)
    net.setInput(blob)
    y_bboxes_output, y_cls_output = net.forward(getOutputsNames(net))
    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes, bbox_max_scores, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
    # keep_idxs  = cv2.dnn.NMSBoxes(y_bboxes.tolist(), bbox_max_scores.tolist(), conf_thresh, iou_thresh)[:,0]
    tl = round(0.002 * (height + width) * 0.5) + 1  # line thickness
    class_id = 1
    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)
        if draw_result:
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors[class_id], thickness=tl)
            if chinese:
                image = puttext_chinese(image, id2chiclass[class_id], (xmin, ymin), colors[class_id])  ###puttext_chinese
            else:
                cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[class_id])
    return class_id,image
def load_model():
    """Load the pre-trained model, you can use your model just as easily.

    """
    global Net
    Net = cv2.dnn.readNet('models/face_mask_detection.caffemodel', 'models/face_mask_detection.prototxt')
    
    #model = resnet50(pretrained=True)
    #model.eval()



def prepare_image(image, target_size):
    """Do image preprocessing before prediction on any data.

    :param image:       original image
    :param target_size: target image size
    :return:
                        preprocessed image
    """

    if image.mode != 'RGB':
        image = image.convert("RGB")

    # Resize the input image nad preprocess it.
    image = T.Resize(target_size)(image)
    image = T.ToTensor()(image)

    # Convert to Torch.Tensor and normalize.
    image = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

    # Add batch_size axis.
    image = image[None]
    if use_gpu:
        image = image.cuda()
    return torch.autograd.Variable(image, volatile=True)
#---------------------------------
def load_network(network):
    save_path = os.path.join('./checkpoints', 'market', 'resnet50_nfc', 'net_last.pth')
    network.load_state_dict(torch.load(save_path))
    print('Resume model from {}'.format(save_path))
    return network

def load_image(src):
    #src = Image.open(path)
    src = transforms(Image.fromarray(src))
    src = src.unsqueeze(dim=0)
    return src

class predict_decoder(object):

    def __init__(self, dataset):
        with open('./doc/label.json', 'r') as f:
            self.label_list = json.load(f)[dataset]
        with open('./doc/attribute.json', 'r') as f:
            self.attribute_dict = json.load(f)[dataset]
        self.dataset = dataset
        self.num_label = len(self.label_list)

    def decode(self, pred,data):
        pred = pred.squeeze(dim=0)
        for idx in range(self.num_label):
            name, chooce = self.attribute_dict[self.label_list[idx]]
            if chooce[pred[idx]]:
                data[name] = chooce[pred[idx]]
                print('{}: {}'.format(name, chooce[pred[idx]]))
        return data

@app.route("/predict", methods=["POST"])
def predict():
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}
    #print(flask.request.form["image"],flask.request.get_data())
    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == 'POST':
        if flask.request.files.get("image"):
            #print('img')
            #img = base64.b64decode(str(flask.request.form['image']))
            #image_data = np.fromstring(img, np.uint8)
            #image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

            # Read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            #img = cv2.imread(args.img_path)
            image = np.uint8(image)
            #img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(image)
            results,_ = inference(Net, image, target_shape=(260, 260))
            # Preprocess the image and prepare it for classification.
            #image = prepare_image(image, target_size=(224, 224))

            # Classify the input image and then initialize the list of predictions to return to the client.
            #preds = F.softmax(model(image), dim=1)
            #results = torch.topk(preds.cpu().data, k=3, dim=1)

            #data['predictions'] = list()

            # Loop over the results and add them to the list of returned predictions
            #for prob, label in zip(results[0][0], results[1][0]):
            #    label_name = idx2label[label]
            #    r = {"label": label_name, "probability": float(prob)}
            #    data['predictions'].append(r)
            #data['img'] = base64.b64encode(results).decode()
            data['img'] = str(results)
            # Indicate that the request was a success.
            data["success"] = True
            print('true')
    #print(data)
    # Return the data dictionary as a JSON response.
    return flask.jsonify(data)

@app.route("/apr", methods=["POST"])
def apr():
    data = {"success": False}
    image = flask.request.files["image"].read()
    image = Image.open(io.BytesIO(image))
    #img = cv2.imread(args.img_path)
    image = np.uint8(image)
    model = get_model('resnet50_nfc', 30, False, 751)
    model = load_network(model)
    model.eval()
    #torch.from_numpy(image[np.newaxis,:,:,:])
    out = model.forward(load_image(image))
    pred = torch.gt(out, torch.ones_like(out)/2 )  
    dec = predict_decoder('market')
    return dec.decode(pred,data)
    #return 1

if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    load_model()
    app.run(port = 5000,host = '0.0.0.0')
