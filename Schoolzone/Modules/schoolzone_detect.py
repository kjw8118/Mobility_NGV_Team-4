"""
camParas : [alpha, beta, gamma, dist, f, H, W]
ipmParas : [ipmMat,]
bgMask : [white_mask, black_mask]
CalibParas : [CamMat, DistMat]
"""

# import modules
import cv2
import numpy as np
import math

##################################################################################################################
###################################### define functions #################################
#################                                                                                #################
###                                                                                                            ###

##################################################################################################################
#####################################  Setup function  ##################################
##################################################################################################################
def cal_trans_mat(camParas):
    alpha = camParas[0]
    beta = camParas[1]
    gamma = camParas[2]
    dist = camParas[3]
    f = camParas[4]
    H = camParas[5]
    W = camParas[6]
    
    A1 = np.array([[1,0,-W/2],[0,1,-H/2],[0,0,0],[0,0,1]],dtype='f')
    RX = np.array([[1,0,0,0],[0,math.cos(alpha), -math.sin(alpha),0],[0,math.sin(alpha),math.cos(alpha),0],[0,0,0,1]],dtype='f')
    RY = np.array([[math.cos(beta),0,-math.sin(beta),0],[0,1,0,0],[math.sin(beta),0,math.cos(beta),0],[0,0,0,1]],dtype='f')
    RZ = np.array([[math.cos(gamma),-math.sin(gamma),0,0],[math.sin(gamma),math.cos(gamma),0,0],[0,0,1,0],[0,0,0,1]],dtype='f')
    R = np.dot(RX,np.dot(RY,RZ))
    T = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,dist],[0,0,0,1]],dtype='f')
    K = np.array([[f,0,W/2,0],[0,f,H/2,0],[0,0,1,0]],dtype='f')
    ipmParas = np.dot(K,np.dot(T,np.dot(R,A1)))
    
    return ipmParas

def cal_ipm_bg(camParas,ipmParas):
    H, W = camParas[5:7]
    mask_none_w = np.full((H, W ,3),255,dtype='uint8')
    mask_ipm_w = cv2.warpPerspective(mask_none_w,ipmParas,(int(W),int(H)), flags=cv2.INTER_CUBIC|cv2.WARP_INVERSE_MAP)
    mask_ipm_k = cv2.bitwise_not(mask_ipm_w)
    
    bgMask = [mask_ipm_w, mask_ipm_k]
    
    return bgMask
    
def setup_ssd_edgetpu(args):
    # Get Args
    MODEL_NAME = args.modeldir
    GRAPH_NAME = args.graph
    LABELMAP_NAME = args.labels
    min_conf_threshold = float(args.threshold)
    resW, resH = args.resolution.split('x')
    imW, imH = int(resW), int(resH)
    use_TPU = args.edgetpu
    
    # Import TensorFlow libraries
    # If tensorflow is not installed, import interpreter from tflite_runtime, else import from regular tensorflow
    # If using Coral Edge TPU, import the load_delegate library
    pkg = importlib.util.find_spec('tensorflow')
    if pkg is None:
        from tflite_runtime.interpreter import Interpreter
        if use_TPU:
            from tflite_runtime.interpreter import load_delegate
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        if use_TPU:
            from tensorflow.lite.python.interpreter import load_delegate

    # If using Edge TPU, assign filename for Edge TPU model
    if use_TPU:
        # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
        if (GRAPH_NAME == 'detect.tflite'):
            GRAPH_NAME = 'edgetpu.tflite'       

    # Get path to current working directory
    CWD_PATH = os.getcwd()

    # Path to .tflite file, which contains the model that is used for object detection
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Have to do a weird fix for label map if using the COCO "starter model" from
    # https://www.tensorflow.org/lite/models/object_detection/overview
    # First label is '???', which has to be removed.
    if labels[0] == '???':
        del(labels[0])

    # Load the Tensorflow Lite model.
    # If using Edge TPU, use special load_delegate argument
    if use_TPU:
        interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                  experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        print(PATH_TO_CKPT)
    else:
        interpreter = Interpreter(model_path=PATH_TO_CKPT)

    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5
    
    tfParas[0] = height
    tfParas[1] = width
    tfParas[2] = floating_model
    tfParas[3] = input_data
    tfParas[4] = input_mean
    tfParas[5] = input_std
    tfParas[6] = inpput_details
    tfParas[7] = min_conf_threshold
    tfParas[8] = imH
    tfParas[9] = imW
    tfParas[10] = labels
    
    
    return tfParas
###                                                                                                            ###
##################################################################################################################


##################################################################################################################
#####################################  Unit Modules  ####################################
##################################################################################################################
"""
description - detect lane by embedded c coding
input
  image : bin image (mask_k)
  output
  lane :  position of lane (x=f(y))
          [lane_left, lane_right]
"""
def detect_lane(mask_k):
    pass
    
    
    
"""
description - detect schoolzone by calculating
input
  image : bin image (mask_r)
  output
  state_schoolzone : 0 - None | 1 - Detected
"""
def detect_schoolzone(mask_r):
    # Get Paras
    H, W = mask_r.shape[0:2]
    
    # ROI
    mask_roi = mask_r[int(H*1/5):H,int(W*3/8):int(W*5/8)]
    num_r = mask_roi.sum()/255
    num_roi = int(H*4/5*W*2/8)
    num_thr = int(num_roi/10)
    if(num_r > num_thr): return 1
    else: return 0

"""
description - detect stopline by ssd tflite
input
  image : bin image (mask_k)
  output
  state_stopline : 0 - None | 1 - Detected
"""
def detect_stopline(mask_k, tfParas):    # From EdjeElectronics
    # Get Paras
    height = tfParas[0]
    width = tfParas[1]
    floating_model = tfParas[2]
    input_data = tfParas[3]
    input_mean = tfParas[4]
    input_std = tfParas[5]
    inpput_details = tfParas[6]
    min_conf_threshold = tfParas[7]
    imH = tfParas[8]
    imW = tfParas[9]
    labels = tfParas[10]
    
    # Get image & preprocessing
    frame = cv2.cvtColor(mask_k, cv2.COLOR_GRAY2RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            print("Detected : ",labels[int(classes[i])])
            if(labels[int(classes[i])] == "stopline"): state_stopline = 1
            else: state_stopline = 0
            """

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            """
    return state_stopline
###                                                                                                            ###
##################################################################################################################
    
    
##################################################################################################################
##################################  Integrated Modules  #################################
##################################################################################################################
def pre_process(image, calibParas, ipmParas, bgMask):  # & loop
    # Get Parameters
    H, W = image.shape[0:2]
    ipmMat = ipmParas
    CamMat, DistMat = calibParas
    
    lower_k = np.array([0,0,0])
    upper_k = np.array([180,255,100])
    lower_r = np.array([-30, 50, 50])
    upper_r = np.array([30,255,255])

    
    # Calibrate Cam
    img_calib = cv2.undistort(image, CamMat, DistMat, None, CamMat)
    
    # Inverse Perspective Mapping
    img_ipm_org = cv2.warpPerspective(img_calib, ipmMat, (W,H), flags=cv2.INTER_CUBIC|cv2.WARP_INVERSE_MAP)
    cv2.waitKey(0)
    
    # Add White Background
    img_ipm = cv2.add(img_ipm_org, bgMask[1])
    
    # Filtering
    img_filtered = cv2.bilateralFilter(img_ipm, 9, 50, 50)
    
    # Color Conversion
    img_hsv = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2HSV)
    
    # Masking
    mask_k = cv2.inRange(img_hsv, lower_k, upper_k)
    mask_r = cv2.inRange(img_hsv, lower_r, upper_r)
    
    mask = [mask_r, mask_k]
    
    # Detect Lane
    
    
    # Detect Schoolzone
    state_schoolzone = detect_schoolzone(mask[0])
    
    # Detect Stopline
    if(state_schoolzone == 1):
        print("schoolzone!")        
        #state_stopline = detect_stopline(mask[1])
        #if(state_stoplilne == 1):
            #print("Stopline!")
            
    
    
    pass
    
###                                                                                                            ###
##################################################################################################################


###                                                                                                            ###
#################                                                                                #################
##################################################################################################################

        
        
    
    
    
    
    
    
    
    
    
    
""" Old Modules

description - line detect & return line array
input
  image : bin image (gray scale, ipm)
  output
  line : left(0) & right(1) line array (np)

def line_detect(image):
    #img = cv2.threshold(cv2.bitwise_not(cv2.flip(image,0)),150,1,cv2.THRESH_BINARY)
    H, W = imgage.shape
    pos_M = int(W/2)
    line_L = [0 for i in range(H)]
    line_R = [0 for i in range(H)]
    band = 10
    
    for j in range(0,H):
        if(j==0):
            list_L = range(int(pos_M*3/4),int(pos_M))
            list_R = range(int(pos_M),int(pos_M*5/4))
        else:
            if(line_L[j-1]-band < 0 ): LL = 0
            else: LL = line_L[j-1]-band
            if(line_R[j-1]+band > W): RR = W
            else: RR = line_R[j-1]+band
            list_L = range(LL,line_L[j-1]+band)
            list_R = range(line_R[j-1]-band,RR)
        
        line_L[j] = int(sum(img_result[j,list_L[0]:list_L[-1]+1]*list_L)/sum(img_result[j,list_L[0]:list_L[-1]+1]))
        line_R[j] = int(sum(img_result[j,list_R[0]:list_R[-1]+1]*list_R)/sum(img_result[j,list_R[0]:list_R[-1]+1]))
        img_cp[j+130,line_L[j]+int(orgW/2)-80] = 255
        img_cp[j+130,line_R[j]+int(orgW/2)-80] = 255
        
        pos_M = int((line_L[j]+line_R[j])/2)
        
    line =np.array([line_L,line_R])
    
    return line
    
    




description - crop image bw lines
input
  image : original image (gray scale, ipm)
  line_left : left line x coord array
  line_right : right line x coord array
output
  image_roi : cropped image inter left & right line

def roi(image, line_left, line_right):
    x1 = max(line_left)
    x2 = max(line_right)
    image_roi = image[:,x1:x2]
    
    return image_roi
"""

if __name__ == '__main__':
    image = cv2.imread('./images/img_200129_022513_27.jpg',cv2.IMREAD_COLOR)
    CamMat = np.array([[314.484, 0, 321.999],[0, 315.110, 259.722],[ 0, 0, 1]],dtype='f')
    DistMat = np.array([ -0.332015,	0.108453,	0.001100,	0.002183],dtype='f')
    calibParas = [CamMat, DistMat]
    H, W = image.shape[0:2]
    camParas = [(8-90)*np.pi/180, (0)*np.pi/180, (0)*np.pi/180, 300, 500, H, W]
    ipmParas = cal_trans_mat(camParas)
    bgMask = cal_ipm_bg(camParas,ipmParas)
    pre_process(image, calibParas, ipmParas, bgMask)
    
    
