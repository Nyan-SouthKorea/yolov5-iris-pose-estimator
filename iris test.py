from cv2 import CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH
import torch
import cv2

def yolo_process(img):
    yolo_results = model(img)
    df = yolo_results.pandas().xyxy[0]
    obj_list = []
    for i in range(len(df)) :
        obj_confi = round(df['confidence'][i], 2)
        obj_name = df['name'][i]
        x_min = int(df['xmin'][i])
        y_min = int(df['ymin'][i])
        x_max = int(df['xmax'][i])
        y_max = int(df['ymax'][i])
        obj_dict = {
                    'class' : obj_name, 
                    'confidence' : obj_confi,
                    'xmin' : x_min,
                    'ymin' : y_min,
                    'xmax' : x_max, 
                    'ymax' : y_max
        }
        obj_list.append(obj_dict)
    return obj_list

model = torch.hub.load('ultralytics/yolov5', 'custom', path = './best.pt')
model.conf = 0.3
model.iou = 0
resize_rate = 1
iris_x_threshold, iris_y_threshold = 0.15, 0.26 # 눈동자가 중앙에서 얼마나 벗어나야 상태 바뀜으로 인정할 것인지
cap = cv2.VideoCapture(0)
# cap.set(CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(CAP_PROP_FRAME_HEIGHT, 1440)
iris_status = 'Center'
left_x_per = 'None'
while True:
    ret, img = cap.read()
    if not ret == True:
        break
    imgS = cv2.resize(img, (0, 0), None, resize_rate, resize_rate)
    results = yolo_process(imgS)

    eye_list = []
    iris_list = []
    
    # 화면에 bbox 그려줌
    for result in results:
        # print(result)
        xmin_resize = int(result['xmin'] / resize_rate)
        ymin_resize = int(result['ymin'] / resize_rate)
        xmax_resize = int(result['xmax'] / resize_rate)
        ymax_resize = int(result['ymax'] / resize_rate)
        if result['class'] == 'eye':
            # cv2.rectangle(img,(xmin_resize, ymin_resize), (xmax_resize, ymax_resize), (255, 255, 255), 1) # 얼굴 b box친다.
            pass
        if result['class'] == 'iris':
            x_length = xmax_resize - xmin_resize
            y_length = ymax_resize - ymin_resize
            circle_r = int((x_length + y_length) / 4)
            x_center = int((xmin_resize + xmax_resize) / 2)
            y_center = int((ymin_resize + ymax_resize) / 2)
            cv2.circle(img, (x_center, y_center), circle_r, (255, 255, 255), 1)
        if result['class'] == 'eye':
            eye_list.append(result)
        elif result['class'] == 'iris':
            iris_list.append(result)
    print('')
    
    # 왼쪽 파트와 오른쪽 파트를 나눔
    if len(eye_list) == 2 and len(iris_list) == 2:
        left_part = []
        right_part = []
        if eye_list[0]['xmin'] > eye_list[1]['xmin']:
            right_part.append(eye_list[0])
            left_part.append(eye_list[1])
        else:
            right_part.append(eye_list[1])
            left_part.append(eye_list[0])
        if iris_list[0]['xmin'] > iris_list[1]['xmin']:
            right_part.append(iris_list[0])
            left_part.append(iris_list[1])
        else:
            right_part.append(iris_list[1])
            left_part.append(iris_list[0])
        print('left list: \n{}'.format(left_part))
        print('right list: \n{}'.format(right_part))

        # 왼쪽 눈동자의 위치 비율
        left_x_iris_center = (left_part[1]['xmin'] + left_part[1]['xmax']) / 2
        left_x_per = (left_x_iris_center - left_part[0]['xmin']) / (left_part[0]['xmax'] - left_part[0]['xmin'])
        left_y_iris_center = (left_part[1]['ymin'] + left_part[1]['ymax']) / 2
        left_y_per = (left_y_iris_center - left_part[0]['ymin']) / (left_part[0]['ymax'] - left_part[0]['ymin'])

        # 오른쪽 눈동자의 위치 비율
        right_x_iris_center = (right_part[1]['xmin'] + right_part[1]['xmax']) / 2
        right_x_per = (right_x_iris_center - right_part[0]['xmin']) / (right_part[0]['xmax'] - right_part[0]['xmin'])
        right_y_iris_center = (right_part[1]['ymin'] + right_part[1]['ymax']) / 2
        right_y_per = (right_y_iris_center - right_part[0]['ymin']) / (right_part[0]['ymax'] - right_part[0]['ymin'])

        # 왼쪽 눈동자와 오른쪽 눈동자 비율의 평균
        avr_x_iris_per = (left_x_per + right_x_per) / 2
        avr_y_iris_per = (left_y_per + right_y_per) / 2

        # Threshold 기준으로 눈동자의 위치를 계산
        if avr_x_iris_per < (0.5 - iris_x_threshold):
            iris_status = 'Left'
        elif avr_x_iris_per > (0.5 + iris_x_threshold):
            iris_status = 'Right'
        elif avr_y_iris_per < (0.5 - iris_y_threshold):
            iris_status = 'Up'
        else:
            iris_status = 'Center'
    elif len(eye_list) == 2 and len(iris_list) == 0:
        iris_status = 'Down'

    cv2.putText(img, 'Iris Direction: {}'.format(iris_status),(10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (30, 30, 30), 2)
    cv2.imshow('img', img)
    cv2.waitKey(1)
cv2.destroyAllWindows()