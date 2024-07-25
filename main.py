import cv2
from ultralytics import YOLO
yolo = YOLO('yolov8s.pt')
videoCap = cv2.VideoCapture(0)

#Getting the colors to differentiate items
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
    (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

while True:
    ret, frame = videoCap.read()
    if not ret:
        continue
    results = yolo.track(frame, stream=True)

    for result in results:
        classes_names = result.names
        # iterate over each box
        for box in result.boxes:
            # confidence over 40%
            if box.conf[0] > 0.4:
                # steps to draw the rectangle, from finding the box, converting to int, setting the class name, coloring, then drawing the rectangle with the name and confidence level.
                [x1, y1, x2, y2] = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = int(box.cls[0])
                class_name = classes_names[cls]
                colour = getColours(cls)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)  
    # show the image
    cv2.imshow('frame', frame)
    # break the loop with q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#end program
videoCap.release()
cv2.destroyAllWindows()
