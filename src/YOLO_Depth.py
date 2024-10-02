import pyrealsense2 as rs
import numpy as np
import cv2
import rospy
from ultralytics import YOLO
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image
import tf2_ros
from cv_bridge import CvBridge

# ROS 노드 초기화
rospy.init_node('object_tf_publisher')

# TF 브로드캐스터 설정
tf_broadcaster = tf2_ros.TransformBroadcaster()

# 이미지 퍼블리셔 설정
image_pub = rospy.Publisher("/detection_image", Image, queue_size=10)

# CvBridge 객체 생성
bridge = CvBridge()

# Realsense 파이프라인 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# YOLO 모델 불러오기
model = YOLO('yolov8s.pt')
class_names = model.names  # 클래스 이름 리스트

# 타겟 클래스 설정 (예: 'bottle' 대신 원하는 클래스 이름으로 변경)
target_class_name = 'bottle'

# 정렬기(align) 객체 설정 (컬러 이미지와 깊이 이미지 정렬)
align_to = rs.stream.color
align = rs.align(align_to)

# 파이프라인 시작
pipeline.start(config)

try:
    while not rospy.is_shutdown():
        # 카메라로부터 프레임 받아오기
        frames = pipeline.wait_for_frames()
        
        # 프레임 정렬 (컬러와 깊이 프레임을 정렬)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # 프레임을 numpy 배열로 변환
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # YOLO 객체 감지 수행
        results = model(color_image)

        # 타겟 클래스 객체만 필터링
        filtered_boxes = [box for box in results[0].boxes if class_names[int(box.cls)] == target_class_name]

        # 타겟 클래스 객체만 시각화 및 깊이 정보 추출
        annotated_image = color_image.copy()
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics  # 깊이 이미지의 내재 파라미터

        for box in filtered_boxes:
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0])  # 좌표를 정수로 변환
            confidence = box.conf.item()  # 텐서를 파이썬 float로 변환
            label = f"{class_names[int(box.cls)]} {confidence:.2f}"  # 클래스 이름과 신뢰도 레이블

            # 박스의 중심 좌표 계산
            center_x = int((xmin + xmax) / 2)
            center_y = int((ymin + ymax) / 2)

            # 중심 좌표에 대한 깊이 값 추출
            depth_value = depth_frame.get_distance(center_x, center_y)

            # 유효한 깊이 값이 있으면 3D 좌표로 변환
            if depth_value > 0:
                point_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [center_x, center_y], depth_value)

                # TF 메시지 생성 및 퍼블리시
                t = TransformStamped()
                t.header.stamp = rospy.Time.now()
                t.header.frame_id = "camera_link"  # 카메라 좌표계 기준
                t.child_frame_id = "bottle"  # 객체의 tf 이름 설정

                t.transform.translation.x = point_3d[0]
                t.transform.translation.y = point_3d[1]
                t.transform.translation.z = point_3d[2]
                t.transform.rotation.x = 0.0  # 회전 정보는 필요 없다면 0으로 설정
                t.transform.rotation.y = 0.0
                t.transform.rotation.z = 0.0
                t.transform.rotation.w = 1.0

                # tf 브로드캐스트
                tf_broadcaster.sendTransform(t)

                # 객체 정보 출력
                print(f"Object: {label}, Depth: {depth_value:.2f}m, 3D Position: {point_3d}")

                # 감지된 객체의 바운딩 박스 시각화
                cv2.rectangle(annotated_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # 박스 그리기
                cv2.putText(annotated_image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 레이블 추가

        # 감지된 이미지 퍼블리시
        image_message = bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")
        image_pub.publish(image_message)

        # 감지된 이미지 출력
        cv2.imshow(f'YOLO Detection - {target_class_name}', annotated_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()

