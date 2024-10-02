import pyrealsense2 as rs
import numpy as np
import cv2
import rospy
from ultralytics import YOLO
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image
import tf2_ros
from cv_bridge import CvBridge

# Initialize ROS node
rospy.init_node('object_tf_publisher')

# Set up the TF broadcaster
tf_broadcaster = tf2_ros.TransformBroadcaster()

# Set up the image publisher
image_pub = rospy.Publisher("/detection_image", Image, queue_size=10)

# Create CvBridge object for converting between ROS and OpenCV images
bridge = CvBridge()

# Configure the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Load the YOLO model
model = YOLO('yolov8s.pt')
class_names = model.names  

# Set the target object class name
target_class_name = 'bottle'

# Align depth to color frames
align_to = rs.stream.color
align = rs.align(align_to)

# Start the RealSense pipeline
pipeline.start(config)

try:
    while not rospy.is_shutdown():
        # Get frames from the camera
        frames = pipeline.wait_for_frames()
        
        # Align the frames (align color and depth frames)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Perform object detection using YOLO
        results = model(color_image)

        # Filter only the target class objects
        filtered_boxes = [box for box in results[0].boxes if class_names[int(box.cls)] == target_class_name]

        # Visualize and extract depth information for the target class objects
        annotated_image = color_image.copy()
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics  # Depth image intrinsic parameters

        for box in filtered_boxes:
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0])  # Convert coordinates to integers
            confidence = box.conf.item()  # Convert tensor to Python float
            label = f"{class_names[int(box.cls)]} {confidence:.2f}"  # Class name and confidence label

            # Calculate the center coordinates of the bounding box
            center_x = int((xmin + xmax) / 2)
            center_y = int((ymin + ymax) / 2)

            # Extract the depth value at the center of the bounding box
            depth_value = depth_frame.get_distance(center_x, center_y)

            # If a valid depth value exists, convert to 3D coordinates
            if depth_value > 0:
                point_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [center_x, center_y], depth_value)

                # Create and publish the TF message
                t = TransformStamped()
                t.header.stamp = rospy.Time.now()
                t.header.frame_id = "camera_link"  # Reference the camera coordinate frame
                t.child_frame_id = "bottle"  # Set the object's tf name

                t.transform.translation.x = point_3d[0]
                t.transform.translation.y = point_3d[1]
                t.transform.translation.z = point_3d[2]
                t.transform.rotation.x = 0.0  # Set rotation information (if not needed, leave as 0)
                t.transform.rotation.y = 0.0
                t.transform.rotation.z = 0.0
                t.transform.rotation.w = 1.0

                # Broadcast the TF
                tf_broadcaster.sendTransform(t)

                # Print object information
                print(f"Object: {label}, Depth: {depth_value:.2f}m, 3D Position: {point_3d}")

                # Visualize the detected object's bounding box
                cv2.rectangle(annotated_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Draw bounding box
                cv2.putText(annotated_image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Add label

        # Publish the annotated image
        image_message = bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")
        image_pub.publish(image_message)

        # Display the annotated image
        cv2.imshow(f'YOLO Detection - {target_class_name}', annotated_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Stop the pipeline and close windows
    pipeline.stop()
    cv2.destroyAllWindows()
