import streamlit as st
import cv2
import numpy as np
from collections import defaultdict
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.torch_utils import select_device
import torch
import time

# Import ReportLab for PDF generation
# from reportlab.lib.pagesizes import letter
# from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image
from reportlab.lib.units import inch
from io import BytesIO
from PIL import Image as PILImage  # Import the correct Image class

# Initialize the Streamlit app
st.title("CAD App")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    img0 = cv2.imdecode(image, 1)
    st.image(img0, caption='Uploaded Image', use_column_width=True)

    # Button to start object detection
    if st.button("Detect Objects"):
        # Placeholder for progress bar
        progress_bar = st.progress(0)

        # Progress counter
        progress_counter = 0

        # Model and image settings
        weights = 'best.pt'  # Change to your custom model name
        img_size = 640  # Set image size
        conf_thres = 0.4  # Confidence threshold
        iou_thres = 0.5  # IoU threshold
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available, else use CPU

        # Load model
        model = attempt_load(weights, map_location=device)
        model.eval()

        # Set device
        device = select_device(device)

        # Initialize combined image
        combined_img = img0.copy()

        # Define tile size and stride
        tile_size = img0.shape[0] // 5  # Assuming equal division for simplicity
        stride = tile_size

        # Process the uploaded image
        img = letterbox(img0, new_shape=img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)[0]

        # Initialize counts dictionary
        obj_counts = defaultdict(int)

        # Class names from data.yaml
        class_names = ['10000-lm-low-bay-light', '10A-1way-1gang-switch', '10A-2way-1gang-switch', '1250-lm-led',
               '13A-socket-outlet-for-higher-level', '13A-twin-table-top-mounted-socket-outlet',
               '13A-wall-mounted-socket-outlet', '13A-wall-mounted-twin-socket-outlet', '1400mm-sweap-fan',
               '2345-lm-led-light', '3250-lm-led', '3800-lm-1200mm-led', '500-lm-law-bay-led', '600-lm-led',
               '600-lm-low-bay-led', '910-Lm-led-down-light', '910-lm-wall-mounted-led', '950-lm-led', 'MCCB',
               'energency-light', 'isolator-with-enclosure', 'metal-distribution-boad', 'universal-fan-controller',
               'wall-fan']  # Update with your class names

       # Process tiles
        for y in range(0, img0.shape[0], stride):
            for x in range(0, img0.shape[1], stride):
                # Extract tile
                tile = img0[y:y + tile_size, x:x + tile_size]

                # Resize tile to model input size
                tile_resized = cv2.resize(tile, (img_size, img_size))

                # Normalize RGB
                img = tile_resized[:, :, ::-1].transpose(2, 0, 1).copy()  # Make a copy to avoid negative strides
                img = torch.from_numpy(img).float().to(device) / 255.0
                img = img.unsqueeze(0)

                # Inference
                pred = model(img)[0]
                pred = non_max_suppression(pred, conf_thres, iou_thres)[0]

                # Update progress
                progress_counter += 1
                progress_bar.progress(min(int(progress_counter * 100 / (img0.shape[0] * img0.shape[1] / (stride ** 2))), 100))


                # Process detections
                if pred is not None and len(pred):
                    for det in pred:
                        det = det.detach().cpu().numpy()  # Convert to numpy array
                        bbox = det[:4]  # Extract bounding box coordinates
                        bbox[0::2] = (bbox[0::2] * tile_size / img_size + x).clip(min=0, max=combined_img.shape[1])  # Scale and adjust x-coordinates
                        bbox[1::2] = (bbox[1::2] * tile_size / img_size + y).clip(min=0, max=combined_img.shape[0])  # Scale and adjust y-coordinates
                        bbox = bbox.round().astype(np.int)  # Round and convert to integer
                        label = int(det[-1])
                        obj_counts[label] += 1  # Increment count for detected object
                                
                        # Draw bounding box on combined image
                        cv2.rectangle(combined_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        
                        # Overlay text with class name, ID, and count
                        class_name = class_names[label] if label < len(class_names) else 'Unknown'
                        text = f'{class_name} ({label}): {obj_counts[label]}'
                        
                        # Reduce text size
                        font_scale = 0.5
                        thickness = 1
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                        
                        # Draw text background rectangle
                        cv2.rectangle(combined_img, (bbox[0], bbox[1] - text_size[1]), (bbox[0] + text_size[0], bbox[1]), (0, 255, 0), cv2.FILLED)
                        
                        # Overlay text on the image
                        cv2.putText(combined_img, text, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)



        # Display object counts and percentage
        # st.subheader("Object Counts")
        total_objects = sum(obj_counts.values())
        detected_objects = len(pred) if pred is not None else 0
        detection_percentage = (detected_objects / total_objects) * 100 if total_objects > 0 else 0

        


        # Display the combined image with detections and text overlay
        st.subheader("Combined Image with Detections")
        st.image(combined_img, channels="BGR")




            # Display object counts
        st.subheader("Object Counts")
        for label, count in obj_counts.items():
            class_name = class_names[label] if label < len(class_names) else 'Unknown'
            st.write(f"{class_name}: {count}")

        # Create PDF table
        doc = SimpleDocTemplate("object_counts.pdf", pagesize=letter)
        data = [["Class Name", "Count"]]
        for label, count in obj_counts.items():
            class_name = class_names[label] if label < len(class_names) else 'Unknown'
            data.append([class_name, count])

        table = Table(data)
        table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), (0, 0, 0)),
                                ('TEXTCOLOR', (0, 0), (-1, 0), (1, 1, 1)),  # White text color
                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                ('BACKGROUND', (0, 1), (-1, -1), (0.96, 0.96, 0.96))]))  # Light gray background

        # doc.build([table])

       # Build the PDF with the table
        elements = [table]

        # Add the combined_img to the PDF
        img_byte_io = BytesIO()
        combined_img_rgb = combined_img[:, :, ::-1]  # Convert BGR to RGB
        img_pil = PILImage.fromarray(combined_img_rgb)  # Convert NumPy array to PIL Image
        img_pil.save(img_byte_io, format='PNG')  # Save image to byte stream
        img_byte_io.seek(0)
        img = Image(img_byte_io, width=6 * inch, height=4 * inch)  # Adjust width and height as needed
        elements.append(img)

        doc.build(elements)

        # Provide download link for the PDF
        with open("object_counts.pdf", "rb") as pdf_file:
            pdf_bytes = pdf_file.read()

        st.download_button(label="Download PDF", data=pdf_bytes, file_name="object_counts.pdf", mime="application/pdf")
                        # Free unused memory

torch.cuda.empty_cache()

        # st.write(f"Total Objects: {total_objects}")
        # st.write(f"Detected Objects: {detected_objects}")
        # st.write(f"Detection Percentage: {detection_percentage:.2f}%")

        