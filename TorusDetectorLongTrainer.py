from TorusDetectorLong import TorusDetector  # Adjust the import statement based on your file structure
import os
import json
import cv2

def process_and_save_image(input_path, output_path, torus_detector, coco_data):
    # Read the image
    image = cv2.imread(input_path)

    # Detect rectangles
    rectangles = torus_detector(image)

    # Create entry for the current image in COCO-style format
    image_info = {
        "file_name": os.path.basename(input_path),
        "height": image.shape[0],
        "width": image.shape[1],
        "id": len(coco_data["images"]) + 1,
        "license": None,
        "flickr_url": None,
        "coco_url": None,
        "date_captured": None,
    }

    # Add image info to COCO data
    coco_data["images"].append(image_info)

    # Process rectangles and add annotations to COCO data
    count = 0
    if rectangles:
        count = len(rectangles)
        for rect_id, rect in enumerate(rectangles):
            x, y, w, h = rect
            annotation_info = {
                "id": len(coco_data["annotations"]) + 1,
                "image_id": image_info["id"],
                "category_id": 1,  # Assuming one category for rectangles
                "segmentation": [],
                "area": w * h,
                "bbox": [x, y, w, h],
                "iscrowd": 0,
            }
            coco_data["annotations"].append(annotation_info)

            # Visualize and save the rectangles on the image
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    text = "Number of Detected Rectangles: " + str(count)
    cv2.putText(image, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Save the labeled image
    output_image_path = os.path.join(output_path, os.path.basename(input_path))
    cv2.imwrite(output_image_path, image)

    # Generate and save the masked image
    mask = torus_detector.generate_color_mask(image)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    masked_image_path = os.path.join(output_path, "masked_" + os.path.basename(input_path))
    cv2.imwrite(masked_image_path, masked_image)

def process_folder(input_folder, output_folder, torus_detector):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # COCO-style data structure
    coco_data = {
        "info": None,
        "licenses": None,
        "categories": [{"id": 1, "name": "rectangle", "supercategory": "object"}],
        "images": [],
        "annotations": [],
    }

    # Loop through all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_folder, filename)
            process_and_save_image(input_path, output_folder, torus_detector, coco_data)

    # Save COCO-style JSON file
    with open(os.path.join(output_folder, "coco_annotations.json"), "w") as json_file:
        json.dump(coco_data, json_file, indent=2)

if __name__ == "__main__":
    input_folder_path = "/Users/paras/downloads/brt/jan_20_capture_2"
    output_folder_path = "/Users/paras/downloads/LabelledStuff"
    torus_detector = TorusDetector()

    process_folder(input_folder_path, output_folder_path, torus_detector)
