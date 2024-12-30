import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt



import numpy as np

# Load YOLOv3 model
model = torch.load('yolov3.onnx')
model.eval()

# Preprocess input image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor

# Perform object detection
def detect_objects(image_path):
    img_tensor = preprocess_image(image_path)
    
    with torch.no_grad():
        detections = model(img_tensor)

    # Process the detections (adjust based on the actual output format)
    # ...

    return detections

# Visualize results
def visualize_results(image_path, detections):
     # Rename images from data set.
    found_images = glob.glob(path_glob)

    for i, img_path in enumerate(found_images):
        print(f"Viewing Image: {i}")
        out_box = model(LoadImage(img_path)).type(torch.int32)[0, :].numpy()
        x, y, w, h = out_box[:4]
        bounds = ((x, y), (x + w, y + h))
        if (out_box < 0).any():
            print(f"Skipping: {img_path} because bounds are negative.")
            continue

        with Image.open(img_path) as im:
            draw_handle = ImageDraw.Draw(im)
            draw_handle.rectangle(bounds, outline="red")
            im = im.rotate(180, Image.NEAREST, expand=1)
            im.save(img_path.replace("data", "output"))

    # Display the image
    img = Image.open(image_path)
    plt.imshow(np.array(img))
    plt.show()

if __name__ == "__main__":
    image_path = 'path/to/your/image.jpg'
    detections = detect_objects(image_path)
    visualize_results(image_path, detections)
