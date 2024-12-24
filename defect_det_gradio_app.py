import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
import gradio as gr
from torchvision import models
import torch.nn as nn

# Load the Best Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['Defect', 'Not Defect']
model_ft = models.resnet18(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))
best_model_path = 'best_model.pth'
model_ft.load_state_dict(torch.load(best_model_path, map_location=device))
model_ft = model_ft.to(device)

# Function to apply CAM
def apply_cam(model, img_tensor, target_layer):
    model.eval()
    features_blobs = []

    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    handle = target_layer.register_forward_hook(hook_feature)
    output = model(img_tensor.to(device))
    handle.remove()

    # Get the softmax weights
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    # Get the class index
    _, predicted = torch.max(output, 1)
    class_idx = predicted.item()

    # Get the CAM
    cam = weight_softmax[class_idx].dot(features_blobs[0].reshape((features_blobs[0].shape[1], -1)))
    cam = cam.reshape(features_blobs[0].shape[2], features_blobs[0].shape[3])
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    return cam, class_idx

# Function to apply Grad-CAM
def apply_gradcam(model, img_tensor, target_layer):
    model.eval()
    features_blobs = []
    gradients = []

    def hook_feature(module, input, output):
        features_blobs.append(output)

    def hook_gradient(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_feature = target_layer.register_forward_hook(hook_feature)
    handle_gradient = target_layer.register_backward_hook(hook_gradient)
    output = model(img_tensor.to(device))

    # Get the class index
    _, predicted = torch.max(output, 1)
    class_idx = predicted.item()

    # Backward pass
    model.zero_grad()
    class_loss = output[0, class_idx]
    class_loss.backward()

    handle_feature.remove()
    handle_gradient.remove()

    # Get the gradients and features
    grads_val = gradients[0].cpu().data.numpy().squeeze()
    target = features_blobs[0].cpu().data.numpy().squeeze()

    # Get the Grad-CAM
    weights = np.mean(grads_val, axis=(1, 2))
    grad_cam = np.zeros(target.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        grad_cam += w * target[i, :, :]
    grad_cam = np.maximum(grad_cam, 0)
    grad_cam = grad_cam - np.min(grad_cam)
    grad_cam = grad_cam / np.max(grad_cam)
    grad_cam = np.uint8(255 * grad_cam)
    return grad_cam, class_idx

# Function to visualize CAM and Grad-CAM
def visualize_cam(img, model, target_layer):
    img = Image.fromarray(img).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(img).unsqueeze(0)

    cam, class_idx = apply_cam(model, img_tensor, target_layer)
    grad_cam, _ = apply_gradcam(model, img_tensor, target_layer)

    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    grad_cam = cv2.resize(grad_cam, (img.shape[1], img.shape[0]))

    heatmap_cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap_grad_cam = cv2.applyColorMap(grad_cam, cv2.COLORMAP_JET)

    result_cam = heatmap_cam * 0.4 + img * 0.6
    result_grad_cam = heatmap_grad_cam * 0.4 + img * 0.6

    return result_cam.astype(np.uint8), result_grad_cam.astype(np.uint8), class_names[class_idx]

# Gradio interface
def predict(img):
    target_layer = model_ft.layer4[1].conv2
    result_cam, result_grad_cam, prediction = visualize_cam(img, model_ft, target_layer)
    return prediction, result_cam, result_grad_cam

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Image(type="numpy", label="CAM"),
        gr.Image(type="numpy", label="Grad-CAM")
    ],
    title="Defect Detection with CAM and Grad-CAM",
    description="Upload an image to get the prediction along with CAM and Grad-CAM visualizations."
)

iface.launch()
