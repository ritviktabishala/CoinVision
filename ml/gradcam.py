import torch
import numpy as np
import cv2

class GradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    #I included comments to show understanding
    
    def generate(self, image_tensor, class_idx=None):
        self.model.eval()
        # 1. Forward Pass
        outputs = self.model(image_tensor)

        if class_idx is None:
            _, class_idx = torch.max(outputs, dim=1)
            class_idx = int(class_idx.item())
        
        self.model.zero_grad()

        # 2. Backward Pass
        score = outputs[:, class_idx]
        score.backward() 

        # 3. Channel Pooling (Weights Generation)
        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # 4. Filter and Stacking
        cam = (weights * self.activations).sum(dim=1)

        cam = cam.squeeze(0)

        # 5. ReLU Positive Filtering
        cam = torch.relu(cam)

        # 6. Min-Max Normalization
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam.cpu().numpy()
    
    
def create_heatmap(cam, h=224, w=224):

    heatmap = cv2.resize(cam, (w, h))

    heatmap_uint8 = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    return heatmap_color


def overlay_heatmap(image_np, heatmap):
    overlay = cv2.addWeighted(
        image_np,
        0.6,
        heatmap,
        0.4,
        0
    )
    return overlay


def generate_gradcam(
    image_tensor,
    original_image,  
    model,
    target_layer,
    class_idx=None
):

    gradcam = GradCam(model, target_layer)
    cam = gradcam.generate(image_tensor, class_idx)
    gradcam.remove_hooks()

    image_np = np.array(original_image, dtype=np.uint8)
    h, w, _ = image_np.shape

    heatmap = create_heatmap(cam, h=h, w=w)
    overlay = overlay_heatmap(image_np, heatmap)

    return overlay