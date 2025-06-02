import cv2
import torch 

def compute_edge_map(img_tensor):
    # img_tensor: CxHxW normalized [0,1]
    
    np_img = (img_tensor.permute(1,2,0).cpu().numpy() * 255).astype('uint8')
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return torch.from_numpy(edges / 255.).unsqueeze(0).float().to(img_tensor.device)