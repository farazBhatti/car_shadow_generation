import os
import cv2
import torch
import numpy as np
from glob import glob
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from ultralytics import YOLO  # <--- NEW IMPORT

# ==========================================
# CONFIGURATION
# ==========================================

# --- BiRefNet Settings ---
USE_HALF_PRECISION = False 
MODEL_NAME = 'BiRefNet' 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- YOLO Settings ---
LICENSE_PLATE_MODEL_PATH = 'license_plate_model.pt' 
CLASSIFIER_MODEL_PATH = 'classifier.pt' # <--- NEW: Path to your classifier
TARGET_CLASS = "ext+shadow+rmbg"        # <--- NEW: Target class to process

# --- Pipeline Directories ---
SOURCE_IMAGES_DIR = "car_images"
MASKS_DIR = "masks"
NO_BG_DIR = "images_background_removed"

COMBINE_DIRS = {
    "step1": "1_full_hull",                  
    "step2": "2_center_line_vis",            
    "step3": "3_filtered_hull",              
    "step3b": "4_longest_lines_output",
    "step4_shadow": "5_shadow_only",         
    "step5_final": "6_final_composite",      
}

# ==========================================
# MODEL INITIALIZATION (BiRefNet)
# ==========================================

print(f"Initializing {MODEL_NAME} on {DEVICE}...")
torch.set_float32_matmul_precision('high')

try:
    birefnet = AutoModelForImageSegmentation.from_pretrained(
        f'zhengpeng7/{MODEL_NAME}', 
        trust_remote_code=True
    )
    birefnet.to(DEVICE)
    birefnet.eval()
    if USE_HALF_PRECISION:
        birefnet.half()
    print(f"{MODEL_NAME} loaded successfully.")
except Exception as e:
    print(f"Error loading BiRefNet: {e}")
    exit()

# Define Transforms
target_size = (2048, 2048) if '_HR' in MODEL_NAME else (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_mask_center_y(mask_img):
    mask_f = mask_img.astype(np.float32) / 255.0
    row_sum = np.sum(mask_f, axis=1)
    cumulative = np.cumsum(row_sum)
    total = cumulative[-1]
    if total == 0: return mask_img.shape[0] // 2 
    center_y = np.searchsorted(cumulative, total / 2)
    return center_y

def get_line_intersection(line1, line2):
    p1, p2 = line1[0], line1[1]
    p3, p4 = line2[0], line2[1]

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0: return None 

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    
    return (int(px), int(py))

# ==========================================
# PIPELINE STEPS
# ==========================================

def step_1_generate_masks():
    print(f"\n--- [Step 1] Generating Masks using {MODEL_NAME} with Classifier Check ---")
    if not os.path.exists(SOURCE_IMAGES_DIR):
        print(f"Directory '{SOURCE_IMAGES_DIR}' not found.")
        return
    os.makedirs(MASKS_DIR, exist_ok=True)
    
    # --- Load Classifier ---
    print(f"Loading Classifier: {CLASSIFIER_MODEL_PATH}")
    classifier = None
    try:
        classifier = YOLO(CLASSIFIER_MODEL_PATH)
        print(f"  -> Classifier loaded.")
    except Exception as e:
        print(f"  -> Error loading classifier: {e}")
        print("  -> CAUTION: Proceeding without classification check? (Script will stop)")
        return

    files = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.webp'):
        files.extend(glob(os.path.join(SOURCE_IMAGES_DIR, ext)))
    
    for input_path in files:
        filename = os.path.basename(input_path)
        output_path = os.path.join(MASKS_DIR, f"{os.path.splitext(filename)[0]}_mask.png")
        if os.path.exists(output_path): continue

        print(f"Processing: {filename}...")
        try:
            # 1. Run Classification First
            # We open with PIL here for consistency, YOLO handles PIL inputs fine.
            image = Image.open(input_path).convert("RGB")
            
            # Run inference
            results = classifier(image, verbose=False)
            
            # Extract top class
            top_class_index = results[0].probs.top1
            top_class_name = results[0].names[top_class_index]
            
            if top_class_name != TARGET_CLASS:
                print(f"  -> Skipping: Detected class '{top_class_name}' (Expected '{TARGET_CLASS}')")
                continue # Skip mask generation

            print(f"  -> Accepted: Class is '{top_class_name}'")

            # 2. Proceed to BiRefNet Mask Generation
            original_size = image.size
            input_tensor = transform_image(image).unsqueeze(0).to(DEVICE)
            if USE_HALF_PRECISION: input_tensor = input_tensor.half()
            
            with torch.no_grad():
                preds = birefnet(input_tensor)[-1].sigmoid().cpu()
            
            pred_pil = transforms.ToPILImage()(preds[0].squeeze())
            mask = pred_pil.resize(original_size, resample=Image.BICUBIC)
            mask.save(output_path)
            
        except Exception as e:
            print(f"  -> Failed: {e}")

def step_2_apply_masks():
    print(f"\n--- [Step 2] Applying Masks ---")
    os.makedirs(NO_BG_DIR, exist_ok=True)
    
    image_paths = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.webp'):
        image_paths.extend(glob(os.path.join(SOURCE_IMAGES_DIR, ext)))
    
    for img_path in image_paths:
        file_id = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(MASKS_DIR, f"{file_id}_mask.png")
        output_path = os.path.join(NO_BG_DIR, f"{file_id}_no_bg.png")
        
        if os.path.exists(output_path): continue
        # If classifier skipped the file in Step 1, mask_path won't exist, so this safely skips.
        if not os.path.exists(mask_path): continue

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None: continue
        
        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            
        b, g, r = cv2.split(img)
        rgba = cv2.merge([b, g, r, mask])
        cv2.imwrite(output_path, rgba)


# Handles adaptive blur via area of parralelogram
# Has a debug step to save with white background as well.

def step_3_process_and_shadow():
    print(f"\n--- [Step 3] Shadow Generation, Overlay & License Plate Blur ---")
    
    # --- Load YOLO Model ---
    print(f"Loading License Plate Model: {LICENSE_PLATE_MODEL_PATH}")
    lp_model = None
    if os.path.exists(LICENSE_PLATE_MODEL_PATH):
        try:
            lp_model = YOLO(LICENSE_PLATE_MODEL_PATH)
            print("  -> Model loaded.")
        except Exception as e:
            print(f"  -> Failed to load YOLO model: {e}")
    else:
        print(f"  -> Warning: {LICENSE_PLATE_MODEL_PATH} not found. Skipping blur.")

    for d in COMBINE_DIRS.values():
        os.makedirs(d, exist_ok=True)

    # Note: Step 3 iterates only over generated masks. 
    # If Step 1 skipped a file due to classification, no mask exists, so Step 3 automatically skips it.
    mask_paths = sorted(glob(os.path.join(MASKS_DIR, "*_mask.png")))

    for mask_path in mask_paths:
        base_name = os.path.basename(mask_path)
        image_id = base_name.replace("_mask.png", "")
        print(f"Processing: {image_id}")
        
        # 1. Load Mask
        mask_orig = cv2.imread(mask_path)
        if mask_orig is None: continue

        # Robust check: If loaded as color (3 channels), convert to grayscale
        if len(mask_orig.shape) == 3:
            mask_orig = cv2.cvtColor(mask_orig, cv2.COLOR_BGR2GRAY)
        
        h, w = mask_orig.shape
        
        # 2. Binary mask for Hull
        _, mask_bin = cv2.threshold(mask_orig, 127, 255, cv2.THRESH_BINARY)
        center_y = get_mask_center_y(mask_bin)

        # 3. Hull Calculation
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_edges = [] 
        for cnt in contours:
            hull = cv2.convexHull(cnt)
            if hull is None or len(hull) < 2: continue
            hull_points = hull.reshape(-1, 2)
            n = len(hull_points)
            for i in range(n):
                pt1 = tuple(hull_points[i])
                pt2 = tuple(hull_points[(i+1) % n])
                length = np.hypot(pt1[0]-pt2[0], pt1[1]-pt2[1])
                all_edges.append((pt1, pt2, length))

        # --- Debug Saves (Steps 1, 2, 3) ---
        canvas_step1 = cv2.cvtColor(mask_orig, cv2.COLOR_GRAY2BGR)
        for (pt1, pt2, _) in all_edges:
            cv2.line(canvas_step1, pt1, pt2, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(COMBINE_DIRS["step1"], f"{image_id}_step1.png"), canvas_step1)

        canvas_step2 = cv2.cvtColor(mask_orig, cv2.COLOR_GRAY2BGR)
        cv2.line(canvas_step2, (0, center_y), (w, center_y), (255, 255, 0), 2) 
        cv2.imwrite(os.path.join(COMBINE_DIRS["step2"], f"{image_id}_step2.png"), canvas_step2)

        background_edges = []
        canvas_step3 = cv2.cvtColor(mask_orig, cv2.COLOR_GRAY2BGR)
        for (pt1, pt2, length) in all_edges:
            if pt1[1] > center_y and pt2[1] > center_y:
                background_edges.append((pt1, pt2, length))
                cv2.line(canvas_step3, pt1, pt2, (255, 0, 0), 2)
        cv2.imwrite(os.path.join(COMBINE_DIRS["step3"], f"{image_id}_step3.png"), canvas_step3)

        # 5. Sort and take Top 2 Longest Lines
        background_edges.sort(key=lambda x: x[2], reverse=True)
        top_edges = background_edges[:2]

        if len(top_edges) < 2:
            print(f"  -> Not enough bottom edges found for {image_id}")
            continue

        canvas_step3b = cv2.cvtColor(mask_orig, cv2.COLOR_GRAY2BGR)
        for (pt1, pt2, _) in background_edges:
            cv2.line(canvas_step3b, pt1, pt2, (0, 100, 0), 1)
        cv2.line(canvas_step3b, top_edges[0][0], top_edges[0][1], (0, 0, 255), 2)
        cv2.line(canvas_step3b, top_edges[1][0], top_edges[1][1], (0, 255, 255), 2)
        cv2.imwrite(os.path.join(COMBINE_DIRS["step3b"], f"{image_id}_step3b.png"), canvas_step3b)

        # 6. Form Parallelogram (Shadow)
        line_a = (top_edges[0][0], top_edges[0][1])
        line_b = (top_edges[1][0], top_edges[1][1])

        shadow_mask_alpha = np.zeros((h, w), dtype=np.uint8)
        intersection_pt = get_line_intersection(line_a, line_b)
        
        # Variable to store area for adaptive blur
        shadow_area = 0.0

        if intersection_pt:
            ix, iy = intersection_pt
            dist_a0 = np.hypot(line_a[0][0] - ix, line_a[0][1] - iy)
            dist_a1 = np.hypot(line_a[1][0] - ix, line_a[1][1] - iy)
            far_pt_a = line_a[0] if dist_a0 > dist_a1 else line_a[1]

            dist_b0 = np.hypot(line_b[0][0] - ix, line_b[0][1] - iy)
            dist_b1 = np.hypot(line_b[1][0] - ix, line_b[1][1] - iy)
            far_pt_b = line_b[0] if dist_b0 > dist_b1 else line_b[1]

            p4_x = far_pt_a[0] + (far_pt_b[0] - ix)
            p4_y = far_pt_a[1] + (far_pt_b[1] - iy)
            fourth_point = (int(p4_x), int(p4_y))

            shape_points_np = np.array([intersection_pt, far_pt_a, fourth_point, far_pt_b], np.int32)

            obj_x, obj_y, obj_w, obj_h = cv2.boundingRect(mask_bin)
            shift_amount_y = int(obj_h * 0.05)
            shape_points_np[:, 1] -= shift_amount_y 
            final_shape_points = shape_points_np.reshape((-1, 1, 2))

            # Calculate Area of the shadow parallelogram
            shadow_area = cv2.contourArea(final_shape_points)

            cv2.fillPoly(shadow_mask_alpha, [final_shape_points], color=255)

        else:
            print('*****************************************************')
            cv2.line(shadow_mask_alpha, line_a[0], line_a[1], 255, 3)
            cv2.line(shadow_mask_alpha, line_b[0], line_b[1], 255, 3)
            # Fallback area: approx sum of line lengths * arbitrary width
            len_a = np.hypot(line_a[0][0]-line_a[1][0], line_a[0][1]-line_a[1][1])
            shadow_area = len_a * 10 

        # Adaptive Blur based on Shadow Area
        if shadow_area > 0:
            # We use sqrt because area is 2D and kernel size is 1D.
            # Multiplier 0.6 is a heuristic to get a nice soft shadow relative to size.
            k_size = int(np.sqrt(shadow_area) * 0.6)
        else:
            print('*****************************************************')
            k_size = int(max(h, w) * 0.08) # Fallback to old logic

        if k_size % 2 == 0: k_size += 1
        if k_size < 3: k_size = 3
        
        blur_kernel_size = (k_size, k_size) 
        print(f"    -> Shadow Area: {shadow_area:.0f}px, Adaptive Blur Kernel: {k_size}")
        
        blurred_alpha = cv2.GaussianBlur(shadow_mask_alpha, blur_kernel_size, 0)
        
        # Shadow Only (Transparent)
        shadow_only_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        shadow_only_rgba[:, :, 3] = blurred_alpha 
        cv2.imwrite(os.path.join(COMBINE_DIRS["step4_shadow"], f"{image_id}_shadow_only.png"), shadow_only_rgba)

        # ===============================================
        # OVERLAY CAR IMAGE (TRANSPARENT RESULT)
        # ===============================================
        car_no_bg_path = os.path.join(NO_BG_DIR, f"{image_id}_no_bg.png")
        
        if os.path.exists(car_no_bg_path):
            car_rgba = cv2.imread(car_no_bg_path, cv2.IMREAD_UNCHANGED)
            if car_rgba is not None:
                if car_rgba.shape[:2] != (h, w):
                    car_rgba = cv2.resize(car_rgba, (w, h))

                car_alpha = car_rgba[:, :, 3].astype(float) / 255.0
                shadow_alpha = blurred_alpha.astype(float) / 255.0
                
                final_alpha = car_alpha + shadow_alpha * (1.0 - car_alpha)
                car_bgr = car_rgba[:, :, :3].astype(float)
                
                final_bgr = np.zeros_like(car_bgr)
                mask_nonzero = final_alpha > 0
                
                for c in range(3):
                    final_bgr[:, :, c][mask_nonzero] = (
                        (car_bgr[:, :, c][mask_nonzero] * car_alpha[mask_nonzero]) / 
                        final_alpha[mask_nonzero]
                    )

                final_output = cv2.merge([
                    final_bgr[:, :, 0].astype(np.uint8),
                    final_bgr[:, :, 1].astype(np.uint8),
                    final_bgr[:, :, 2].astype(np.uint8),
                    (final_alpha * 255).astype(np.uint8)
                ])
            else:
                final_output = shadow_only_rgba
        else:
            final_output = shadow_only_rgba

        # ===============================================
        # NEW: DETECT AND BLUR LICENSE PLATES
        # ===============================================
        if lp_model is not None:
            detect_img = cv2.cvtColor(final_output, cv2.COLOR_RGBA2RGB)
            results = lp_model(detect_img, verbose=False)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf > 0.35:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        roi_w = x2 - x1
                        roi_h = y2 - y1
                        
                        k_blur = int(roi_h * 0.8)
                        if k_blur % 2 == 0: k_blur += 1 
                        if k_blur < 3: k_blur = 3
                        
                        roi = final_output[y1:y2, x1:x2]
                        blurred_roi = cv2.GaussianBlur(roi, (k_blur, k_blur), 0)
                        final_output[y1:y2, x1:x2] = blurred_roi
                        print(f"    -> License Plate Blurred (Conf: {conf:.2f})")

        # Save Final Result
        save_path = os.path.join(COMBINE_DIRS["step5_final"], f"{image_id}_final_composite.png")
        cv2.imwrite(save_path, final_output)
        print(f"  -> Saved Final (Transparent): {save_path}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("====================================")
    print("      GEOMETRY SHAPE PIPELINE       ")
    print("====================================")
    
    step_1_generate_masks()
    step_2_apply_masks() 
    step_3_process_and_shadow()
    
    print(f"\nAll operations complete.")
    print(f"Check '{COMBINE_DIRS['step5_final']}' for final composites.")
