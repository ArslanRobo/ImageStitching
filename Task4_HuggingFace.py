"""We can use Gradio to build the UI and then make it compatible for the Hugging face."""
import gradio as gr
import cv2
import numpy as np
import imutils
from PIL import Image

cv2.ocl.setUseOpenCL(False)

# Sharpening function
def image_sharpening(image):
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel_sharpening)
    return sharpened
# this is a test commit section

# Remove black borders function
def remove_black_region(result):
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)
    crop = result[y:y + h, x:x + w]
    return crop

# Key point detection and descriptor function
def detectAndDescribe(image, method='orb'):
    if method == 'sift':
        descriptor = cv2.SIFT_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()
    (kps, features) = descriptor.detectAndCompute(image, None)
    return kps, features

# Matcher creation
def createMatcher(method, crossCheck):
    if method in ['sift', 'surf']:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf

# Matching key points
def matchKeyPointsKNN(featuresA, featuresB, ratio, method):
    bf = createMatcher(method, crossCheck=False)
    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    matches = []
    for m, n in rawMatches:
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches

# Homography calculation
def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh=4.0):
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])
    if len(matches) > 4:
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
        return matches, H, status
    else:
        return None

# Stitching function for two images
def stitch_two_images(queryImg, trainImg, feature_extractor):
    queryImg_gray = cv2.cvtColor(queryImg, cv2.COLOR_BGR2GRAY)
    trainImg_gray = cv2.cvtColor(trainImg, cv2.COLOR_BGR2GRAY)
    kpsA, featuresA = detectAndDescribe(trainImg_gray, method=feature_extractor)
    kpsB, featuresB = detectAndDescribe(queryImg_gray, method=feature_extractor)
    matches = matchKeyPointsKNN(featuresA, featuresB, ratio=0.75, method=feature_extractor)
    M = getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh=5)
    if M is None:
        return None
    (matches, H, status) = M
    width = trainImg.shape[1] + queryImg.shape[1]
    height = trainImg.shape[0] + queryImg.shape[0]
    result = cv2.warpPerspective(trainImg, H, (width, height))
    result[0:queryImg.shape[0], 0:queryImg.shape[1]] = queryImg
    crop_image = remove_black_region(result)
    return crop_image

# Calculate target brightness
def calculate_target_brightness(images):
    brightness_values = [np.mean(image.astype(np.float32)) for image in images]
    return np.mean(brightness_values)

# Brightness adjustment
def global_brightness_adjustment(images, target_brightness):
    adjusted_images = []
    for image in images:
        image_float = image.astype(np.float32)
        avg_brightness = np.mean(image_float)
        brightness_shift = target_brightness - avg_brightness
        adjusted_image = image_float + brightness_shift
        adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)
        adjusted_images.append(adjusted_image)
    return adjusted_images   

# Main Stitching function
def stitch_images(uploaded_files, feature_extractor):
    images = [cv2.cvtColor(np.array(Image.open(file)), cv2.COLOR_RGB2BGR) for file in uploaded_files]
    if len(images) == 0:
        return None
    target_brightness = calculate_target_brightness(images)
    adjusted_images = global_brightness_adjustment(images, target_brightness)
    stitched_image = adjusted_images[0]
    for i in range(1, len(adjusted_images)):
        queryImg = stitched_image
        trainImg = adjusted_images[i]
        stitched_image = stitch_two_images(queryImg, trainImg, feature_extractor)
    return cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB)
  
# Gradio interface with feature extractor selector
# Works well for two input images
with gr.Blocks() as demo:
    gr.Markdown("## Image Stitching App with Feature Extractor Selection")
    image_input = gr.Files(label="Upload Images", type="filepath")
    extractor_input = gr.Dropdown(choices=["orb", "sift", "brisk"], label="Feature Extractor", value="orb")
    image_output = gr.Image(type="numpy", label="Stitched Image")
    process_button = gr.Button("Process Image")
    process_button.click(stitch_images, inputs=[image_input, extractor_input], outputs=image_output)

# Launch the Gradio app
demo.launch()
