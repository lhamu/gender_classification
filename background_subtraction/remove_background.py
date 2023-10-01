import os
import cv2
from rembg import remove
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# import cvzone
# from cvzone.SelfiSegmentationModule import SelfiSegmentation

def showimage(myimage):
    if (myimage.ndim>2):  #This only applies to RGB or RGBA images (e.g. not to Black and White images)
        myimage = myimage[:,:,::-1] #OpenCV follows BGR order, while matplotlib likely follows RGB order
         
    fig, ax = plt.subplots(figsize=[10,10])
    ax.imshow(myimage, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

def bgremove2(myimage):
    # First Convert to Grayscale
    myimage_grey = cv2.cvtColor(myimage, cv2.COLOR_BGR2GRAY)
 
    ret,baseline = cv2.threshold(myimage_grey,127,255,cv2.THRESH_TRUNC)
 
    ret,background = cv2.threshold(baseline,126,255,cv2.THRESH_BINARY)
 
    ret,foreground = cv2.threshold(baseline,126,255,cv2.THRESH_BINARY_INV)
 
    foreground = cv2.bitwise_and(myimage,myimage, mask=foreground)  # Update foreground with bitwise_and to extract real foreground
 
    # Convert black and white back into 3 channel greyscale
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
 
    # Combine the background and foreground to obtain our final image
    finalimage = background+foreground
    return finalimage

def bgremove3(myimage):
    # BG Remover 3
    myimage_hsv = cv2.cvtColor(myimage, cv2.COLOR_BGR2HSV)
     
    #Take S and remove any value that is less than half
    s = myimage_hsv[:,:,1]
    s = np.where(s < 127, 0, 1) # Any value below 127 will be excluded
 
    # We increase the brightness of the image and then mod by 255
    v = (myimage_hsv[:,:,2] + 127) % 255
    v = np.where(v > 127, 1, 0)  # Any value above 127 will be part of our mask
 
    # Combine our two masks based on S and V into a single "Foreground"
    foreground = np.where(s+v > 0, 1, 0).astype(np.uint8)  #Casting back into 8bit integer
 
    background = np.where(foreground==0,255,0).astype(np.uint8) # Invert foreground to get background in uint8
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)  # Convert background back into BGR space
    foreground=cv2.bitwise_and(myimage,myimage,mask=foreground) # Apply our foreground map to original image
    finalimage = background+foreground # Combine foreground and background
 
    return finalimage

def bg_remove_test(myimage):
    # blur
    blur = cv2.GaussianBlur(myimage, (3,3), 0)
    
    # convert to hsv and get saturation channel
    sat = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)[:,:,1]
    
    # threshold saturation channel
    thresh = cv2.threshold(sat, 50, 255, cv2.THRESH_BINARY)[1]
    
    # apply morphology close and open to make mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # do OTSU threshold to get circuit image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    
    # write black to otsu image where mask is black
    otsu_result = otsu.copy()
    otsu_result[mask==0] = 0
    
    # write black to input image where mask is black
    img_result = img.copy()
    img_result[mask==0] = 1
    
    # write result to disk
    cv2.imwrite("circuit_board_mask.png", mask)
    cv2.imwrite("circuit_board_otsu.png", otsu)
    cv2.imwrite("circuit_board_otsu_result.png", otsu_result)
    cv2.imwrite("circuit_board_img_result.png", img_result)
    
    
    # display it
    cv2.imshow("IMAGE", img)
    cv2.imshow("SAT", sat)
    cv2.imshow("MASK", mask)
    cv2.imshow("OTSU", otsu)
    cv2.imshow("OTSU_RESULT", otsu_result)
    cv2.imshow("IMAGE_RESULT", img_result)
    cv2.waitKey(0)
    
    return img_result

def remove_background_using_rembg(input_path, output_path):
    input = Image.open(input_path)
    output = remove(input)
    output.save(output_path)

# folder_path = r"D:\research_internship_data\complete_males"
# destination_path = r"D:\research_internship_data\cleaned_male_images"

folder_path = r"C:\Users\upech\Downloads\OneDrive_1_4-5-2023\female\all_females"
destination_path = r"C:\Users\upech\Downloads\OneDrive_1_4-5-2023\female\all_cleaned_female_images"

for filename in os.listdir(folder_path):
    f = os.path.join(folder_path, filename)
    file_name = filename.split('.')[0]
    destination_file = os.path.join(destination_path, file_name+".png")
    print(destination_file)
    # img = cv2.imread(f)
    # updated_image = bgremove2(img)
    # # showimage(updated_image)
    # cv2.imwrite(destination_file, updated_image)
    remove_background_using_rembg(f, destination_file)
    