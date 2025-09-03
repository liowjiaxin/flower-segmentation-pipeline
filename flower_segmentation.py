import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Function to create sub-folder for each input image
def create_subfolder(image_path, difficulty):
    folder_name = os.path.basename(image_path).split('.')[0]
    subfolder_path = os.path.join("imageprocessing-pipeline", difficulty, folder_name)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    return subfolder_path

# Function for flower segmentation
def flower_segmentation(image, green_threshold=70, image_path='', difficulty=''):
    # Eliminate background using graph-cut algorithm (GrabCut)
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Define the rectangle enclosing the flower region (slightly smaller than image dimensions)
    rect = (10, 10, image.shape[1] - 20, image.shape[0] - 20)

    # Apply GrabCut algorithm to get the foreground mask
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Create mask where 1 and 3 indicate definite foreground and probable foreground
    mask = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')

    # Apply the mask to the original image
    segmented_image = cv2.bitwise_and(image, image, mask=mask)

    # Save intermediate result
    subfolder_path = create_subfolder(image_path, difficulty)
    cv2.imwrite(os.path.join(subfolder_path, "After_GrabCut.jpg"), segmented_image)

    # Morphological Operations
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    flower_mask = cv2.dilate(closing, kernel_dilate, iterations=2)

    result = cv2.bitwise_and(image, image, mask=flower_mask)

    # Save intermediate result
    cv2.imwrite(os.path.join(subfolder_path, "After_Morphological_Operations.jpg"), flower_mask)

    # Remove green pixels using a threshold
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, green_threshold, 100])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    result[mask_green != 0] = [0, 0, 0]  # Set the masked green pixels to black

    # Save final result
    cv2.imwrite(os.path.join(subfolder_path, "After_Colour-Based_Segmentation.jpg"), result)

    return result

def calculate_iou(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask)
    union = np.logical_or(gt_mask, pred_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score, intersection.astype(np.uint8) * 255

def process_image(image_path, difficulty):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to read the image.")
        return False

    segmented_image = flower_segmentation(image, image_path=image_path, difficulty=difficulty)

    # Convert segmented image to grayscale
    segmented_gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

    # Load ground truth mask
    gt_folder = "ground_truths"  # Modify this path according to your directory structure
    image_folder = os.path.dirname(image_path)
    gt_image_path = os.path.join(gt_folder, os.path.basename(image_folder),
                                 os.path.basename(image_path).replace(".jpg", ".png"))
    gt_mask = cv2.imread(gt_image_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is not None:
        # Apply Otsu's thresholding to create a binary mask
        _, binary_gt_mask = cv2.threshold(gt_mask, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Convert segmented grayscale image to binary mask
        _, binary_segmented = cv2.threshold(segmented_gray, 0, 255, cv2.THRESH_BINARY)

        # Calculate IoU
        iou, binary_iou = calculate_iou(binary_gt_mask, binary_segmented)
        print(f"IoU: {iou:.2f}")

        # Plot binary masks for visualization
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(binary_segmented, cmap='gray')
        axes[0].set_title('Segmented Binary')
        axes[0].axis('off')

        axes[1].imshow(binary_gt_mask, cmap='gray')
        axes[1].set_title('Ground Truth Binary')
        axes[1].axis('off')

        axes[2].imshow(binary_iou, cmap='gray')
        axes[2].set_title('Intersection over Union (IoU)')
        axes[2].axis('off')

        plt.tight_layout()

        # Save the result image
        output_folder = os.path.join("output-image", difficulty)
        os.makedirs(output_folder, exist_ok=True)
        output_image_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_image_path, binary_segmented)

        return iou, fig

def select_difficulty(difficulty):

    def select_difficulty_menu(difficulty):
        select_difficulty(difficulty)
        root.quit()  # Quit the main event loop to close the window

    image_folder = os.path.join("input_images", difficulty)
    image_files = sorted(os.listdir(image_folder))

    def select_image(image_file):
        file_path = os.path.join(image_folder, image_file)
        result = process_image(file_path, difficulty)
        if result:
            iou, fig = result
            show_results(image_file, iou, fig)
        else:
            messagebox.showerror("Error", "Failed to process the image.")

    def show_results(image_file, iou, fig):
        subroot = tk.Toplevel()
        subroot.title(f"Results for {image_file}")
        subroot.geometry("800x400")

        iou_label = tk.Label(subroot, text=f"IoU: {iou:.2f}")
        iou_label.pack(pady=10)

        canvas = FigureCanvasTkAgg(fig, master=subroot)
        canvas.draw()
        canvas.get_tk_widget().pack()

        exit_button = tk.Button(subroot, text="Close", command=subroot.destroy)
        exit_button.pack(pady=10)

        # Center the window
        subroot.update_idletasks()
        width = subroot.winfo_width()
        height = subroot.winfo_height()
        x_offset = (subroot.winfo_screenwidth() - width) // 2
        y_offset = (subroot.winfo_screenheight() - height) // 2
        subroot.geometry(f"{width}x{height}+{x_offset}+{y_offset}")

    root = tk.Tk()
    root.title("Flower Segmentation and Evaluation")
    root.geometry("400x300")

    label = tk.Label(root, text="Select an image to process:")
    label.pack(pady=10)

    for image_file in image_files:
        image_button = tk.Button(root, text=image_file, command=lambda file=image_file: select_image(file))
        image_button.pack()

    exit_button = tk.Button(root, text="Back to Difficulty Menu", command=root.destroy)
    exit_button.pack(pady=10)

    # Center the window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x_offset = (root.winfo_screenwidth() - width) // 2
    y_offset = (root.winfo_screenheight() - height) // 2
    root.geometry(f"{width}x{height}+{x_offset}+{y_offset}")

    root.mainloop()


def view_all_images():
    # Close all existing result windows
    plt.close('all')

    images = []
    labels = []

    input_images_folder = "input_images"  # Folder containing input images

    # Load and append images from all difficulty categories
    for difficulty in ['easy', 'medium', 'hard']:
        image_folder = os.path.join(input_images_folder, difficulty)
        image_files = sorted(os.listdir(image_folder))
        for image_file in image_files:
            file_path = os.path.join(image_folder, image_file)
            image = cv2.imread(file_path)
            images.append(image)
            labels.append(f"{difficulty.capitalize()} - {image_file}")

    # Display the images
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            ax.set_title(labels[i])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def view_all_segmented_images():
    # Close all existing result windows
    plt.close('all')

    images = []
    labels = []
    has_images = False

    # Load and append segmented images from all difficulty categories
    for difficulty in ['easy', 'medium', 'hard']:
        image_folder = os.path.join("output-image", difficulty)
        try:
            image_files = sorted(os.listdir(image_folder))
        except FileNotFoundError:
            continue  # Skip to the next difficulty level if the directory doesn't exist
        if image_files:  # Check if the directory contains images
            has_images = True
            for image_file in image_files:
                file_path = os.path.join(image_folder, image_file)
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
                images.append(image)
                labels.append(f"{difficulty.capitalize()} - {os.path.splitext(image_file)[0]}")

    if has_images:
        # Display the images
        num_images = len(images)
        if num_images >= 9:
            fig, axes = plt.subplots(3, 3, figsize=(10, 10))
            for i, ax in enumerate(axes.flat):
                if i < len(images):
                    ax.imshow(images[i], cmap='gray')
                    ax.set_title(labels[i])
                ax.axis('off')
            plt.tight_layout()
            plt.show()
        else:
            messagebox.showinfo("Info", "There are fewer than 9 segmented images available. Please perform segmentation for all images first.")
    else:
        messagebox.showinfo("Info", "No segmented images found. Please perform segmentation first.")


def main():
    root = tk.Tk()
    root.title("Flower Segmentation and Evaluation")
    root.geometry("400x300")

    label = tk.Label(root, text="Select a difficulty level:")
    label.pack(pady=10)

    def select_difficulty_menu(difficulty):
        select_difficulty(difficulty)
        root.destroy()  # Close the main window after selecting difficulty

    for difficulty in ['easy', 'medium', 'hard']:
        difficulty_button = tk.Button(root, text=difficulty.capitalize(), command=lambda d=difficulty: select_difficulty_menu(d))
        difficulty_button.pack(pady=5)

    view_all_button = tk.Button(root, text="View All Images from Dataset", command=view_all_images)
    view_all_button.pack(pady=10)

    view_all_button = tk.Button(root, text="View All Segmented Images from Dataset", command=view_all_segmented_images)
    view_all_button.pack(pady=10)

    exit_button = tk.Button(root, text="Exit", command=sys.exit)
    exit_button.pack(pady=10)

    # Center the window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x_offset = (root.winfo_screenwidth() - width) // 2
    y_offset = (root.winfo_screenheight() - height) // 2
    root.geometry(f"{width}x{height}+{x_offset}+{y_offset}")

    root.mainloop()

if __name__ == "__main__":
    main()