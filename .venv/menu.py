import cv2
import numpy as np
import os

# Flower segmentation
def flower_segmentation(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu's thresholding
    _, binary_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert binary image to binary mask
    mask = np.zeros_like(binary_image)
    mask[binary_image > 0] = 255

    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to remove small noise
    min_area = 1000
    flower_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            flower_contours.append(contour)

    # Create a blank image to draw contours
    flower_segmented = np.zeros(image.shape[:2], dtype=np.uint8)

    # Draw contours of flowers on the blank image
    cv2.drawContours(flower_segmented, flower_contours, -1, 255, thickness=cv2.FILLED)

    # Apply dilation to smooth edges of the segmented mask
    kernel_dilate = np.ones((5, 5), np.uint8)
    segmented_smoothed = cv2.dilate(flower_segmented, kernel_dilate, iterations=2)

    # Apply the mask to the original image to segment the flower
    result = cv2.bitwise_and(image, image, mask=segmented_smoothed)

    return result


def print_menu():
    print("=== Menu ===")
    print("1. Easy")
    print("2. Medium")
    print("3. Hard")
    print("4. Quit")


def select_image(folder):
    files = os.listdir(folder)
    print("Select an image:")
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")

    while True:
        choice = input(f"Enter your choice (1-{len(files)}) or '0' to go back: ")
        if choice == '0':
            return None
        try:
            index = int(choice) - 1
            if 0 <= index < len(files):
                return os.path.join(folder, files[index])
            else:
                print("Invalid choice. Please enter a valid option.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def main():
    while True:
        print_menu()
        choice = input("Enter your choice (1/2/3/4): ")

        if choice == "1":
            print("You selected Easy!")
            easy_image = select_image("Images/easy")
            if easy_image is not None:
                image = cv2.imread(easy_image)
                result = flower_segmentation(image)
                cv2.imshow('Original Image', image)
                cv2.imshow('Segmented Flowers', result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        elif choice == "2":
            print("You selected Medium!")
            medium_image = select_image("Images/medium")
            if medium_image is not None:
                image = cv2.imread(medium_image)
                result = flower_segmentation(image)
                cv2.imshow('Original Image', image)
                cv2.imshow('Segmented Flowers', result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        elif choice == "3":
            print("You selected Hard!")
            hard_image = select_image("Images/hard")
            if hard_image is not None:
                image = cv2.imread(hard_image)
                result = flower_segmentation(image)
                cv2.imshow('Original Image', image)
                cv2.imshow('Segmented Flowers', result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        elif choice == "4":
            print("Quitting the program...")
            break
        else:
            print("Invalid choice. Please enter a valid option (1/2/3/4).")

#test
if __name__ == "__main__":
    main()

