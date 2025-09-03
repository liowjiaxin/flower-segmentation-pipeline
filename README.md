
  
# Flower Segmentation Program

This program provides a Python-based graphical user interface (GUI) application for segmenting flowers from images, isolating them from the background using a combination of techniques including **GrabCut algorithm**, **morphological operations**, and **colour-based segmentation**.

## Features

-   **Image Segmentation**: Utilises the GrabCut algorithm to segment flowers from images.
-   **Evaluation**: Calculates the Intersection over Union (IoU) metric to evaluate the accuracy of segmentation results.
-   **Graphical User Interface (GUI)**: Provides an intuitive interface for users to select difficulty levels, view images, and evaluate segmentation results.
-   **Image Visualisation**: Displays segmented images along with ground truth and binary masks for visual comparison.


## Usage

 - **Install Dependencies**:
	 - Ensure you have Python installed on your system.
	 - Install the required Python libraries using pip:
	     ```
	     pip install opencv-python numpy matplotlib
	     ```

 - **Run the Program**:
 - Download the `flower_segmentation.py` file.
 - Create the following directory structure in the same directory as the script:
	   - **flower_segmentation.py**  (Main script)
     - **input_images/** 
	     - **easy/** 
		     - easy_1.jpg 
		     - easy_2.jpg 
		     - easy_3.jpg
		- **medium/**
			 - medium_1.jpg 
			 - medium_2.jpg
			 - medium_3.jpg
		-  **hard/**
			- hard_1.jpg
			- hard_2.jpg
			- hard_3.jpg 
			
     - **ground_truths/** 
	     - **easy/** 
		     - easy_1.png 
		     - easy_2.png 
		     - easy_3.png 
		- **medium/** 
			- medium_1.png 
			- medium_2.png 
			- medium_3.png 
		- **hard/**
			- hard_1.png
			- hard_2.png
			- hard_3.png 
   - Navigate to the directory containing `flower_segmentation.py` in your terminal.
   - Run the program using the following command:
	     ```
     python flower_segmentation.py
	     ```

3. **Select Difficulty Level**:
   - Upon running the program, you'll be prompted to select a difficulty level (easy, medium, or hard).
   - Choose the desired difficulty level to process images accordingly.

4. **View Results**:
   - The program will display the **processed images** along with the calculated **Intersection over Union (IoU)** scores.
   - Additionally, you can view all **original** images from the dataset for visual inspection.
   - You can also view all **segmented** images from the dataset for visual inspection.

## Dependencies

-   OpenCV (`opencv-python`)
-   NumPy (`numpy`)
-   Matplotlib (`matplotlib`)
-   Tkinter (`tkinter`) (for GUI)

## File Structure

-   **flower_segmentation.py**: Main script containing the GUI application logic.
-   **README.md**: Documentation file explaining the project and its usage.
-   **input_images/**: Directory containing images for segmentation.
-   **output-image/ (Automatically created)**: Directory containing segmented images and evaluation results.
-   **ground_truths/**: Directory containing ground truth masks for evaluation. 
-   **imageprocessing-pipeline/ (Automatically created)**: Directory containing images generated as part of the image processing pipeline.
