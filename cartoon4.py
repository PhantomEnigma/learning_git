import cv2
import numpy as np

def pixelate(image, pixel_size):
    """Pixelates an image."""
    height, width = image.shape[:2]
    for y in range(0, height, pixel_size):
        for x in range(0, width, pixel_size):
            avg_color = np.mean(image[y:y + pixel_size, x:x + pixel_size], axis=(0, 1))
            image[y:y + pixel_size, x:x + pixel_size] = avg_color.astype(np.uint8)
    return image

def oil_paint(image):
    """Applies an oil painting effect with more detail."""
    return cv2.xphoto.oilPainting(image, 3, 1)  # Even smaller brush size for more detail

def sketch(image):
    """Applies a sketch effect."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    return cv2.divide(gray, 255 - blurred, scale=256.0)

def cartoon_effect(img):
    """Applies a cartoon effect."""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply median blur
    gray = cv2.medianBlur(gray, 5)
    
    # Detect edges
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    
    # Apply bilateral filter
    color = cv2.bilateralFilter(img, 9, 300, 300)
    
    # Combine edges with color image
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    
    return cartoon

def watercolor_effect(img):
    """Applies a watercolor effect."""
    # Apply bilateral filter multiple times
    watercolor = img.copy()
    for _ in range(2):
        watercolor = cv2.bilateralFilter(watercolor, 9, 75, 75)
    
    # Use mean shift filtering
    watercolor = cv2.pyrMeanShiftFiltering(watercolor, 21, 51)
    
    return watercolor

def main():
    """Main function for the image conversion program."""
    while True:
        try:
            # Get image file name from user
            image_filename = input("Enter the image file name (e.g., image.jpg): ")

            # Check for 'exit' command
            if image_filename.lower() == "exit":
                break

            image = cv2.imread(image_filename)

            if image is None:
                print(f"Error: Could not load image '{image_filename}'. Please check the file path/integrity.")
                continue

            # Display original image
            cv2.imshow("Original Image", image)
            cv2.waitKey(0) 

            # Display menu
            print("\nChoose an art style:")
            print("1. Pixel Art")
            print("2. Oil Paint")
            print("3. Sketch")
            print("4. Cartoon Effect")
            print("5. Watercolor Effect")
            print("6. Exit")

            choice = int(input("Enter your choice: "))

            if choice == 1:
                pixel_size = int(input("Enter pixel size: "))
                converted_image = pixelate(image.copy(), pixel_size)
            elif choice == 2:
                converted_image = oil_paint(image.copy())
            elif choice == 3:
                converted_image = sketch(image.copy())
            elif choice == 4:
                converted_image = cartoon_effect(image.copy())
            elif choice == 5:
                converted_image = watercolor_effect(image.copy())
            elif choice == 6:
                break
            else:
                print("Invalid choice. Please try again.")
                continue

            # Display converted image
            cv2.imshow("Converted Image", converted_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except FileNotFoundError:
            print(f"Error: File '{image_filename}' not found. Please check the file name.")
        except ValueError:
            print("Invalid input. Please enter a valid number for the choice and pixel size.")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()