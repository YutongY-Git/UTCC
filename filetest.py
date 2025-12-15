import os
def print_images(directory):
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(root, filename)
                print(image_path)
    print(directory)

if __name__ == '__main__':
    directory = r'E:\UTB-finish\UTB_master\Train_dataset\image_label\images'  # Replace with the actual directory path
    print_images(directory)



