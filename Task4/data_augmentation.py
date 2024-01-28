from PIL import Image, ImageEnhance, ImageOps
import random
import numpy as np
import os


def random_horizontal_flip(image):
    if random.random() < 0.5:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    return image


def random_rotation(image, max_angle=0.2):
    angle = random.uniform(-max_angle, max_angle)
    return image.rotate(angle)


def random_crop(image):
    width, height = image.size
    crop_size = 224  # Adjust as needed
    left = random.randint(0, width - crop_size)
    top = random.randint(0, height - crop_size)
    right = left + crop_size
    bottom = top + crop_size
    return image.crop((left, top, right, bottom))


def random_brightness(image, factor_range=(0.5, 1.5)):
    enhancer = ImageEnhance.Brightness(image)
    factor = random.uniform(*factor_range)
    return enhancer.enhance(factor)


def random_contrast(image, factor_range=(0.5, 1.5)):
    enhancer = ImageEnhance.Contrast(image)
    factor = random.uniform(*factor_range)
    return enhancer.enhance(factor)


def augment_image(image_path, output_folder, num_augmentations=30):
    original_image = Image.open(image_path)
    image_name = os.path.basename(image_path)[: -4]

    for i in range(num_augmentations):
        augmented_image = original_image.copy()

        augmented_image = random_horizontal_flip(augmented_image)
        augmented_image = random_rotation(augmented_image)
        augmented_image = random_crop(augmented_image)
        augmented_image = random_brightness(augmented_image)
        augmented_image = random_contrast(augmented_image)

        # Resize all images to 256x256
        augmented_image = augmented_image.resize((256, 256))

        # Save the augmented image
        output_path = f"{output_folder}/{image_name}_augmented_{i + 1}.png"
        augmented_image.save(output_path)


os.mkdir("./MIT_small_train_1_augmented")
os.mkdir("./MIT_small_train_1_augmented/train")
for folder in os.listdir("./MIT_small_train_1/train"):
    os.mkdir(f"./MIT_small_train_1_augmented/train/{folder}")
    for file in os.listdir(f"./MIT_small_train_1/train/{folder}"):
        # Copy file to new folder
        os.system(f"cp ./MIT_small_train_1/train/{folder}/{file} ./MIT_small_train_1_augmented/train/{folder}")
        # And add 4 augmented images
        augment_image(f"./MIT_small_train_1/train/{folder}/{file}", f"./MIT_small_train_1_augmented/train/{folder}", 4)