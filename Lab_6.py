import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def house():
    img = cv2.imread("house.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. Resize
    img = cv2.resize(img, (224, 224))

    # 2. Flip ngang
    flip = cv2.flip(img, 1)

    # 3. Rotate
    M = cv2.getRotationMatrix2D((112,112), 15, 1)
    rotate = cv2.warpAffine(img, M, (224,224))

    # 4. Brightness +20%
    bright = np.clip(img * 1.2, 0, 255).astype(np.uint8)

    # 5. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 6. Normalize
    norm = gray / 255.0

    # Hiển thị
    fig, ax = plt.subplots(2,5, figsize=(15,6))

    images = [img, flip, rotate, bright, gray]
    titles = ["Original","Flip","Rotate","Bright","Gray"]

    for i in range(5):
        ax[0,i].imshow(images[i], cmap="gray" if i==4 else None)
        ax[0,i].set_title(titles[i])
        ax[0,i].axis("off")

    for i in range(5):
        ax[1,i].imshow(norm, cmap="gray")
        ax[1,i].set_title("Normalized")
        ax[1,i].axis("off")

    plt.show()
# house()

def car():
    img = cv2.imread("car.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))

    # Gaussian Noise
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    noisy = cv2.add(img, noise)

    # Brightness -15%
    dark = np.clip(img * 0.85, 0, 255).astype(np.uint8)

    # Rotate 10°
    M = cv2.getRotationMatrix2D((112,112), 10, 1)
    rotate = cv2.warpAffine(img, M, (224,224))

    # Normalize
    norm = img / 255.0

    # Show
    fig, ax = plt.subplots(1,4, figsize=(14,4))
    imgs = [img, noisy, dark, rotate]
    titles = ["Original","Noise","Dark","Rotate"]

    for i in range(4):
        ax[i].imshow(imgs[i])
        ax[i].set_title(titles[i])
        ax[i].axis("off")

    plt.show()
# car()

def plant():
    img = cv2.imread("plant.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))

    augmented = []

    for _ in range(9):
        temp = img.copy()

        # Flip random
        if random.random() > 0.5:
            temp = cv2.flip(temp, 1)

        # Rotate random
        angle = random.randint(-20,20)
        M = cv2.getRotationMatrix2D((112,112), angle, 1)
        temp = cv2.warpAffine(temp, M, (224,224))

        # Random Crop
        x = random.randint(0,20)
        y = random.randint(0,20)
        crop = temp[y:200+y, x:200+x]
        temp = cv2.resize(crop, (224,224))

        augmented.append(temp/255.0)

    # Show grid
    fig, ax = plt.subplots(3,3, figsize=(10,10))
    k=0
    for i in range(3):
        for j in range(3):
            ax[i,j].imshow(augmented[k])
            ax[i,j].axis("off")
            k+=1
    plt.show()
# plant()

def room():
    img = cv2.imread("room.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))

    # Rotate
    M = cv2.getRotationMatrix2D((112,112), -15, 1)
    rotate = cv2.warpAffine(img, M, (224,224))

    # Flip
    flip = cv2.flip(img, 1)

    # Brightness
    bright = np.clip(img*1.2,0,255).astype(np.uint8)

    # Gray
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Normalize
    norm = gray/255.0

    # Show
    fig, ax = plt.subplots(1,4, figsize=(14,4))
    imgs = [img, rotate, flip, bright]
    titles = ["Original","Rotate","Flip","Bright"]

    for i in range(4):
        ax[i].imshow(imgs[i], cmap="gray" if i==0 else None)
        ax[i].set_title(titles[i])
        ax[i].axis("off")

    plt.show()
room()