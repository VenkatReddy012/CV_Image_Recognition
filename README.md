# 🧠 Computer Vision - Image Classification with PyTorch

This project demonstrates a simple computer vision pipeline using PyTorch and a pre-trained convolutional neural network to classify images. It includes loading an image, preprocessing it, running it through a model, and interpreting the results.

## 📂 Project Structure

```
├── Computer_Vision.ipynb     # Main Jupyter Notebook
├── imagenet_classes.txt      # Class labels for ImageNet (download separately)
├── images/                   # Directory to store test images
```

## 🔍 Features

- Load and preprocess input images.
- Use a pretrained ResNet model for inference.
- Display predicted class label and confidence.
- Simple and modular PyTorch-based code.
- Works with any image supported by PIL.

## 🛠️ Requirements

Install the required dependencies using pip:

```bash
pip install torch torchvision matplotlib pillow
```

## 📥 Usage

1. **Clone this repository:**

```bash
git clone https://github.com/yourusername/computer-vision-pytorch.git
cd computer-vision-pytorch
```

2. **Download the `imagenet_classes.txt` file:**

You can get it from [here](https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt) and place it in the project directory.

3. **Run the notebook:**

Open the notebook with Jupyter:

```bash
jupyter notebook Computer_Vision.ipynb
```

Follow the cells to test the image classification pipeline.

4. **Add Your Own Images:**

Put your test images in the `images/` directory and change the path in the notebook code accordingly.

## 🧠 Model Details

- **Architecture:** ResNet (e.g., ResNet-18, ResNet-50)
- **Dataset:** ImageNet (pretrained)
- **Framework:** PyTorch

## 🖼️ Example Output

```
Predicted: "golden retriever" with 87.32% confidence
```

## ⚠️ Troubleshooting

- **File Not Found Error:** Make sure `imagenet_classes.txt` exists in the same directory.
- **CUDA errors:** This code runs on CPU by default. Modify the device to `cuda` if using a GPU.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🙌 Acknowledgments

- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/stable/)
- ImageNet Dataset