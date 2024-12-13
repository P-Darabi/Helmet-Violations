# Helmet Violation Detection with License Plate Recognition

## Overview
This project detects and tracks helmet violations in video footage and extracts license plate numbers of violators using YOLO and PaddleOCR. It provides annotated output videos and a text file containing detected license plate numbers for further analysis.

## Features
- **Helmet Violation Detection**: Identifies motorcyclists not wearing helmets using YOLO.
- **License Plate Recognition**: Extracts license plate information using PaddleOCR.
- **Annotated Video Output**: Generates a video with bounding boxes around violators and their license plate numbers displayed.
- **Structured Output**: Saves license plate details in a text file.

## Dataset
The dataset used in this project is hosted on Kaggle: [Helmet Dataset](https://www.kaggle.com/datasets/pkdarabi/helmet/data). It contains annotated images for training and validating the detection model.

## File Structure
```plaintext
project/
├── img/                      # Contains sample images
├── output/                   # Directory for output files (e.g., text files)
├── videos/                   # Input video files
├── yolo-weights/             # Directory for YOLO weight files
├── result.mp4                # Annotated video output
├── main.py                   # Main script for video processing
├── README.md                 # Project documentation
├── requirements.txt          # List of required Python libraries
└── Training.ipynb            # Training notebook
```

## Prerequisites
### Software Requirements
- Python 3.8 or later
- Required Python libraries (install using `requirements.txt`):
  - `torch`, `opencv-python`, `numpy`, `Pillow`, `matplotlib`, `paddleocr`
- YOLO weights for helmet detection.

### Hardware Requirements
- A GPU-enabled machine for faster processing is recommended.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/helmet-violation-tracker.git
   cd helmet-violation-tracker
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the YOLO weight file (`best.pt`) in the `yolo-weights/` directory.

4. Install PaddleOCR:
   ```bash
   pip install paddleocr
   ```

## Usage
1. Place the input video file in the `videos/` directory.
2. Run the detection script:
   ```bash
   python main.py
   ```
3. Output:
   - The processed video will be saved as `result.mp4`.
   - License plate numbers will be stored in `output/plate_numbers.txt`.

## Example Output
- **Annotated Video**: Bounding boxes around violators and their license plate numbers.
- **Text File**:
  ```plaintext
  Detected Plate Numbers:
  ABC1234
  XYZ5678
  ```

## Future Enhancements
- Add real-time detection from live video feeds or CCTV.
- Enhance OCR accuracy for challenging conditions.
- Create a web-based dashboard for analytics.

## Contributions
Contributions are welcome! Feel free to fork this repository and submit pull requests.

## License
This project is licensed under the MIT License.

### Highlights of the Update:
1. **Accurate File Structure**: Matches the project layout visible in the image.
2. **Kaggle Dataset Link**: Directly references your dataset.
3. **Usage Details**: Includes information on how to run the project and what to expect as output.
4. **Future Enhancements**: Lists possible improvements.

![Demo](https://raw.githubusercontent.com/P-Darabi/Helmet-Violations/main/output/result.gif)

## 🔗 Additional Resources

- 🎥 **Project Demo**: Watch the live demonstration of this project on [YouTube](https://www.youtube.com/watch?v=vTjrQVMCbJ4).
- 🌐 **Kaggle Notebook**: Interested in a Kaggle environment? Explore the notebook [here](https://www.kaggle.com/code/pkdarabi/Helmet-Violations).
- 🌐 **Dataset Source**: Available on [Kaggle](https://www.kaggle.com/datasets/pkdarabi/helmet).
- 🤝 **Connect on LinkedIn**: Have questions or looking for collaboration? Let's connect on [LinkedIn](www.linkedin.com/in/p-karimi-darabi).
**Let me know if you need further refinements!**
