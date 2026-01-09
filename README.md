[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zP0O23M7)

# Project Documentation

🌌 Star Reduction on Astronomical Images
Project developed as part of SAE S3.C2
BUT Informatique – Astronomical Image Processing

Goal:
Reduce the visibility of stars in astronomical images while preserving diffuse structures
such as nebulae (Horsehead) and galaxies (M31).

---

📌 General Principle
The project is based on a localized star reduction approach following these steps:

Create an eroded version of the original image.
Automatically generate a star mask.
Smooth the mask edges using a Gaussian blur.
Compute the final image by interpolation:

[
I{final} = (M \times I{erode}) + ((1 - M) \times I_{original})
]

This method allows:
Strong attenuation of stars
Preservation of nebulae and galaxy structures
Smooth transitions without visible halos

---
## Installation


### Virtual Environment

It is recommended to create a virtual environment before installing dependencies :

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```


### Dependencies
```bash
pip install -r requirements.txt
```

Or install dependencies manually:
```bash
pip install [package-name]
```

## Usage


### Command Line
```bash
python main.py [arguments]
```

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

## Examples files
Example files are located in the `examples/` directory. You can run the scripts with these files to see how they work.
- Example 1 : `examples/HorseHead.fits` (Black and whiteFITS image file for testing)
- Example 2 : `examples/test_M31_linear.fits` (Color FITS image file for testing)
- Example 3 : `examples/test_M31_raw.fits` (Color FITS image file for testing)


## Authors
 MOREL Théo
 DARTOIS Samuel
 CESAIRE Lilian
