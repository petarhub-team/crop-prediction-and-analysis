# PetarHub Team Crop Prediction and Analysis

This project was developed at **PETAR Hub** by **Md. Emran Biswas**, a research fellow at PETAR Hub. It focuses on providing a **crop recommendation system** based on environmental factors such as **NPK values**, **pH levels**, **Temperature**, and **Rainfall**. The goal is to assist farmers, agronomists, and agricultural researchers in optimizing crop selection for different environmental conditions.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Input Data](#input-data)
- [Model](#model)
- [Requirements](#requirements)
- [Disclaimer](#disclaimer)

## Overview

This project aims to assist farmers, agronomists, and agricultural researchers by providing a system that suggests optimal crop choices based on environmental factors. It takes into account four primary parameters that influence crop growth:
- **NPK values**: These represent the essential nutrients (Nitrogen, Phosphorus, and Potassium) required for plant growth.
- **pH levels**: Soil pH determines the availability of nutrients to plants.
- **Temperature**: Certain crops have specific temperature requirements.
- **Rainfall**: The amount of rainfall in a given area influences water availability for crops.

Based on these inputs, the system suggests the most suitable crops, helping optimize agricultural practices.

## Features

- **Input-based crop recommendation**: Users input environmental factors, and the system recommends suitable crops.
- **Data-driven insights**: The model uses historical data for better crop prediction.
- **Easy-to-use interface**: Built with Streamlit, making it accessible to users without technical knowledge.

## Installation

To run this project locally or on your own Streamlit server, follow these steps:

### 1. Clone the repository

```bash
git clone https://github.com/petarhub-team/crop-prediction-and-analysis.git
```

### 2. Navigate to the project directory

```bash
cd petarhub-team-crop-prediction-and-analysis
```

### 3. Install dependencies

Create a virtual environment and install the necessary libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Usage

### Run the application

To start the app locally with Streamlit, run the following command:

```bash
streamlit run crop.py
```

This will launch the app in your default web browser, where you can enter the environmental factors (NPK values, pH, temperature, and rainfall) and receive crop recommendations.

### Input fields

1. **NPK (Nitrogen, Phosphorus, Potassium)**: Enter the values for these essential nutrients.
2. **pH level**: Specify the pH of the soil.
3. **Temperature**: Enter the temperature in °C.
4. **Rainfall**: Provide the average rainfall data for the area (in mm).

The app will output a list of recommended crops based on these inputs.

## Input Data

The model uses a dataset named `crop.csv`, which contains historical data about crops and their respective environmental conditions. This dataset is used to train the model and make predictions.
You can customize the dataset as per your requirements.

## Model

The recommendation system is built using a machine learning model that maps the environmental conditions (NPK, pH, temperature, and rainfall) to the most suitable crops. The exact model architecture and training details are specified in the `crop.py` script.

## Requirements

This project requires the following Python libraries:

- `streamlit` - for building the interactive web app
- `pandas` - for handling the dataset
- `scikit-learn` - for machine learning models (if applicable)
- `numpy` - for numerical calculations

You can install these dependencies using:

```bash
pip install -r requirements.txt
```

## Disclaimer

Any alteration, modification, distribution, or use of this project requires explicit consent from **PETAR Hub** and the respected contributors. Unauthorized use or alteration of this project may be subject to legal consequences. Please contact **PETAR Hub** and **Md. Emran Biswas** for any inquiries related to licensing and usage.
