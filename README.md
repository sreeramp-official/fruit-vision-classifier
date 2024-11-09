
# Fruit Classifier Web App

This repository contains a Fruit Classifier model built using a Random Forest classifier, deployed as a web application using Streamlit. The app allows users to upload an image of a fruit, and it predicts the fruit type based on RGB color features.



## Features

- **Model:** The model is trained to classify fruits based on their RGB color values.
- **Streamlit Web App:** An easy-to-use interface for uploading fruit images and viewing predictions.


## Data Limitations

- The dataset currently only includes images of **Apple**, **Banana**, **Grape**, and **Orange**.
- The dataset is limited, which may affect the model's accuracy and generalization to other fruits.
## Performance Considerations

- The model uses RGB color values to classify fruits. Therefore, fruits with similar colors might lead to reduced accuracy as the model relies on color-based features.
- As the project uses basic color features, it may not be suitable for more complex classification tasks.

## Dataset

- The dataset was sourced from [Fruits Dataset on Kaggle](https://www.kaggle.com/datasets/shreyapmaher/fruits-dataset-images/data).

## Purpose

This project is part of a learning exercise to explore machine learning model development and deployment using Streamlit. The project aims to enhance understanding of image classification and web app integration.
## Setup and Installation

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Streamlit app using `streamlit run app/fruit-classification-app.py`.

    