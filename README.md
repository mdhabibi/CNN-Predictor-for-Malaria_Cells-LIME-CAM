# **Malaria Cell Image Classification Using CNN**

## Overview
Malaria is a life-threatening disease caused by parasites that are transmitted to people through the bites of infected female Anopheles mosquitoes. It's preventable and curable but can be fatal if not treated promptly. The Plasmodium falciparum parasite is the most dangerous, with the highest rates of complications and mortality.

According to global health organizations, hundreds of thousands succumb to malaria annually, with a significant impact on child mortality rates. The annual death toll is certainly in the hundreds of thousands, but estimates differ between different global health organizations: The World Health Organization (WHO) estimates that 558,000 people died because of malaria in 2019; the Institute of Health Metrics and Evaluation (IHME) puts this estimate at 643,000.

Most victims are children. It is one of the leading causes of child mortality. Every twelfth child that died in 2017, died because of malaria.<sup>[1]</sup> This project aims to assist in the fight against malaria by automating the detection of infected cells using deep learning.

![Malaria Microscopy Image](https://ourworldindata.org/uploads/2022/03/Previous-prevalence-of-malaria-world-map.png)

---

<sup>[1]</sup> [Source Link](https://ourworldindata.org/malaria#)



## Project Objective
The goal of this project is to develop a Convolutional Neural Network (CNN) that can classify microscopic images of cells as either infected with malaria or uninfected.

## Methodology
We've employed a robust workflow to train our model:
- **Data Preprocessing**: Standardizing the cell images for model input.
- **Data Augmentation**: Enhancing the dataset to prevent overfitting and improve model robustness.
- **Model Architecture**: Designing a CNN that learns features from cell images for classification.
- **Training and Validation**: Using an iterative approach, with early stopping to prevent overfitting.
- **Evaluation**: Assessing model performance with accuracy, precision, recall, F1-score, and AUC.

## Results
The model achieved an impressive **96.03% accuracy** on the test set. The precision, recall, and F1-scores were high for both classes, suggesting balanced classification ability. Notably, the model attained an **AUC score of 0.99**, indicating excellent discriminative power.

Here are some visualizations and results from our Malaria Cell Image Classification project:

## Confusion Matrix

![Confusion Matrix](Images/Confusion_Matrix.png)

The confusion matrix shows the model's performance in classifying infected and uninfected cells.

## Precision-Recall Curve

![Precision-Recall Curve](https://example.com/precision_recall_curve.png)

This graph illustrates the training accuracy and loss curves, showing the model's learning progress over epochs during training.

## ROC Curve

![ROC Curve](https://example.com/roc_curve.png)

The ROC curve displays the model's true positive rate against the false positive rate at various threshold levels.

Feel free to explore these visualizations to gain insights into our model's performance.

## Dataset
The dataset is from [Kaggle's Malaria Cell Images Dataset](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria), which is hosted on Kaggle. It contains a large number of labeled microscopic images of cells infected with malaria as well as uninfected cells.

To download the dataset, you will need a Kaggle account and the Kaggle API installed on your machine. Follow these steps:

1. Go to your Kaggle account settings and scroll down to the API section to create a new API token. This will download a `kaggle.json` file containing your API credentials.
3. Place the `kaggle.json` file inside the main notebook when the snippet below is executed:

from google.colab import files

uploaded = files.upload()  # Upload the kaggle.json file here

## How to Use
Refer to the Jupyter notebook in this repository for detailed methodology and code implementation.

## Conclusion and Future Work
The model's strong performance underscores its potential as a diagnostic aid for rapid malaria detection. The project showcases the efficacy of CNNs in medical image analysis and their potential to support healthcare initiatives.

Future enhancements may include:
- Exploring advanced neural network architectures.
- Further hyperparameter optimization.
- Investigating transfer learning approaches for performance improvement.


## Acknowledgments
- Data provided by [Kaggle's Malaria Cell Images Dataset](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria).
- World Health Organization (WHO) for malaria statistics.

## References
1. [Our World in Data - Malaria](https://ourworldindata.org/malaria)

