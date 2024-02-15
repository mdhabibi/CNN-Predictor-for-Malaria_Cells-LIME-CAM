# **Malaria Cell Image Classification Using CNN and LIME Test**

<div align="center">
  <img src="Images/meresome-release-MIT.gif" width="500">
</div>
*This animation illustrates the emergence of the parasites from the infected cell. For more information on the growth of human malaria parasites in their dormant form, visit [MIT News](https://news.mit.edu/2018/human-malaria-parasites-grown-first-time-dormant-form-0222).* 



## Overview
**Malaria** is a life-threatening disease caused by parasites that are transmitted to people through the bites of infected female Anopheles mosquitoes. It's preventable and curable, but it can be fatal if not treated promptly. The Plasmodium falciparum parasite is the most dangerous, with the highest rates of complications and mortality.

According to global health organizations, hundreds of thousands succumb to malaria annually, with a significant impact on child mortality rates. The annual death toll is certainly in the hundreds of thousands, but estimates differ between different global health organizations. The World Health Organization (WHO) estimates that **558,000** people died because of malaria in **2019**; the Institute of Health Metrics and Evaluation (IHME) puts this estimate at **643,000**.

Most victims are children. It is one of the leading causes of child mortality. Every twelfth child that died in **2017**, died because of malaria.<sup>[1]</sup> This project aims to assist in the fight against malaria by automating the detection of infected cells using deep learning.

![Malaria Microscopy Image](https://ourworldindata.org/uploads/2022/03/Previous-prevalence-of-malaria-world-map.png)

---

<sup>[1]</sup> [Source Link](https://ourworldindata.org/malaria#)



## Project Objective
The goal of this project is to develop a **Convolutional Neural Network (CNN)** that can classify microscopic images of cells as either infected with malaria or uninfected.

## Methodology
We've employed a robust workflow to train our model:
- **Data Preprocessing**: Standardizing the cell images for model input.
- **Data Augmentation**: Enhancing the training data to prevent overfitting and improve model robustness.
- **Model Architecture**: Designing a CNN that learns features from cell images for classification.
- **Training and Validation**: Using an iterative approach with early stopping to prevent overfitting.
- **Evaluation**: Assessing model performance with accuracy, precision, recall, F1-score, and AUC.

## Dataset Download Instructions

The dataset used in this notebook is the "Malaria Cell Images Dataset", which is available on Kaggle. Due to its size (approximately 708 MB), it is not included in this repository. 

To use this notebook, please download the dataset by following these steps:

1. Visit the [Malaria Cell Images Dataset](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria) on Kaggle.
2. Download the dataset to your local machine.
3. Extract the dataset to a known directory.
4. Update the `dataset_path` variable in the notebook with the path to the extracted dataset on your machine.

In addition, we have provided a script named "download_dataset_from_Kaggle.py" in Script directory for user who want to download the dataset directly from Kaggle.


## Exploratory Data Analysis (EDA)

In this section, we explored the dataset to gain insights into the characteristics of the cell images used for malaria detection.

### Sample Images

To visualize the dataset, we displayed a sample of images from both classes: parasitized and uninfected cells. These sample images provide a visual representation of the cells and their differences.

<div align="center">
  <img src="Images/Sample_Images.png" width="800">
</div>


The sample images showcase the diversity of cell appearances and highlight the challenges of classifying them accurately.

### Class Distribution

We examined the distribution of classes in the dataset to understand the balance between infected and uninfected cells. This analysis revealed that the dataset contains a well-balanced distribution of both classes, which is essential for model training and evaluation.

<div align="center">
  <img src="Images/Class_Distribution.png" width="500">
</div>

While there were other aspects explored in the **EDA**, these findings provide a brief overview of the dataset's characteristics and challenges, which guided our approach in building the **CNN model** for malaria cell classification.


## Results
Here are some visualizations and results from our Malaria Cell Image Classification project:

<div align="center">
  <img src="Images/Model_predictions.png" width="800">
</div>

### Model Performance Metrics

To evaluate the effectiveness of our malaria cell image classification model, we've compiled key performance metrics. These metrics provide insights into the model's ability to accurately classify cells as either infected or uninfected. Below is a summary table showcasing these metrics:


<div align="center">

| Metric         | Infected (0) | Uninfected (1) | Overall     |
|----------------|--------------|----------------|-------------|
| Precision      | 0.97         | 0.95           | -           |
| Recall         | 0.95         | 0.97           | -           |
| F1-Score       | 0.96         | 0.96           | -           |
| Accuracy       | -            | -              | 96.19%      |
| Test Loss      | -            | -              | 0.16538     |
| Samples Tested | 2797         | 2715           | 5512        |

</div>



**Key Aspects of the Table:**

- **Accuracy (96.19%)**: This is the overall correctness of the model in classifying the images. A high accuracy indicates that the model is able to correctly identify most of the infected and uninfected cells.

- **Test Loss (0.16538)**: Represents how well the model is performing against the test dataset. A lower test loss indicates that the model is making fewer mistakes in its predictions.

- **Precision, Recall, and F1-Score for Each Class**: These metrics are calculated separately for both 'Infected' and 'Uninfected' classes. 
  - _Precision_ reflects the proportion of true positive identifications among all positive identifications made by the model.
  - _Recall_ (or sensitivity) indicates the proportion of actual positives correctly identified.
  - _F1-Score_ is the harmonic mean of precision and recall, providing a balance between these two metrics.

- **Macro and Weighted Averages**: These provide an overall picture of the model's performance across both classes. The macro average calculates metrics independently for each class and then takes the average, treating all classes equally. The weighted average takes class imbalance into account.

The above metrics are essential for understanding the model's strengths and areas for improvement, especially in a medical imaging context where accuracy and reliability are crucial.


### Confusion Matrix

<div align="center">
  <img src="Images/cnn_confusion_matrix.png" width="500">
</div>

- **True Positives (TP) - 2671:** The model correctly identified **2671** cells as **'Infected'**.
- **True Negatives (TN) - 2631:** The model correctly identified **2631** cells as **'Uninfected'**.
- **False Positives (FP) - 126:** The model incorrectly identified **126** cells as **'Infected'** when they were actually **'Uninfected'**.
- **False Negatives (FN) - 84:** The model incorrectly identified **84** cells as **'Uninfected'** when they were actually **'Infected'**.


### Receiver Operating Characteristic (ROC) Curve and Area Under the Curve (AUC) Score


<div align="center">
  <img src="Images/cnn_roc_curve.png" width="500">
</div>

The **ROC** curve graphically represents the trade-off between the true positive rate and the false positive rate at various thresholds. An **AUC** score of **0.99** signifies that the model has an outstanding discriminative ability to differentiate between the classes. This high **AUC** score suggests that the model can reliably rank predictions with a high degree of separability between '**Parasitized**' and '**Uninfected**' outcomes.



### Training Accuracy and Loss Curves
<div align="center">
  <img src="Images/cnn_model_performance.png" width="800">
</div>


The figures present the model's performance across the training epochs. 

1. The **Accuracy Plot** on the left indicates the trend in classification accuracy for both the training set (**<span style="color:blue">blue curve</span>**) and the validation set (**<span style="color:orange">orange curve</span>**). A higher accuracy indicates better performance of the model in correctly classifying the input data.

2. The **Loss Plot** on the right tracks the model's loss or error rate over the same epochs for both training (**<span style="color:blue">blue curve</span>**) and validation (**<span style="color:orange">orange curve</span>**). The declining trend signifies the model's improving ability to make accurate predictions by minimizing error.

## LIME (Local Interpretable Model-Agnostic Explanations) Analysis

### Overview
In addition to building a CNN model for malaria cell image classification, we have conducted an interpretability analysis using the LIME framework. This analysis aims to understand the decision-making process of our CNN model by highlighting the specific features in the cell images that contribute to its predictions.

### Purpose of LIME Analysis
The primary goal of the LIME analysis is to bring transparency to our CNN model, making its predictions more understandable and trustworthy, especially in a medical context where explanations are as crucial as the predictions themselves.

### Key Steps in the LIME Analysis
The LIME analysis involved several key steps:
1. **Image Perturbation**: Generating variations of the input image to understand how different features affect the model's predictions.
2. **Model Prediction on Perturbations**: Analyzing how these variations influence the model's classification decisions.
3. **Interpretable Model Construction**: Building a simpler, linear model to approximate the predictions of the complex CNN model.
4. **Explanation Extraction**: Identifying which features (or superpixels in this case) of the image were most influential in the model's predictions.

### Insights from the LIME Analysis
The LIME analysis provided valuable insights into the model's behavior, revealing which aspects of the malaria cell images were most critical for classification. This understanding can help improve model design and provide medical professionals with additional context for the model's predictions.

### Visualizations and Results
The LIME analysis included visualizations that highlighted the influential regions in the cell images that led to specific classifications. These visualizations offer a tangible way to interpret the complex workings of the CNN model.

<div align="center">
  <img src="Images/LIME.png" width="800">
</div>


### Conclusion of the LIME Analysis
The LIME analysis played a crucial role in validating and interpreting our CNN model's predictions. By understanding the 'why' behind the model's decisions, we can ensure that our approach to malaria cell classification is both effective and interpretable, leading to better trust and usability in real-world applications.

## Enhancing Model Interpretability with Class Activation Mapping (CAM)

Following our exploration with LIME for model interpretability, we delve deeper into understanding our Convolutional Neural Network's decision-making process through Class Activation Mapping (CAM). This technique, rooted in the innovative work by MIT researchers ("Learning Deep Features for Discriminative Localization"), enables us to visualize the specific regions within the cell images that our model focuses on when making predictions. This visualization not only demystifies the CNN's operations but also significantly enhances the model's transparency.

### The Power of Global Average Pooling (GAP)

At the heart of our CAM implementation is the Global Average Pooling (GAP) layer, strategically positioned after the last convolutional layer of our CNN. Unlike traditional dense layers, the GAP layer reduces the dimensionality of the feature maps while retaining spatial information. This allows for a direct correlation between the last convolutional layer's activations and the final classification output, making it possible to highlight the discriminative features within the image that lead to a prediction.

### Visualizing Discriminative Regions with CAM

The process of generating Class Activation Maps involves several key steps:
1. **Prediction and Weight Extraction**: For a given image, we first predict its class (infected or uninfected) and extract the weights associated with that class from the model's output layer.
2. **Activation Map Retrieval**: We then fetch the activation maps from the last convolutional layer, which contain rich spatial information about the features detected in the image.
3. **Weighted Summation and Upsampling**: By computing a weighted sum of these activation maps, using the extracted weights, we generate a CAM that highlights the regions of interest. This CAM is then upsampled to the size of the original input image for easy visualization.
4. **Interpretation and Insight**: Overlaying the CAM on the original image allows us to see exactly which parts of the cell the model identifies as indicative of malaria infection, providing invaluable insights into the model's focus areas and prediction rationale.

### Implementation and Results

We applied CAM to our CNN model trained for malaria cell classification, resulting in compelling visual evidence of the model's attention in both parasitized and uninfected cells. The generated heatmaps clearly delineate the areas within the cells that most significantly influence the model's predictions, offering a window into the model's "thought process".

<div align="center">
  <img src="Images/cam_plots.png" width="800">
</div>

*Example of Class Activation Maps highlighting discriminative regions for malaria classification.*

### Conclusion on CAM's Impact

The integration of CAM into our project not only enhances the interpretability of our CNN model but also provides a powerful tool for medical professionals to understand and trust the automated diagnosis process. By revealing the critical areas within cell images that lead to a model's prediction, CAM bridges the gap between AI's advanced capabilities and the need for transparent, explainable medical diagnostics.


## Setting Up the Environment
To run the notebooks in this repository, you can create a Conda environment with all the necessary dependencies using the provided **malaria_detection_env.yml** file. Follow these steps:

1. Ensure you have [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.
2. Clone or download this repository to your local machine.
3. Navigate to the repository directory in your terminal.
4. Run `conda env create -f malaria_detection_env.yml` to create the environment.
5. Once the environment is created, activate it with `conda activate malaria_detection_env.yml`.

## Conclusion and Future Work
The model's strong performance underscores its potential as a diagnostic aid for rapid malaria detection. The project showcases the efficacy of CNNs in medical image analysis and their potential to support healthcare initiatives.

Future enhancements may include:
- Exploring advanced neural network architectures.
- Further hyperparameter optimization.
- Investigating transfer learning approaches for performance improvement.


## Acknowledgments
- A special thanks to **Dr. Ehsan (Sam) Gharib-Nezhad** and **Dr. Amirhossein Kardoost** for their insightful feedback and suggestions that significantly enhanced the quality of this project. Their expertise and thoughtful reviews were tremendously helpful. You can find more about their works on [EhsanGharibNezhad](https://github.com/EhsanGharibNezhad) and [Amir_kd](https://github.com/Amirhk-dev).
- Data provided by [Kaggle's Malaria Cell Images Dataset](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria).
- World Health Organization (WHO) for malaria statistics.

## References
1. [Our World in Data - Malaria](https://ourworldindata.org/malaria)

