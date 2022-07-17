
<!-- TOC -->

- [Introduction](#introduction)
  - [Udemy:  DP-100: A-Z Machine Learning using Azure Machine Learning](#udemy--dp-100-a-z-machine-learning-using-azure-machine-learning)
- [Azure ML Studio - GUI and Designer plus Azure Portal](#azure-ml-studio---gui-and-designer-plus-azure-portal)
  - [Getting started with Azure ML](#getting-started-with-azure-ml)
  - [Set-up Azure machine Learning Workspace](#set-up-azure-machine-learning-workspace)
  - [Run Experiments and Train Models](#run-experiments-and-train-models)
  - [Deploy and consume the model](#deploy-and-consume-the-model)
  - [Data processing using Azure ML Designer](#data-processing-using-azure-ml-designer)
- [Machine Learning concepts and theory in Azure ML Studio - GUI](#machine-learning-concepts-and-theory-in-azure-ml-studio---gui)
  - [Classification](#classification)
  - [Regression using Azure ML Designer](#regression-using-azure-ml-designer)
  - [Introduction do pandas liblary](#introduction-do-pandas-liblary)
- [Azure Machine Learning with AzureML SDK](#azure-machine-learning-with-azureml-sdk)
  - [Set-up Azure Machine Learning Workspace using SDK](#set-up-azure-machine-learning-workspace-using-sdk)
  - [Run Experiments and Train Models using SDK](#run-experiments-and-train-models-using-sdk)
  - [Optimize and Manage Models using SDK](#optimize-and-manage-models-using-sdk)
    - [Use AutomatedML to Create Optimial Models](#use-automatedml-to-create-optimial-models)
    - [Use Azure Hyperdrive to Tune Hyperparameters](#use-azure-hyperdrive-to-tune-hyperparameters)
    - [use model explainers to interpret models](#use-model-explainers-to-interpret-models)
  - [Model deplyment using SDK](#model-deplyment-using-sdk)
  - [Databricks section](#databricks-section)

<!-- /TOC -->

# Introduction

All concepts and labs - hands on parts

## Udemy:  DP-100: A-Z Machine Learning using Azure Machine Learning

- https://www.udemy.com/course/machine-learning-using-azureml/


# Azure ML Studio - GUI and Designer plus Azure Portal

## Getting started with Azure ML

- 12. What is Azure ML and high level architecture.

- 13. Azure ML Experiment Workflow

- 14. Azure ML Cheat Sheet for Model Selection


## Set-up Azure machine Learning Workspace

- 17. Understand the AzureMLService Architecture

- 18. Create the AzureML Workspace

- 18.  Create the AzureML Workspace

- 19. View and Manage Workspace Settings

- 20. Overview of New AzureML Studio

- 21. DP-100 Exam Coverage So far.

- 22. What is AzureML Datastore and Dataset?

- 23. Create and Register a Datastore

- 24. Create a Dataset

- 25. Explore the AzureML Dataset

- 26. Understanding the AzureML Compute Resources

- 27. Create a Compute Cluster and Compute Instance


## Run Experiments and Train Models

- 28.  What is an AzureML Pipeline?

- 29.  Create a Pipeline using AzureML Designer

- 30.  Submit the Designer Pipeline run


## Deploy and consume the model

- 31. Important Update before you begin next video.

- 32. Create an Inference Pipeline

- 33. Deploy a real-time endpoint using Designer

- 34. Create a batch inference pipeline using Designer

- 35. Run a Batch Inference Pipeline from Designer


## Data processing using Azure ML Designer

- 36. Important update - AzureML Designer UI Changes

- 37. Get Data to the workspace

- 38. Import Data to the workspace from external sources

- 39. Edit Metadata - Column Names

- 40. Understanding the Run

- 41. Edit Metadata - Data Type

- 42. Export Data to the Blob Storage

- 43. Add Columns to the Dataset

- 44. Add Rows to the Dataset

- 45. Normalization of Data Part 1

- 46. Normalization of Data Part 2

- 47. Clean Missing Data

- 48. Partition and Sample Data Part 1

- 49. Partition and Sample Data Part 2


# Machine Learning concepts and theory in Azure ML Studio - GUI

## Classification

- 50. What is Logistic Regression

- 51. Two Class Logistic Regression - Problem Statement

- 52. Data Preparation for Two Class Classification

- 53. Train the Model for Logistic Regression

- 54. Evaluate the Model Part 1

- 55. Evaluate the Model - Confusion Matrix

- 56. Evaluate the Model - AUC ROC

- 57. Parameters of Two Class Logistics Regression

- 58. What is Decision Tree?

- 59. Ensemble Learning in Decision Tree

- 60. Bagging and Boosting in Decision Tree

- 61. Hands On - Train the Two Class Boosted Decision Tree

- 62. Evaluate and Compare Decision Tree output


## Regression using Azure ML Designer 

- 63.  What is Linear Regression?

- 64. Ordinary Least Square and Common Errors

- 65. Hands On - Automobile Price Predictions Data Analysis

- 66. Hands On - Automobile Price Predictions Data Processing

- 67. Hands On - Automobile Price Predictions Train Model

- 68. Hands On - Automobile Price Predictions Evaluate

- 69. R-Squared or Coefficient of Determination

- 70. Math Behind Gradient Descent

- 71. Gradient Descent Explained

- 72. Online or Stochastic Gradient Descent

- 73. Experiment - Linear Regression using Online Gradient Descent

- 74. Evaluate Linear Regression using Online Gradient Descent


## Introduction do pandas liblary

- 75. A note on Anaconda and Spyder.

- 76. What this section is about?

- 77. Pandas - Import Data for Experiments

- 78. Pandas - Import Data Part 2

- 79. Select Columns using Pandas

- 80. Select Columns By drop method

- 81. Add columns and rows

- 82. Clean Missing Data

- 83. Edit Metadata of columns using Pandas

- 84. Create Summary Statistics using describe

- 85. Clip Values - Remove Outliers using Constants

- 86. Clip Values - Remove Outliers with Percentiles

- 87. Convert and Save a delimited file using Pandas

- 88. Data Normalization

- 89. Label Encoding of String Categorical data

- 90. Why Hot encoding is required?

- 91. Hot Encoding using Pandas get_dummies

- 92. Split The Data for training and testing

- 93. Build Logistic Regression using Python - Part 1

- 94. Build Logistic Regression using Python - Part 2


# Azure Machine Learning with AzureML SDK


## Set-up Azure Machine Learning Workspace using SDK

- 95. Introduction to AzureML SDK

- 96. Create AzureML Workspace using SDK

- 97. Verify the Workspace and Write the Workspace Config File

- 98. Create and Register a Datastore using AzureML SDK

- 99. Create and Register a Dataset using SDK

- 100. Access Workspace, Datastore and Datasets using SDK

- 101. Pandas Dataframe and AzureML Dataset conversions

- 102. Upload local data to storage account via datastore


## Run Experiments and Train Models using SDK

- 103. Problem Statement - Run a sample experiment and log values

- 104. Run a sample experiment using AzureML SDK - Part 1

- 105. Run a sample experiment using AzureML SDK - Part 2

- 106. Run a script in Azureml environment - Part 1

- 107. Run a script in Azureml environment - Part 2

- 108. Run a script in Azureml environment - Part 3

- 109. Run a script in Azureml environment - Part 4

- 110. Run a script in Azureml environment - Part 5

- 111. DP-100 Exam Coverage So far.

- 112. Train and Run a Model Script in AzureML Part 1

- 113. Train and Run a Model Script in AzureML Part 2

- 114. Train and Run a Model Script in AzureML Part 3

- 115. Train and Run a Model Script in AzureML Part 4

- 116. Train and Run a Model Script in AzureML Part 5

- 117. Provisioning Compute Cluster using SDK

- 118. Automate Model Training using AzureML SDK

- 119. Automate Model Training - Define Pipeline Steps

- 120. Automate Model Training - Define Run Configuration

- 121. Automate Model Training - Define Build and Run

- 122. Detour - Command Line Arguments

- 123. Automate Model Training - Create Dataprep Step

- 124. Automate Model Training - Create Training Step

- 125. Run the pipeline and see the results


## Optimize and Manage Models using SDK

### Use AutomatedML to Create Optimial Models

- 130. What is Azure AutoML?

- 131. Use the Automated ML interface in Azure Machine Learning studio

- 132. View the AutoML Run Result

- 133. Note on Normalized Macro Recall

- 134. Use Automated ML from the Azure Machine Learning SDK

- 135. Retrieve the Best Model and View results


### Use Azure Hyperdrive to Tune Hyperparameters

- 136. Introduction to Azure Hyperdrive

- 137. Define the Hyperparameter Search Space

- 138. Select a Sampling method

- 139. Define Early Termination Options

- 140. Configure the Hyperdrive run

- 141. Create the Training Script for Hyperdrive run

- 142. Retrieve the Best Model


### use model explainers to interpret models

- 143. Why model explanation is important?

- 144. Understanding Shapley Value

- 145. Interpretability Techniques in Azure

- 146. Implement Interpretability - Initial Set-up

- 147. Implement Interpretability - Global Explanations

- 148. Implement Interpretability - Local Explanations

- 149. Implement Interpretability - Local Explanations Part 2

- 150. Run Interpret Model Script in Azure Workspace

- 151. Visualize Explanations in AzureML Studio

- 152. Retrieve/Download Feature Importance Values.


## Model deplyment using SDK

- 153. Model Deployment Steps

- 154. Understanding Model/Object Serialization

- 155. Hands on - Serialization using Joblib

- 156. Handling OneHotEncoding/Dummy Values in Production

- 157. Hands on - Dummy Variables in Production

- 158. Train the model for webservice deployment

- 159. Register the Model using Run_ID

- 160. Register the Model using local pkl file

- 161. Retrieve all the registered models from the workspace

- 162. Provisioning AKS Production Cluster using SDK

- 163. Create the Inference and Deployment Configuration for Webservice

- 164. Entry Script - Init Function

- 165. Understanding Data processing using JSON, Dictionary and Dataframe

- 166. Entry Script - run Function

- 167. Create webservice deployment object

- 168. Deploy a real-time endpoint using SDK

- 169. Consume the web service from Python program

- 170. Consume the web service as an End Point.


## Databricks section

- 171. Databricks Update to DP-100

- 172. (Optional) What is Big Data?

- 173. (Optional) What is Hadoop?

- 174. What is Spark and Databricks?

- 175. Create an Azure Databricks workspace

- 176. Note on Deleting Databricks resource in Azure Portal

- 177. Note on Increasing vCPU Quota Limits

- 178. Create an Azure Databricks cluster

- 179. Link AzureML Workspace with the Databricks Workspace

- 180. Create and run notebooks in Azure Databricks Part-1

- 181. Create and run notebooks in Azure Databricks Part-2

- 182. Mount Blob storage to Databricks using dbutils Part-1

- 183. Mount Blob storage to Databricks using dbutils Part-2

- 184. Run an sklearn experiment with Databricks Notebook

- 185. Overview to Run a Training script using DatabricksStep in a pipeline

- 186. Saving data to Azure Blob storage from Databricks

- 187. Passing parameters between Azure Databricks notebooks

- 188. Attach a Databricks Cluster as an Attached Compute Target

- 189. Verify Databricks Cluster as Attached Compute

- 190. Databricks Pipeline - Initial Set-up

- 191. Databricks Pipeline - Build DatabricksStep

- 192. Databricks Pipeline - Databricks and Python notebook

- 193. Databricks Pipeline - Submit the pipeline and verify the output
