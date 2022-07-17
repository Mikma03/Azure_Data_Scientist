
# Azure Data Scientist Associate Certification Guide

- GitHub repo for the book
  - https://github.com/PacktPublishing/Azure-Data-Scientist-Associate-Certification-Guide


# TOC


# Chapter 1

- An Overview Of Modern Data Science
  - The evolution of data science
  - Working on a data science project
  - Show More Items
  - Using Spark in data science
  - Adopting the DevOps mindset
  - Summary
  - Further reading

# CHapter 2
- Deploying Azure Machine Learning Workspace Resources
queue
  - Technical requirements
  - Deploying Azure ML through the portal
    - Using the deployment wizard
  - Deploying Azure ML via the CLI
    - Deploying Azure Cloud Shell
    - Using the Azure CLI
    - Installing the Azure ML CLI extension
    - Deploying Azure ML using the az ml command
    - Cleaning up the CLI resource group
  - Alternative ways to deploy an Azure ML workspace
  - Exploring the deployed Azure resources
    - Understanding Role-Based Access Control (RBAC)
    - RBAC inheritance
    - Creating custom roles
    - Assigning roles in the Azure ML workspace
  - Summary
  - Questions
  - Further reading


# Chapter 3

- Azure Machine Learning Studio Components
  - Technical requirements
  - Interacting with the Azure ML resource
  - Exploring the Azure ML Studio experience
  - Authoring experiments within Azure ML Studio
  - Tracking data science assets in Azure ML Studio
  - Managing infrastructure resources in Azure ML Studio
  - Summary


# Chapter 4

- Configuring The Workspace
  - Technical requirements
  - Provisioning compute resources
    - Compute instances
    - Compute clusters
    - Inference clusters
    - Attached compute
  - Connecting to datastores
    - Types of datastores
    - Datastore security considerations
    - Working with datasets
  - Registering datasets
  - Exploring the dataset
  - Data drift detection
  - Summary
  - Questions
  - Further reading


# Chapter 5

- Letting The Machines Do The Model Training
  - Technical requirements
  - Configuring an AutoML experiment
    - Registering the dataset
    - Returning to the AutoML wizard
  - Monitoring the execution of the experiment
  - Deploying the best model as a web service
    - Understanding the deployment of the model
    - Cleaning up the model deployment
  - Summary
  - Question
  - Further reading


# Chapter 6

- Visual Model Training And Publishing
  - Technical requirements
  - Overview of the designer
    - The authoring screen/view
    - Understanding the asset library
    - Exploring the asset's inputs and outputs
  - Building the pipeline with the designer
    - Acquiring the data
    - Preparing the data and training the model
    - Executing the training pipeline
  - Creating a batch and real-time inference pipeline
    - Creating a batch pipeline
    - Creating a real-time pipeline
  - Deploying a real-time inference pipeline
  - Summary
  - Question
  - Further reading


# Chapter 7

- The AzureML Python SDK
  - Technical requirements
  - Overview of the Python SDK
  - Working in AzureML notebooks
  - Basic coding with the AzureML SDK
    - Authenticating from your device
    - Working with compute targets
    - Defining datastores
    - Working with datasets
  - Working with the AzureML CLI extension
  - Summary
  - Questions
  - Further reading


# Chapter 8: 

- Experimenting With Python Code
  - Technical requirements
  - Training a simple sklearn model within notebooks
  - Tracking metrics in Experiments
    - Tracking model evolution
    - Using MLflow to track Experiments
  - Scaling the training process with compute clusters
    - Exploring the outputs and logs of a run
    - Understanding execution environments
    - Training the diabetes model on a compute cluster
    - Utilizing more than a single compute node during model training
  - Summary
  - Questions
  - Further reading


# Chapter 9

- Optimizing The ML Model
  - Technical requirements
  - Hyperparameter tuning using HyperDrive
    - Using the early termination policy
  - Running AutoML experiments with code
  - Summary
  - Questions
  - Further reading


# Chapter 10

- Understanding Model Results queue
  - Technical requirements
  - Creating responsible machine learning models
  - Interpreting the predictions of the model
    - Training a loans approval model
    - Using the tabular explainer
    - Understanding the tabular data interpretation techniques
    - Reviewing the interpretation results
  - Analyzing model errors
  - Detecting potential model fairness issues
  - Summary
  - Questions
  - Further reading


# Chapter 11

- Working With Pipelines queue
  - Technical requirements
  - Understanding AzureML pipelines
  - Authoring a pipeline
    - Troubleshooting code issues
  - Publishing a pipeline to expose it as an endpoint
  - Scheduling a recurring pipeline
  - Summary
  - Questions
  - Further reading


# Chapter 12

- Operationalizing Models With Code
  - Technical requirements
  - Understanding the various deployment options
  - Registering models in the workspace
  - Deploying real-time endpoints
    - Understanding the model deployment options
    - Profiling the model's resource requirements
    - Monitoring with Application Insights
    - Integrating with third-party applications
  - Creating a batch inference pipeline
  - Summary
  - Questions
  - Further reading