

<!-- TOC -->

- [Microsoft Certified: Azure Data Scientist Associate (DP-100) Study Guide](#microsoft-certified-azure-data-scientist-associate-dp-100-study-guide)
  - [Certification Path](#certification-path)
    - [Exam DP-100: Designing and Implementing a Data Science Solution on Azure](#exam-dp-100-designing-and-implementing-a-data-science-solution-on-azure)
  - [Skills measured](#skills-measured)
    - [Manage Azure resources for machine learning (25-30%)](#manage-azure-resources-for-machine-learning-25-30)
      - [Create an Azure Machine Learning workspace](#create-an-azure-machine-learning-workspace)
      - [Manage data in an Azure Machine Learning workspace](#manage-data-in-an-azure-machine-learning-workspace)
      - [Manage compute for experiments in Azure Machine Learning](#manage-compute-for-experiments-in-azure-machine-learning)
      - [Implement security and access control in Azure Machine Learning](#implement-security-and-access-control-in-azure-machine-learning)
      - [Set up an Azure Machine Learning development environment](#set-up-an-azure-machine-learning-development-environment)
      - [Set up an Azure Databricks workspace](#set-up-an-azure-databricks-workspace)
    - [Run experiments and train models (20-25%)](#run-experiments-and-train-models-20-25)
      - [Create models by using the Azure Machine Learning designer](#create-models-by-using-the-azure-machine-learning-designer)
      - [Run model training scripts](#run-model-training-scripts)
      - [Generate metrics from an experiment run](#generate-metrics-from-an-experiment-run)
      - [Use Automated Machine Learning to create optimal models](#use-automated-machine-learning-to-create-optimal-models)
      - [Tune hyperparameters with Azure Machine Learning](#tune-hyperparameters-with-azure-machine-learning)
    - [Deploy and operationalize machine learning solutions (35-40%)](#deploy-and-operationalize-machine-learning-solutions-35-40)
      - [Select compute for model deployment](#select-compute-for-model-deployment)
      - [Deploy a model as a service](#deploy-a-model-as-a-service)
      - [Manage models in Azure Machine Learning](#manage-models-in-azure-machine-learning)
      - [Create an Azure Machine Learning pipeline for batch inferencing](#create-an-azure-machine-learning-pipeline-for-batch-inferencing)
      - [Publish an Azure Machine Learning designer pipeline as a web service](#publish-an-azure-machine-learning-designer-pipeline-as-a-web-service)
      - [Implement pipelines by using the Azure Machine Learning SDK](#implement-pipelines-by-using-the-azure-machine-learning-sdk)
      - [Apply ML Ops practices](#apply-ml-ops-practices)
    - [Implement responsible machine learning (5-10%)](#implement-responsible-machine-learning-5-10)
      - [Use model explainers to interpret models](#use-model-explainers-to-interpret-models)
      - [Describe fairness considerations for models](#describe-fairness-considerations-for-models)
      - [Describe privacy considerations for data](#describe-privacy-considerations-for-data)
  - [Additional resources](#additional-resources)

<!-- /TOC -->


# Microsoft Certified: Azure Data Scientist Associate (DP-100) Study Guide


The Microsoft Certified: Azure Data Scientist Associate certification enables you to assess and enhance your knowledge and experience in data science and Machine Learning on Azure. This is an *intermediate level* certification, so I would recommend you to set aside some time to study for the exam and practice using some of the services related to this exam - *'Practice makes perfect'*

In this study guide, I will share with you some of the useful resources you can use to guide you during your learning journey to get this certification.

## Certification Path

| Exam Name                                                                |                                        Link                                        |
| ------------------------------------------------------------------------ | :--------------------------------------------------------------------------------: |
| Exam DP-100: Designing and Implementing a Data Science Solution on Azure | [Exam Details](https://docs.microsoft.com/en-us/learn/certifications/exams/dp-100) |

### Exam DP-100: Designing and Implementing a Data Science Solution on Azure

First place to go is the *Microsoft Learn platform* where a [dedicated learning path](https://docs.microsoft.com/en-ie/users/hugobarona/collections/n532i5pqdez7q6?WT.mc_id=AI-MVP-5004069) is available, **for free**.
Also, you should have a look to the *Resources* section in this study guide where you have useful resources to help you consolidate the knowledge that will help you get the exam and certification. If you prefer to watch videos, instead of read, explaining these core concepts and showing how to get prepared to the exam, then I invite you to have a look at the [Microsoft Exam DP-100 : Designing and Implementing a Data Science Solution on Azure](https://app.pluralsight.com/paths/certificate/microsoft-exam-dp-100-designing-and-implementing-a-data-science-solution-on-azure), available on Pluralsight.

## Skills measured

### Manage Azure resources for machine learning (25-30%)

#### Create an Azure Machine Learning workspace

- [create an Azure Machine Learning workspace](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace?WT.mc_id=AI-MVP-5004069)
- [configure workspace settings](https://docs.microsoft.com/en-us/cli/azure/security/workspace-setting?view=azure-cli-latest?WT.mc_id=AI-MVP-5004069)
- [manage a workspace by using Azure Machine Learning studio](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace?WT.mc_id=AI-MVP-5004069)

#### Manage data in an Azure Machine Learning workspace

- [select Azure storage resources](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-access-data?WT.mc_id=AI-MVP-5004069)
- [register and maintain datastores](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-access-data#create-and-register-datastores?WT.mc_id=AI-MVP-5004069)
- [create and manage datasets](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-register-datasets#create-datasets?WT.mc_id=AI-MVP-5004069)

#### Manage compute for experiments in Azure Machine Learning

- [determine the appropriate compute specifications for a training workload](https://docs.microsoft.com/en-us/azure/machine-learning/concept-compute-target?WT.mc_id=AI-MVP-5004069)
- [create compute targets for experiments and training](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-attach-compute-targets?WT.mc_id=AI-MVP-5004069)
- [configure Attached Compute resources including Azure Databricks](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-attach-compute-targets?WT.mc_id=AI-MVP-5004069)
- [monitor compute utilization](https://docs.microsoft.com/en-us/azure/machine-learning/monitor-azure-machine-learning?WT.mc_id=AI-MVP-5004069)

#### Implement security and access control in Azure Machine Learning

- [determine access requirements and map requirements to built-in roles](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-assign-roles?WT.mc_id=AI-MVP-5004069)
- [create custom roles](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-assign-roles?WT.mc_id=AI-MVP-5004069)
- [manage role membership](https://docs.microsoft.com/en-us/azure/role-based-access-control/built-in-roles?WT.mc_id=AI-MVP-5004069)
- [manage credentials by using Azure Key Vault](https://docs.microsoft.com/en-us/azure/machine-learning/concept-enterprise-security?WT.mc_id=AI-MVP-5004069)

#### Set up an Azure Machine Learning development environment

- [create compute instances](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-manage-compute-instance?tabs=python&WT.mc_id=AI-MVP-5004069)
- [share compute instances](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-manage-compute-instance?tabs=python&WT.mc_id=AI-MVP-5004069)
- [access Azure Machine Learning workspaces from other development environments](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-local?WT.mc_id=AI-MVP-5004069)

#### Set up an Azure Databricks workspace

- [create an Azure Databricks workspace](https://docs.microsoft.com/en-us/azure/databricks/scenarios/quickstart-create-databricks-workspace-portal?WT.mc_id=AI-MVP-5004069)
- [create an Azure Databricks cluster](https://docs.microsoft.com/en-us/azure/databricks/clusters/create?WT.mc_id=AI-MVP-5004069)
- [create and run notebooks in Azure Databricks](https://docs.microsoft.com/en-us/azure/databricks/notebooks/notebooks-manage#create-a-notebook?WT.mc_id=AI-MVP-5004069)
- [link and Azure Databricks workspace to an Azure Machine Learning workspace](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-databricks-automl-environment?WT.mc_id=AI-MVP-5004069)

### Run experiments and train models (20-25%)

#### Create models by using the Azure Machine Learning designer

- [create a training pipeline by using Azure Machine Learning designer](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-designer-automobile-price-train-score?WT.mc_id=AI-MVP-5004069)
- [ingest data in a designer pipeline](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-designer-automobile-price-train-score?WT.mc_id=AI-MVP-5004069)
- [use designer modules to define a pipeline data flow](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-designer-automobile-price-train-score?WT.mc_id=AI-MVP-5004069)
- [use custom code modules in designer](https://docs.microsoft.com/en-us/azure/machine-learning/algorithm-module-reference/create-python-model?WT.mc_id=AI-MVP-5004069)

#### Run model training scripts

- [create and run an experiment by using the Azure Machine Learning SDK](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-1st-experiment-sdk-setup?WT.mc_id=AI-MVP-5004069)
- [configure run settings for a script](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets?WT.mc_id=AI-MVP-5004069)
- [consume data from a dataset in an experiment by using the Azure Machine Learning SDK](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-access-data#create-and-register-datastores?WT.mc_id=AI-MVP-5004069)
- [run a training script on Azure Databricks compute](https://docs.microsoft.com/en-us/azure/databricks/applications/machine-learning/tutorial?WT.mc_id=AI-MVP-5004069)
- [run code to train a model in an Azure Databricks notebook](https://docs.microsoft.com/en-us/azure/databricks/applications/machine-learning/tutorial?WT.mc_id=AI-MVP-5004069)

#### Generate metrics from an experiment run

- [log metrics from an experiment run](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-log-view-metrics?WT.mc_id=AI-MVP-5004069)
- [retrieve and view experiment outputs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-track-monitor-analyze-runs?WT.mc_id=AI-MVP-5004069)
- [use logs to troubleshoot experiment run errors](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-track-monitor-analyze-runs?WT.mc_id=AI-MVP-5004069)
- [use MLflow to track experiments](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow?WT.mc_id=AI-MVP-5004069)
- [track experiments running in Azure Databricks](https://docs.microsoft.com/en-us/azure/databricks/applications/mlflow/tracking?WT.mc_id=AI-MVP-5004069)

#### Use Automated Machine Learning to create optimal models

- [use the Automated ML interface in Azure Machine Learning studio](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-automated-ml-for-ml-models?WT.mc_id=AI-MVP-5004069)
- [use Automated ML from the Azure Machine Learning SDK](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train?WT.mc_id=AI-MVP-5004069)
- [select pre-processing options](https://docs.microsoft.com/en-us/python/api/azureml-train-automl-client/azureml.train.automl.automlconfig.automlconfig?view=azure-ml-py?WT.mc_id=AI-MVP-5004069)
- [select the algorithms to be searched](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train#supported-models?WT.mc_id=AI-MVP-5004069)
- [define a primary metric](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train#primary-metric?WT.mc_id=AI-MVP-5004069)
- [get data for an Automated ML run](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train#training-validation-and-test-data?WT.mc_id=AI-MVP-5004069)
- [retrieve the best model](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train#explore-models-and-metrics?WT.mc_id=AI-MVP-5004069)

#### Tune hyperparameters with Azure Machine Learning

- [select a sampling method](https://docs.microsoft.com/en-us/azure/machine-learning/algorithm-module-reference/tune-model-hyperparameters?WT.mc_id=AI-MVP-5004069)
- [define the search space](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters#define-the-search-space?WT.mc_id=AI-MVP-5004069)
- [define the primary metric](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters#specify-primary-metric-to-optimize?WT.mc_id=AI-MVP-5004069)
- [define early termination options](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters#early-termination?WT.mc_id=AI-MVP-5004069)
- [find the model that has optimal hyperparameter values](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters#find-the-best-model?WT.mc_id=AI-MVP-5004069)

### Deploy and operationalize machine learning solutions (35-40%)

#### Select compute for model deployment

- [consider security for deployed services](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-secure-web-service?WT.mc_id=AI-MVP-5004069)
- [evaluate compute options for deployment](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?WT.mc_id=AI-MVP-5004069)

#### Deploy a model as a service

- [configure deployment settings](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?WT.mc_id=AI-MVP-5004069)
- [deploy a registered model](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?WT.mc_id=AI-MVP-5004069)
- [deploy a model trained in Azure Databricks to an Azure Machine Learning endpoint](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-databricks-automl-environment?WT.mc_id=AI-MVP-5004069)
- [consume a deployed service](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-consume-web-service?WT.mc_id=AI-MVP-5004069)
- [troubleshoot deployment container issues](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-troubleshoot-deployment?WT.mc_id=AI-MVP-5004069)

#### Manage models in Azure Machine Learning

- [register a trained model](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?WT.mc_id=AI-MVP-5004069)
- [monitor model usage](https://docs.microsoft.com/en-us/learn/modules/monitor-models-with-azure-machine-learning/?WT.mc_id=AI-MVP-5004069)
- [monitor data drift](https://docs.microsoft.com/en-us/learn/modules/monitor-models-with-azure-machine-learning/?WT.mc_id=AI-MVP-5004069)

#### Create an Azure Machine Learning pipeline for batch inferencing

- [configure a ParallelRunStep](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-pipeline-batch-scoring-classification?WT.mc_id=AI-MVP-5004069)
- [configure compute for a batch inferencing pipeline](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-run-batch-predictions-designer?WT.mc_id=AI-MVP-5004069)
- [publish a batch inferencing pipeline](https://docs.microsoft.com/en-us/learn/modules/deploy-batch-inference-pipelines-with-azure-machine-learning?WT.mc_id=AI-MVP-5004069)
- [run a batch inferencing pipeline and obtain outputs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-run-batch-predictions-designer?WT.mc_id=AI-MVP-5004069)
- [obtain outputs from a ParallelRunStep](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-debug-parallel-run-step?WT.mc_id=AI-MVP-5004069)

#### Publish an Azure Machine Learning designer pipeline as a web service

- [create a target compute resource](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-attach-compute-targets?WT.mc_id=AI-MVP-5004069)
- [configure an inference pipeline](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-designer-automobile-price-deploy?WT.mc_id=AI-MVP-5004069)
- [consume a deployed endpoint](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-consume-web-service?WT.mc_id=AI-MVP-5004069)

#### Implement pipelines by using the Azure Machine Learning SDK

- [create a pipeline](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-machine-learning-pipelines?WT.mc_id=AI-MVP-5004069)
- [pass data between steps in a pipeline](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-machine-learning-pipelines?WT.mc_id=AI-MVP-5004069)
- [run a pipeline](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-machine-learning-pipelines?WT.mc_id=AI-MVP-5004069)
- [monitor pipeline runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-log-pipelines-application-insights?WT.mc_id=AI-MVP-5004069)

#### Apply ML Ops practices

- [trigger an Azure Machine Learning pipeline from Azure DevOps](https://azure.microsoft.com/pt-pt/blog/automated-machine-learning-and-mlops-with-azure-machine-learning?WT.mc_id=AI-MVP-5004069)
- [automate model retraining based on new data additions or data changes](https://azure.microsoft.com/en-us/blog/retraining-and-updating-azure-machine-learning-models-with-azure-data-factory?WT.mc_id=AI-MVP-5004069)
- [refactor notebooks into scripts](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-convert-ml-experiment-to-production?WT.mc_id=AI-MVP-5004069)
- [implement source control for scripts](https://docs.microsoft.com/en-us/azure/machine-learning/concept-train-model-git-integration?WT.mc_id=AI-MVP-5004069)

### Implement responsible machine learning (5-10%)

#### Use model explainers to interpret models

- select a model interpreter
  - [Model interpretability in Azure Machine Learning (preview)](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-machine-learning-interpretability?WT.mc_id=AI-MVP-5004069)
  - [Use the interpretability package to explain ML models & predictions in Python (preview)](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-machine-learning-interpretability-aml?WT.mc_id=AI-MVP-5004069)
- [generate feature importance data](https://docs.microsoft.com/en-us/azure/machine-learning/algorithm-module-reference/permutation-feature-importance?WT.mc_id=AI-MVP-5004069)

#### Describe fairness considerations for models

- [evaluate model fairness based on prediction disparity](https://docs.microsoft.com/en-us/azure/machine-learning/concept-fairness-ml?WT.mc_id=AI-MVP-5004069)
- [mitigate model unfairness](https://docs.microsoft.com/en-us/azure/machine-learning/concept-fairness-ml#mitigate-unfairness-in-machine-learning-models?WT.mc_id=AI-MVP-5004069)

#### Describe privacy considerations for data

- [describe principles of differential privacy](https://docs.microsoft.com/en-us/azure/machine-learning/concept-differential-privacy?WT.mc_id=AI-MVP-5004069)
- [specify acceptable levels of noise in data and the effects on privacy](https://docs.microsoft.com/en-us/azure/machine-learning/concept-differential-privacy?WT.mc_id=AI-MVP-5004069)


## Additional resources

Below follows the list of additional resources that you should consider and a quick note to the Microsoft Learn collection shared there. I tried to extend the learning paths you have available on the exam's page with some extra modules that I consider relevant to the exam.

<br/>

Best of Luck and share your results with the community once you get certified! 😊💪

| Resource                                                                                                                                                                                                               |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Exam DP-100: Designing and Implementing a Data Science Solution on Azure](https://docs.microsoft.com/en-us/learn/certifications/exams/dp-100?WT.mc_id=AI-MVP-5004069)                                                 |
| [Microsoft Exam DP-100 : Designing and Implementing a Data Science Solution on Azure](https://app.pluralsight.com/paths/certificate/microsoft-exam-dp-100-designing-and-implementing-a-data-science-solution-on-azure) |
| [Microsoft Learn - Microsoft Certified: Azure Data Scientist Associate](https://docs.microsoft.com/en-ie/users/hugobarona/collections/n532i5pqdez7q6?WT.mc_id=AI-MVP-5004069)                                          |  |



