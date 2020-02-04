#### This is the repository that supplements the blogs posts published [here](https://techcommunity.microsoft.com/t5/ai-customer-engineering-team/deep-learning-with-bert-on-azure-ml-for-text-classification/ba-p/1149262).

The code is organized in the following notebooks:
- 01-data-preparation: prepares and sample the data for fine-tuning the BERT-based model.
- 02-data-classification: performs fine-tuning of the BERT-based model for text classification, showing step by step and running locally.
- 03-data-registration: uploads the prepared and sampled data to an Azure Blob Storage location and registers it as a Dataset within Azure ML.
- 04-data-classification-aml: performs fine-tuning of the BERT-based model for text classification, using Azure ML for distributed remote training and model hyperparameter search.

Notebooks *01-data-preparation* and *02-data-classification* can be run independently of Azure ML.

To run notebooks *03-data-registration* and *04-data-classification-aml* you will need access to an [Azure Subscription]( https://azure.microsoft.com/en-us/free/) and an [Azure ML Worspace]( https://azure.microsoft.com/en-us/free/machine-learning/).
