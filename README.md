# About

This repository contains the code for the AI service for the FAIRWork project.

## Setup
1. Clone the repository
2. Install the requirements by running `pip install -r requirements_dev.txt`
3. mark the `src` folder as a source root (in your IDE)
4. mark the `resources` folder as a resource root

For the integration of the knowledge base, the following step is required:

5. (optional) add the files `knowledge_base_config.yaml` and `knowledge_base_credentials.yaml` to the `resources` folder (NOTE: these files are not included in the repository for security reasons). However, examples for their structure are provided.

The repo contains a data preparation script (`dummy_data_generator.py`). 
So in order to train a model one does not need to connect to the knowledge base.


## Usage
The repository includes an already trained model (`resources/trained-models/model1.pt`).
For querying the model for predicts/allocations you jump to section [API integration via REST API](#API integration via REST API).


## Swagger-UI

The REST API endpoint is documented using Swagger-UI. 
In the example case `http://127.0.0.1:5000/` will redirect to the Swagger-UI documentation.

The Swagger-UI looks like this:
![Swagger-UI-screenshot](resources/readme-content/swagger-ui.png)

One can also test the API using the Swagger-UI and perform the same request as above:
![Swagger-UI-screenshot](resources/readme-content/swagger-request.png)
Which will yield the same response as above:
![Swagger-UI-screenshot](resources/readme-content/swagger-response.png)

## Troubleshooting
todo

# FAIRWork Project
Development Repository for AI Services for the FAIRWork Project

“This work has been supported by the FAIRWork project (www.fairwork-project.eu) and has been funded within the European Commission’s Horizon Europe Programme under contract number 101049499. This paper expresses the opinions of the authors and not necessarily those of the European Commission. The European Commission is not liable for any use that may be made of the information contained in this presentation.”

Copyright © RWTH of FAIRWork Consortium
