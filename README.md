# Theme Tagging using ELMO

## Overview
The "Theme tagging using ELMO" project is a Python-based tool that facilitates topic detection of employee reviews about their employer using ELMO embeddings. It aims to help identify and categorize the main themes or topics discussed in employee reviews, making it easier to understand the sentiments and concerns of employees.

## How it Works
This project relies on ELMO (Embeddings from Language Models) embeddings to analyze employee reviews and determine the most relevant themes or topics. Here's a high-level overview of how it works:

1. **Ontology File**: To get started with this project, you need an ontology file (sample is already there with this repo). This file should contain a predefined list of topics, along with a set of well-distinct sentences or phrases associated with each topic. The ontology file serves as a reference for identifying topics within employee reviews.

2. **ELMO Embeddings**: ELMO embeddings are used to represent words and sentences as continuous vector representations. The project computes the average ELMO embedding vector for each sentence in an employee review.

3. **Topic Assignment**: For each sentence in the review, the project calculates the similarity between the sentence's embedding vector and the embedding vectors of sentences associated with each topic in the ontology file. The topic with the highest similarity score is assigned to the sentence.

4. **Theme Identification**: Once all sentences are processed, the project aggregates the assigned topics to identify the main themes discussed in the employee review. This provides insights into the specific areas of concern or interest among employees.

## Requirements
To run this project, you'll need the following:

- Python (>=3.6)
- ELMO embeddings model
- Ontology file with predefined topics and sentences
- Required Python libraries (e.g., TensorFlow, NumPy)
