# DS-4300 Practical 2: Chapin Wilson, Celine Cerezci, Joji Araki, Daniyal Khalid

# Overview

This repository implements Llama and DeepSeek to create a custom LLM using our team's notes for DS4300 as data.

# Installation

Ensure you are using Python 3.11. You can check this using the following command:
python --version
Ensure you have Ollama, Docker, Redis, Llama3.2 and DeepSeek DeepSeek-r1:1.5b downloaded.

# Implementation

## Step 1: Preprocessing the data
To begin, make sure you have DS4300NotesTXT downloaded in your directory of choice. Then, head to preprocess1.py. This code will save a new folder called "processed" to your directory of choice. In preprocces1.py, change the varaible "INPUT_DIR" to your directory that contains the DS4300NotesTXT folder. Change the "OUTPUT_DIR" variable to the your directory of choice, but make sure the folder name is still named "processed". Run the code.

## Step 2: Ingesting the Data

After preprocessing the data, head to compare.py. This python file ingests documents, generates embeddings using a pre-trained language model, and index/query them using three different vector search engines. Update the folder path to your chunk and overlap size of choosing (Ex: ./backend/processed/___). Our team chose 500 -- 50, so the folder path would look like this "./backend/processed/500 -- 50. At this point, you should open docker and run a redis stack. Update the port number on this line as needed: 
redis_client = redis.Redis(host='localhost', port=6383)

Now, run compare.py.

## Step 3: Running main.py

Main.py retrieves text passages from a Redis-based vector index and then use those passages to build a context-specific prompt for a local language model. Update the the port number on this line as needed with the necessary port number: 

redis_client = redis.Redis(host='localhost', port=6383)

Run main.py. The terminal should prompt you to enter a query. You can now query the model for anything related to DS4300. 


# Citations
Mark Fontenot's Website Resources 
https://markfontenot.net/teaching/ds4300/25s-ds4300/

StackOverFlow for general code suggestions

OpenSource AI models including deepseek, OpenAI, Perplexity, ChatGPT
Uses include greater clarification on subjects, he debugging, and for assistance when it comes to
preprocessing.
