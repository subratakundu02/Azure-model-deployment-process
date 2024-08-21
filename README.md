# Movie Review Sentiment Analysis

This is a Flask web application that predicts the sentiment of a movie review (positive or negative) using a custom-trained model based on the Google PaLM model.

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Deploying to Azure](#deploying-to-azure)
- [Environment Variables](#environment-variables)

## Project Structure


## Prerequisites

Before you begin, ensure you have the following installed on your system:

- Python 3.10 or later
- pip (Python package manager)
- Azure CLI (for deployment)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up your environment variables:

    Create a `.env` file in the root directory and add your Google PaLM API key:

    ```
    PALM_API_KEY=your-google-palm-api-key
    ```

## Running the Application

To run the Flask application locally, use the following command:

```bash
python app.py
