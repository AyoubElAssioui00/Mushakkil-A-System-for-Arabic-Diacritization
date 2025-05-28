# Mushakkil-A-System-for-Arabic-Diacritization

# Mushakkil: An Arabic Diacritization System

Welcome to Mushakkil, a project dedicated to developing systems for automatic Arabic diacritization (Tashkeel). This README provides an overview of the project.

## Table of Contents

1.  [Introduction](#introduction)
2.  [Chapter 1: HMM-Based Diacritization](#chapter-1-hmm-based-diacritization)
    *   [Overview](#overview)
    *   [Dataset](#dataset)
    *   [Methodology](#methodology)
    *   [Performance](#performance)
3.  [Project Structure (HMM Component)](#project-structure-hmm-component)
4.  [Getting Started](#getting-started)
    *   [Prerequisites](#prerequisites)
    *   [Installation](#installation)
5.  [Usage: HMM-Based System](#usage-hmm-based-system)
    *   [Inference with a Pre-trained HMM Model](#inference-with-a-pre-trained-hmm-model)
    *   [Training a New HMM Model](#training-a-new-hmm-model)
    *   [Data Visualization (HMM Training Data)](#data-visualization-hmm-training-data)
6.  [Future Enhancements for the HMM Component](#future-enhancements-for-the-hmm-component)
7.  [Contributing](#contributing)
8.  [License](#license)

## Introduction

Arabic script is primarily consonantal, and diacritics (short vowel marks) are often omitted in everyday texts. However, these diacritics are crucial for correct pronunciation, disambiguation, and various NLP applications like Text-to-Speech (TTS), Information Retrieval, and Machine Translation. Mushakkil aims to provide tools and models for automatically adding these diacritics to undiacritized Arabic text.

This document currently details the implementation and usage of an HMM-based system for this task.

## Chapter 1: HMM-Based Diacritization

This section outlines the system for Arabic diacritization using a Hidden Markov Model (HMM), as detailed in the project report.

### Overview

The HMM-based approach models the diacritization task as a sequence labeling problem.
*   **Observations:** The sequence of Arabic characters in the input text.
*   **Hidden States:** The diacritics (including "no diacritic") corresponding to each character.

The model learns:
*   Initial State Probabilities (π): The likelihood of a diacritic starting a sequence.
*   Transition Probabilities (A): The likelihood of transitioning from one diacritic to another.
*   Emission Probabilities (B): The likelihood of an Arabic character being associated with a specific diacritic.

### Dataset

The HMM model is trained and evaluated using a benchmark dataset derived from the **Tashkeela Corpus**. This dataset was meticulously cleaned and prepared, inspired by the work of Fadel et al. (2019), to address inconsistencies and ensure suitability for training.
*   **Total Lines:** ~55,000
*   **Total Words:** ~2.3 Million
*   **Splits:** Training (50,000 lines), Validation (2,500 lines), Test (2,500 lines).
    *   *Note: The HMM implementation uses a direct train/test split, but the data organization reflects the benchmark for comparability.*

### Methodology

1.  **Data Preprocessing:**
    *   Loading text data.
    *   Cleaning text to retain valid Arabic letters, diacritics, spaces, and periods.
    *   Splitting lines into (character, diacritic_string) pairs.
    *   Extracting character-only sequences for testing.
2.  **HMM Training:**
    *   Parameters are estimated using Maximum Likelihood Estimation (MLE).
    *   Counts of initial diacritics, transitions, and emissions are collected.
    *   Probabilities are calculated with a small smoothing value (e.g., 1e-10) to handle unseen events.
    *   A constraint ensures the space character is always emitted with an empty diacritic.
3.  **Prediction:**
    *   The Viterbi algorithm is used to find the most likely sequence of diacritics for a given sequence of characters.
    *   Log probabilities are used for numerical stability.

### Performance

On the designated test set, the HMM-based system achieved:
*   **Diacritic Error Rate (DER): 39.75%**

While HMMs are simpler and computationally less expensive than deep learning models, this DER provides a solid baseline and demonstrates the capability of HMMs for this complex task. The Markov assumption (dependency only on the previous state) is a known limitation for capturing long-range dependencies crucial in Arabic.

## Project Structure (HMM Component)

The HMM-related components of the Mushakkil project are organized as follows:

Mushakkil/

├── Algorithms/

│ └── HMM.py # Core HMM class (training, Viterbi, DER calculation)

├── constants/

│ ├── ARABIC_LETTERS_LIST.pickle

│ ├── CLASSES_LIST.pickle

│ └── DIACRITICS_LIST.pickle # Pre-defined lists for letters, diacritics

├── Data/

│ └── HMM.py # Scripts for HMM data loading, cleaning, splitting

├── Inference/

│ └── HMM.py # Script to run diacritization using a trained HMM model

├── models/

│ └── arabic_diacritization_hmm.pkl # Example placeholder for a saved HMM model

├── Train/

│ └── HMM.py # Script to train an HMM model

├── Visualization/

│ └── HMM.py # Script for visualizing HMM data characteristics

├── test.ipynb # Jupyter notebook for general testing/experiments

└── README.md # This file



## Getting Started

### Prerequisites

*   Python 3.10
*   `requirements.txt` 

### Installation

1.  Clone the repository:
    ```bash
    git clone <your-repository-url>
    cd Mushakkil
    ```
2.  (Optional) Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  (Optional) Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Ensure your dataset (e.g., the cleaned Tashkeela corpus) is available and accessible by the scripts if you intend to train a new model. 

## Usage: HMM-Based System

The following commands assume you are in the root directory of the `Mushakkil` project (e.g., `~/Mushakkil`).

### Inference with a Pre-trained HMM Model

To diacritize new Arabic text using a pre-trained HMM model (e.g., one saved as `models/arabic_diacritization_hmm.pkl`):

  Run the inference script:
    ```bash
    (base) user@hostname:~/Mushakkil$ python -m Inference.HMM
    ```
   

### Training a New HMM Model

To train the HMM model on your own dataset (or the Tashkeela dataset if you have it prepared):

  Run the training script:
    ```bash
    (base) user@hostname:~/Mushakkil$ python -m Train.HMM
    ```
  This will process the training data, estimate HMM parameters, and should save the trained model (e.g., to the `models/` directory as `arabic_diacritization_hmm.pkl`).

### Data Visualization (HMM Training Data)

To generate visualizations related to the HMM training data (like letter and diacritic frequencies):

1.  Ensure your training data is accessible and `Visualization/HMM.py` is configured to load it.
2.  Run the visualization script:
    ```bash
    (base) user@hostname:~/Mushakkil$ python -m Visualization.HMM
    ```
    This will typically generate and save plots (e.g., to `plots/` directory).

## Future Enhancements for the HMM Component

While other advanced techniques may be explored for Mushakkil, specific improvements for the HMM-based system could include:
*   **Higher-order HMMs:** To capture more context beyond the immediately preceding diacritic.
*   **Integration of Morphological Analysis:** Using morphological features to better inform emission or transition probabilities.
*   **Rule-based Post-processing:** Applying linguistic rules to correct common HMM errors.

## Contributing

Contributions are welcome! If you'd like to contribute to the HMM component or other aspects of Mushakkil, please:
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and commit them (`git commit -am 'Add some feature'`).
4.  Push to the branch (`git push origin feature/your-feature-name`).
5.  Create a new Pull Request.

Please make sure to update tests as appropriate.

## License

MIT License

Copyright (c) [2025] [Ayoub EL Assioui & Mohammed Stifi/Ensias]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
