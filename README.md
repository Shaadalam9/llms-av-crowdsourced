# Analysing reports of events involving automated vehicles with LLM model


## Overview

## Usage of the code
The code is open-source and free to use. It is aimed for, but not limited to, academic research. We welcome forking of this repository, pull requests, and any contributions in the spirit of open science and open-source code 😍😄 For inquiries about collaboration, you may contact Md Shadab Alam (md_shadab_alam@outlook.com) or Pavlo Bazilinskyy (pavlo.bazilinskyy@gmail.com).

## Getting Started
Tested with Python 3.9.21. To setup the environment run these two commands in a parent folder of the downloaded repository (replace `/` with `\` and possibly add `--user` if on Windows:

**Step 1:**

Clone the repository
```command line
git clone https://github.com/Shaadalam9/llms-av-crowdsourced
```

**Step 2:**

Create a new virtual environment
```command line
python -m venv venv
```

**Step 3:**

Activate the virtual environment
```command line
source venv/bin/activate
```

On Windows use
```command line
venv\Scripts\activate
```

**Step 4:**

Install dependencies
```command line
pip install -r requirements.txt
```

**Step 5:**

Download the supplementary material from [4TU Research Data](xxxx) and save them in the current folder.

**Step 6:**
Download Ollama locally in your system and run it. [Link](https://ollama.com/)

**Step 7:**
Run the main.py script for analysing the images.
```command line
python3 main.py
```

**Step 8:**
Run the analysis.py script for getting the comaprison of the LLMs output and the crowdsourced results.
```command line
python3 analysis.py
```


### Configuration of project

Configuration of the project needs to be defined in `llms-av-crowdsourced/config`. Please use the `default.config` file for the required structure. If no custom config file is provided, `default.config` is used. The config file has the following parameters:

- **data** [String]: Specifies the directory where your image files are stored (e.g., `"data"`).
- **output** [String]: Directory path where model outputs will be saved (e.g., `"_output"`).
- **font_family** [String]: Lists the font families to be used in visualisation outputs (e.g., `"Open Sans, verdana, arial, sans-serif"`).
- **font_size** [Integer]: Sets the base font size for graphs and other visual elements (e.g., `12`).
- **random_seed** [List of Integers]: A list of seed values for reproducible runs (e.g., `[42]`).
- **use_history** [Boolean]: Determines whether conversation history is maintained and utilised during processing. When `true`, previous interactions are stored and influence current model responses.
- **true_history_files** [Boolean]: Controls whether history files (conversation logs and memory) are retained after processing. If set to `false`, these files are deleted after processing.
- **max_memory_messages** [Integer]: Specifies the maximum number of previous conversation messages to retain in memory for context (e.g., `6`).
- **temperature** [Float]: Adjusts the randomness of the model's response generation. Lower values produce more deterministic outputs, while higher values introduce variability (e.g., `0.8`).
- **plotly_template** [String]: Determines the styling template used for graphs and visualisations (e.g., `"plotly_white"`).
- **logger_level** [String]: Sets the logging level (e.g., `"info"` or `"debug"`) to control the verbosity of runtime messages.
- **model_names** [List of Strings]: A list of model identifiers that the system will sequentially use to process images (e.g., `["gemma"]`).
- **base_prompt** [String]: Provides the initial context for the model, setting the stage for how it should interpret its task (e.g., describing the user's perspective as a pedestrian).
- **history_intro** [String]: Contains introductory text that precedes the conversation history, informing the model how previous responses may influence its current decision-making.
- **current_image_instruction** [String]: Offers specific instructions for interpreting the current image, guiding the model on how to generate its response.
- **prompt** [String]: The main directive provided to the model for evaluating the image. This prompt instructs the model to first interpret the meaning of any digital display, then provide a confidence rating in this format:  
  `"Meaning: [briefly explain]. Confidence: [numeric value]"`


## License
This project is licensed under the MIT License - see the LICENSE file for details.