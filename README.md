# Analysing reports of events involving automated vehicles with LLM model


## Overview
![LLM Personas](figures/LLMs_personas.png)
A Ghibli-style persona illustration of the Large Language Models.😁

This study evaluates the effectiveness of large language model-based personas for assessing external Human-Machine Interfaces (eHMIs) in automated vehicles. 13 different models namely BakLLaVA, ChatGPT-4o, DeepSeek-VL2-Tiny, Gemma3:12B, Gemma3:27B, Granite Vision 3.2, LLaMA 3.2 Vision, LLaVA-13B, LLaVA-34B, LLaVA-LLaMA-3, LLaVA-Phi3, MiniCPM-V and Moondream were tasked with simulating pedestrian decision making for 227 vehicle images equipped with eHMI. Confidence scores (0-100) were collected under two conditions: no memory (images independently assessed) and memory-enabled (conversation history preserved), each in 15 independent trials. The model outputs were compared with the ratings of 1,438 human participants. Gemma3:27B achieved the highest correlation with humans without memory (r = 0.85), while ChatGPT-4o performed best with memory (r = 0.81). DeepSeek-VL2-Tiny and BakLLaVA showed little sensitivity to context, and LLaVA-LLaMA-3, LLaVA-Phi3, LLaVA-13B and Moondream consistently produced limited-range output.

## Citation
If you use the gans-traffic for academic work please cite the following paper:

> Alam, M. S., & Bazilinskyy, P. (2025). Cross or Nah? LLMs Get in the Mindset of a Pedestrian in front of Automated Car with an eHMI. 17th International Conference on Automotive User Interfaces and Interactive Vehicular Applications. Brisbane, QLD, Australia. https://doi.org/10.1145/3744335.3758477

## Usage of the code


## Getting started
[![Python Version](https://img.shields.io/badge/python-3.9.21-blue.svg)](https://www.python.org/downloads/release/python-3919/)
[![Package Manager: uv](https://img.shields.io/badge/package%20manager-uv-green)](https://docs.astral.sh/uv/)

Tested with **Python 3.9.21** and the [`uv`](https://docs.astral.sh/uv/) package manager.  
Follow these steps to set up the project.

**Step 1:** Install `uv`. `uv` is a fast Python package and environment manager. Install it using one of the following methods:

**macOS / Linux (bash/zsh):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

**Alternative (if you already have Python and pip):**
```bash
pip install uv
```

**Step 2:** Fix permissions (if needed):t

Sometimes `uv` needs to create a folder under `~/.local/share/uv/python` (macOS/Linux) or `%LOCALAPPDATA%\uv\python` (Windows).  
If this folder was created by another tool (e.g. `sudo`), you may see an error like:
```lua
error: failed to create directory ... Permission denied (os error 13)
```

To fix it, ensure you own the directory:

### macOS / Linux
```bash
mkdir -p ~/.local/share/uv
chown -R "$(id -un)":"$(id -gn)" ~/.local/share/uv
chmod -R u+rwX ~/.local/share/uv
```

### Windows
```powershell
# Create directory if it doesn't exist
New-Item -ItemType Directory -Force "$env:LOCALAPPDATA\uv"

# Ensure you (the current user) own it
# (usually not needed, but if permissions are broken)
icacls "$env:LOCALAPPDATA\uv" /grant "$($env:UserName):(OI)(CI)F"
```

**Step 3:** After installing, verify:
```bash
uv --version
```

**Step 4:** Clone the repository:
```command line
git clone https://github.com/Shaadalam9/llms-av-crowdsourced
cd llms-av-crowdsourced
```

**Step 5:** Ensure correct Python version. If you don’t already have Python 3.9.19 installed, let `uv` fetch it:
```command line
uv python install 3.9.21
```
The repo should contain a .python-version file so `uv` will automatically use this version.

**Step 6:** Create and sync the virtual environment. This will create **.venv** in the project folder and install dependencies exactly as locked in **uv.lock**:
```command line
uv sync --frozen
```

**Step 7:** Activate the virtual environment:

**macOS / Linux (bash/zsh):**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (cmd.exe):**
```bat
.\.venv\Scripts\activate.bat
```

**Step 8:** 

Download Ollama locally in your system and run it. [https://ollama.com/](https://ollama.com/)


**Step 9:** Run the code:
```command line
python3 analysis.py
```


### Configuration of project

Configuration of the project needs to be defined in `llms-av-crowdsourced/config`. Please use the `default.config` file for the required structure. If no custom config file is provided, `default.config` is used. The config file has the following parameters:

- **data** [String]: Specifies the directory where your image files are stored.  
- **output** [String]: Directory path where model outputs will be saved.  
- **crowdsourced_data** [String]: Specifies the directory path where crowdsourcing results are present.  
- **figures** [String]: Directory path to save final figures.  
- **font_family** [String]: Lists the font families to be used in visualisation outputs (e.g., `"Open Sans, verdana, arial, sans-serif"`).  
- **font_size** [Integer]: Sets the base font size for graphs and other visual elements (e.g., `12`).  
- **random_seed** [List of Integers]: A list of seed values for reproducible runs (e.g., `[42]`).  
- **process_csv_files** [Boolean]: Determines whether raw model output CSVs should be processed and aggregated before analysis. When `true`, the system runs CSV parsing and cleaning steps prior to generating figures or statistics.  
- **use_history** [Boolean]: Determines whether conversation history is maintained and utilised during processing. When `true`, previous interactions are stored and influence current model responses.  
- **delete_history_files** [Boolean]: 
- **max_memory_messages** [Integer]: Specifies the maximum number of previous conversation messages to retain in memory for context (e.g., `6`).  
- **temperature** [Float]: Adjusts the randomness of the model's response generation. Lower values produce more deterministic outputs, while higher values introduce variability (e.g., `0.8`).  
- **plotly_template** [String]: Determines the styling template used for graphs and visualisations (e.g., `"plotly_white"`).  
- **logger_level** [String]: Sets the logging level (e.g., `"info"` or `"debug"`) to control the verbosity of runtime messages.  
- **model_names** [List of Strings]: A list of model identifiers that the system will sequentially use to process images (e.g., `["gemma"]`).  
- **base_prompt** [String]: Provides the initial context for the model, setting the stage for how it should interpret its task (e.g., describing the user's perspective as a pedestrian).  
- **history_intro** [String]: Contains introductory text that precedes the conversation history, informing the model how previous responses may influence its current decision-making.  
- **current_image_instruction** [String]: Offers specific instructions for interpreting the current image, guiding the model on how to generate its response.  
- **prompt** [String]: The main directive provided to the model for evaluating the image. This prompt instructs the model to first interpret the meaning of any digital display, then provide a confidence rating in this format: `"Confidence: [numeric value]. Meaning: [briefly explain]"`.  



## Results

### Model-by-Model Comparison (Without vs With Past History)

---

#### **BakLLaVA**

| Without Past History | With Past History |
|---|---|
| [![BakLLaVA without memory](figures/scatter_plot_bakllava_without_memory.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/llms-av-crowdsourced/blob/main/figures/scatter_plot_bakllava_without_memory.html)<br>**Without memory context** | [![BakLLaVA with memory](figures/scatter_plot_bakllava_with_memory.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/llms-av-crowdsourced/blob/main/figures/scatter_plot_bakllava_with_memory.html)<br>**With memory context** |

---

#### **Gemma3 12B**

| Without Past History | With Past History |
|---|---|
| [![Gemma3 12B without memory](figures/scatter_plot_gemma3:12b_without_memory.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/llms-av-crowdsourced/blob/main/figures/scatter_plot_gemma3:12b_without_memory.html) | [![Gemma3 12B with memory](figures/scatter_plot_gemma3:12b_with_memory.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/llms-av-crowdsourced/blob/main/figures/scatter_plot_gemma3:12b_with_memory.html) |

---

#### **Gemma3 27B**

| Without Past History | With Past History |
|---|---|
| [![Gemma3 27B without memory](figures/scatter_plot_gemma3:27b_without_memory.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/llms-av-crowdsourced/blob/main/figures/scatter_plot_gemma3:27b_without_memory.html) | [![Gemma3 27B with memory](figures/scatter_plot_gemma3:27b_with_memory.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/llms-av-crowdsourced/blob/main/figures/scatter_plot_gemma3:27b_with_memory.html) |

---

#### **Granite3.2 Vision**

| Without Past History | With Past History |
|---|---|
| [![Granite3.2 Vision without memory](figures/scatter_plot_granite3.2-vision_without_memory.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/llms-av-crowdsourced/blob/main/figures/scatter_plot_granite3.2-vision_without_memory.html) | [![Granite3.2 Vision with memory](figures/scatter_plot_granite3.2-vision_with_memory.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/llms-av-crowdsourced/blob/main/figures/scatter_plot_granite3.2-vision_with_memory.html) |

---

#### **LLaMA3.2 Vision**

| Without Past History | With Past History |
|---|---|
| [![LLaMA3.2 Vision without memory](figures/scatter_plot_llama3.2-vision_without_memory.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/llms-av-crowdsourced/blob/main/figures/scatter_plot_llama3.2-vision_without_memory.html) | [![LLaMA3.2 Vision with memory](figures/scatter_plot_llama3.2-vision_with_memory.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/llms-av-crowdsourced/blob/main/figures/scatter_plot_llama3.2-vision_with_memory.html) |

---

#### **LLaVA-LLaMA3**

| Without Past History | With Past History |
|---|---|
| [![LLaVA-LLaMA3 without memory](figures/scatter_plot_llava-llama3_without_memory.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/llms-av-crowdsourced/blob/main/figures/scatter_plot_llava-llama3_without_memory.html) | [![LLaVA-LLaMA3 with memory](figures/scatter_plot_llava-llama3_with_memory.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/llms-av-crowdsourced/blob/main/figures/scatter_plot_llava-llama3_with_memory.html) |

---

#### **LLaVA-Phi3**

| Without Past History | With Past History |
|---|---|
| [![LLaVA-Phi3 without memory](figures/scatter_plot_llava-phi3_without_memory.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/llms-av-crowdsourced/blob/main/figures/scatter_plot_llava-phi3_without_memory.html) | [![LLaVA-Phi3 with memory](figures/scatter_plot_llava-phi3_with_memory.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/llms-av-crowdsourced/blob/main/figures/scatter_plot_llava-phi3_with_memory.html) |

---

#### **LLaVA 13B**

| Without Past History | With Past History |
|---|---|
| [![LLaVA 13B without memory](figures/scatter_plot_llava:13b_without_memory.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/llms-av-crowdsourced/blob/main/figures/scatter_plot_llava:13b_without_memory.html) | [![LLaVA 13B with memory](figures/scatter_plot_llava:13b_with_memory.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/llms-av-crowdsourced/blob/main/figures/scatter_plot_llava:13b_with_memory.html) |

---

#### **LLaVA 34B**

| Without Past History | With Past History |
|---|---|
| [![LLaVA 34B without memory](figures/scatter_plot_llava:34b_without_memory.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/llms-av-crowdsourced/blob/main/figures/scatter_plot_llava:34b_without_memory.html) | [![LLaVA 34B with memory](figures/scatter_plot_llava:34b_with_memory.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/llms-av-crowdsourced/blob/main/figures/scatter_plot_llava:34b_with_memory.html) |

---

#### **MiniCPM-V**

| Without Past History | With Past History |
|---|---|
| [![MiniCPM-V without memory](figures/scatter_plot_minicpm-v_without_memory.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/llms-av-crowdsourced/blob/main/figures/scatter_plot_minicpm-v_without_memory.html) | [![MiniCPM-V with memory](figures/scatter_plot_minicpm-v_with_memory.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/llms-av-crowdsourced/blob/main/figures/scatter_plot_minicpm-v_with_memory.html) |

---

#### **MoonDream**

| Without Past History | With Past History |
|---|---|
| [![MoonDream without memory](figures/scatter_plot_moondream_without_memory.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/llms-av-crowdsourced/blob/main/figures/scatter_plot_moondream_without_memory.html) | [![MoonDream with memory](figures/scatter_plot_moondream_with_memory.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/llms-av-crowdsourced/blob/main/figures/scatter_plot_moondream_with_memory.html) |

---

#### **DeepSeek VL2**

| Without Past History | With Past History |
|---|---|
| [![DeepSeek VL2 without memory](figures/scatter_plot_deepseek-vl2_without_memory.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/llms-av-crowdsourced/blob/main/figures/scatter_plot_deepseek-vl2_without_memory.html) | [![DeepSeek VL2 with memory](figures/scatter_plot_deepseek-vl2_with_memory.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/llms-av-crowdsourced/blob/main/figures/scatter_plot_deepseek-vl2_with_memory.html) |

---

#### **All Models (Merged)**

| Without Past History | With Past History |
|---|---|
| [![All models without memory](figures/merged_without_memory.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/llms-av-crowdsourced/blob/main/figures/merged_without_memory.html) | [![All models with memory](figures/merged_with_memory.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/llms-av-crowdsourced/blob/main/figures/merged_with_memory.html) |

---

#### **Correlation Matrix**

**Without Past History**  
[![Correlation matrix without memory](figures/spearman_correlation_matrix_without_memory.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/llms-av-crowdsourced/blob/main/figures/spearman_correlation_matrix_without_memory.html)  

**With Past History**  
[![Correlation matrix with memory](figures/spearman_correlation_matrix_with_memory.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/llms-av-crowdsourced/blob/main/figures/spearman_correlation_matrix_with_memory.html)  




## License
This project is licensed under the MIT License - see the LICENSE file for details.
