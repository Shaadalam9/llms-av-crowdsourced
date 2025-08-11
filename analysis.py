import os
import re
import glob
import shutil
import pandas as pd
import plotly.express as px
import plotly.offline as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # noqa: F401
import common
from custom_logger import CustomLogger
from logmod import logs
from models.ollama import OllamaClient

logs(show_level='info', show_color=True)
logger = CustomLogger(__name__)  # use custom logger

# Paths
output_path = common.get_configs("data")
crowdsourced_data = common.get_configs("crowdsourced_data")
font_size = common.get_configs("font_size")
font_family = common.get_configs("font_family")

# Initialise the client for generating text ratings
client = OllamaClient()


class Analysis:
    """
    A class for processing CSV files, averaging LLM results,and plotting comparisons
    between eHMI means and LLM scores.
    """

    # Centralised column rename mapping
    RENAME_MAP = {
        'minicpm-v': 'MiniCPM-V',
        'llava:13b': 'LLaVA-13B',
        'llava:34b': 'LLaVA-34B',
        'llava-llama3': 'LLaVA-LLaMA-3',
        'llama3.2-vision': 'LLaMA 3.2 Vision',
        'moondream': 'Moondream',
        'bakllava': 'BakLLaVA',
        'granite3.2-vision': 'Granite Vision 3.2',
        'llava-phi3': 'LLaVA-Phi3',
        'gemma3:12b': 'Gemma3: 12B',
        'gemma3:27b': 'Gemma3: 27B',
        'deepseek-vl2': 'DeepSeek-VL2',
        'gpt-4o': 'ChatGPT-4o',
        'cross': 'Cross',
        'wait': 'Wait',
        'egocentric': 'Egocentric',
        'allocentric': 'Allocentric',
        'med': 'eHMI Median',
        'ehmi_mean': 'eHMI Mean',
        'lang_encoded': 'Language (es=1)'
    }

    def __init__(self):
        # Set up logging and plotly template
        logs(show_level=common.get_configs("logger_level"), show_color=True)
        self.logger = CustomLogger(__name__)
        self.template = common.get_configs('plotly_template')

        # Constants (moved from module-level)
        self.save_png = True
        self.save_eps = True
        self.base_height_per_row = 20  # Adjust as needed
        self.flag_size = 12
        self.text_size = 12
        self.scale = 1  # scale=3 hangs often

    def save_plotly_figure(self, fig, filename, width=1600, height=1400, save_final=True):
        """Saves a Plotly figure as HTML, PNG, and EPS formats.

        Args:
            fig (plotly.graph_objs.Figure): Plotly figure object.
            filename (str): Name of the file (without extension) to save.
            width (int, optional): Width of the image in pixels. Defaults to 1600.
            height (int, optional): Height of the image in pixels. Defaults to 900.
            save_final (bool, optional): Whether to save a copy to the final folder.
        """
        output_folder = "_output"
        figures = common.get_configs("figures")
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(figures, exist_ok=True)

        self.logger.info(f"Saving html file for {filename}.")
        py.offline.plot(fig, filename=os.path.join(output_folder, filename + ".html"))
        if save_final:
            py.offline.plot(fig, filename=os.path.join(figures, filename + ".html"), auto_open=False)

        try:
            if self.save_png:
                self.logger.info(f"Saving png file for {filename}.")
                fig.write_image(os.path.join(output_folder, filename + ".png"),
                                width=width, height=height, scale=self.scale)
                if save_final:
                    shutil.copy(os.path.join(output_folder, filename + ".png"),
                                os.path.join(figures, filename + ".png"))
            if self.save_eps:
                self.logger.info(f"Saving eps file for {filename}.")
                fig.write_image(os.path.join(output_folder, filename + ".eps"),
                                width=width, height=height)

        except ValueError:
            self.logger.error(f"Value error raised when attempting to save image {filename}.")

    def average_llm_results(self, folder_path, image_column="image", output_csv_path="data"):
        """
        Aggregate CSV files from a folder, compute the average for each numeric column grouped
        by a specified image column, and optionally save the result to a CSV file.

        This method processes all CSV files in the given folder by:
        1. Reading each CSV into a pandas DataFrame.
        2. Replacing numeric values outside the range [0, 100] with NaN.
        3. Combining all DataFrames.
        4. Grouping by the specified image column and calculating mean values for numeric columns.
        5. Saving the aggregated results to a CSV file (if a path is provided).

        Args:
            folder_path (str): Path to the folder containing CSV files.
            image_column (str, optional): The name of the column representing images to group by.
                Defaults to "image".
            output_csv_path (str, optional): Path to save the aggregated CSV file.
                If None, results will not be saved. Defaults to "data".

        Returns:
            pandas.DataFrame: DataFrame containing the average values for each numeric column grouped by
            the image column. Returns None if no CSV files are successfully read.

        Raises:
            ValueError: If the specified image_column is not found in the combined DataFrame.

        """
        # Find all CSV files in the given folder
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        dataframes = []

        # Read and process each CSV file
        for file in csv_files:
            try:
                df = pd.read_csv(file)

                # Ensure numeric values are between 0 and 100; replace invalid values with NaN
                df[df.select_dtypes(include='number').columns] = df.select_dtypes(
                    include='number'
                ).where(lambda x: (x >= 0) & (x <= 100))

                # Append the cleaned DataFrame to the list
                dataframes.append(df)
            except Exception as e:
                # Log and skip files that cannot be read
                self.logger.error(f"Error reading {file}: {e}")

        # If no valid dataframes were read, log error and exit
        if not dataframes:
            self.logger.error("No CSV files were successfully read.")
            return None

        # Combine all DataFrames into one
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Ensure the image column exists before grouping
        if image_column not in combined_df.columns:
            raise ValueError(f"The column '{image_column}' was not found in the CSV files.")

        # Identify numeric columns for aggregation
        numeric_columns = combined_df.select_dtypes(include='number').columns.tolist()

        # Group by image column and calculate mean for numeric columns
        avg_df = combined_df.groupby(image_column)[numeric_columns].mean().reset_index()

        # Save to CSV if an output path is provided
        if output_csv_path:
            avg_df.to_csv(output_csv_path, index=False)

        return avg_df

    def _prepare_merged_df(self, mapping_csv_path, ehmi_csv_path, avg_df):
        """
        Read and merge mapping and eHMI CSV files with an averaged DataFrame.

        This helper method performs the following steps:
        1. Reads a mapping CSV file and an eHMI CSV file.
        2. Renames the 'mean' column in the eHMI file to 'ehmi_mean'.
        3. Cleans and normalises key columns for merging:
           - Converts IDs to strings and strips prefixes/suffixes from image names.
           - Converts relevant text columns to uppercase for consistency.
        4. Merges the averaged DataFrame with mapping and eHMI data.

        Args:
            mapping_csv_path (str): Path to the CSV file containing mapping data.
            ehmi_csv_path (str): Path to the CSV file containing eHMI data.
            avg_df (pandas.DataFrame): DataFrame containing averaged numeric results per image.

        Returns:
            pandas.DataFrame: Merged DataFrame combining avg_df, mapping_df, and ehmi_df.
        """
        # Read the mapping and eHMI CSV files
        mapping_df = pd.read_csv(mapping_csv_path)
        ehmi_df = pd.read_csv(ehmi_csv_path).rename(columns={"mean": "ehmi_mean"})

        # Ensure 'id' and 'image' are comparable as strings
        mapping_df["id"] = mapping_df["id"].astype(str)
        avg_df["image"] = (
            avg_df["image"].astype(str)
            .str.replace("image_", "", regex=False)  # Remove 'image_' prefix
            .str.replace(".jpg", "", regex=False)   # Remove '.jpg' suffix
        )

        # Normalise text columns to uppercase for consistent merging
        mapping_df["text"] = mapping_df["text"].str.upper()
        ehmi_df["eHMI"] = ehmi_df["eHMI"].str.upper()

        # Merge avg_df with mapping_df on image/id
        merged_df = pd.merge(avg_df, mapping_df, left_on="image", right_on="id", how="inner")

        # Merge the result with eHMI data on text/eHMI
        merged_df = pd.merge(merged_df, ehmi_df, left_on="text", right_on="eHMI", how="inner")

        return merged_df

    def plot_ehmi_vs_llm(self, mapping_csv_path, ehmi_csv_path, avg_df, memory_type, save_final=False):
        """
        Plot eHMI mean values against LLM score columns for each text category.

        This method merges averaged LLM scores with mapping and eHMI data,
        then creates an interactive scatter plot comparing eHMI mean values
        to multiple LLM score columns. Each point is labeled by its text category.

        Args:
            mapping_csv_path (str): Path to the CSV file containing mapping data.
            ehmi_csv_path (str): Path to the CSV file containing eHMI mean values.
            avg_df (pandas.DataFrame): Averaged LLM results DataFrame, one row per image.
            memory_type (str): Identifier used in the saved plot filename.
            save_final (bool, optional): If True, the plot is saved to disk. Defaults to False.

        Returns:
            None: Displays or saves the generated plot.
        """
        # Merge averaged LLM scores with mapping and eHMI data
        merged_df = self._prepare_merged_df(mapping_csv_path, ehmi_csv_path, avg_df)

        # Identify numeric columns from avg_df (excluding the 'image' column)
        numeric_columns = [col for col in avg_df.columns if col != "image"]

        # Create an empty Plotly figure
        fig = go.Figure()

        # Add one scatter trace per numeric LLM score column
        for col in numeric_columns:
            fig.add_trace(go.Scatter(
                x=merged_df["ehmi_mean"],  # x-axis: eHMI mean values
                y=merged_df[col],          # y-axis: LLM score
                mode='markers',            # scatter plot with markers
                name=col,                  # legend entry
                text=merged_df["text"]     # hover text
            ))

        # Update layout styling for titles, fonts, and axes
        fig.update_layout(
            title=dict(
                text="",
                font=dict(
                    family=font_family,
                    size=font_size
                )
            ),
            font=dict(  # Global font settings (affects legend, hover labels, etc.)
                family=font_family,
                size=font_size
            ),
            xaxis=dict(
                title=dict(
                    text="eHMI Mean",
                    font=dict(
                        family=font_family,
                        size=font_size
                    )
                ),
                tickfont=dict(
                    family=font_family,
                    size=font_size
                )
            ),
            yaxis=dict(
                title=dict(
                    text="LLM Score",
                    font=dict(
                        family=font_family,
                        size=font_size
                    )
                ),
                tickfont=dict(
                    family=font_family,
                    size=font_size
                )
            ),
            legend=dict(
                title=dict(
                    text="LLM Score Column",
                    font=dict(
                        family=font_family,
                        size=font_size
                    )
                ),
                font=dict(
                    family=font_family,
                    size=font_size
                )
            )
        )

        # Save the plot to file if requested
        self.save_plotly_figure(fig, f"merged_{memory_type}", save_final=save_final)

    def plot_individual_ehmi_vs_llm(self, mapping_csv_path, ehmi_csv_path, avg_df, memory_type, save_final=False):
        """
        Create individual scatter plots comparing eHMI mean values with each LLM score column.

        This method:
        1. Reads mapping and eHMI CSV files.
        2. Cleans and normalises IDs, image names, and text fields for consistent merging.
        3. Merges averaged LLM scores with mapping and eHMI data.
        4. Generates one scatter plot per numeric LLM score column, plotting:
           - X-axis: eHMI mean values (participant responses)
           - Y-axis: LLM scores
        5. Saves each plot to disk if requested.

        Args:
            mapping_csv_path (str): Path to the CSV file containing mapping data.
            ehmi_csv_path (str): Path to the CSV file containing eHMI mean values.
            avg_df (pandas.DataFrame): DataFrame of averaged LLM scores, one row per image.
            memory_type (str): Identifier used in saved plot filenames.
            save_final (bool, optional): If True, saves each generated plot to disk. Defaults to False.

        Returns:
            None: Plots are displayed or saved; no DataFrame is returned.
        """
        # Load mapping data and eHMI data
        mapping_df = pd.read_csv(mapping_csv_path)
        ehmi_df = pd.read_csv(ehmi_csv_path).rename(columns={"mean": "ehmi_mean"})

        # Ensure IDs are strings for consistent merging
        mapping_df["id"] = mapping_df["id"].astype(str)

        # Clean image column in avg_df by removing 'image_' prefix and '.jpg' suffix
        avg_df["image"] = (
            avg_df["image"].astype(str)
            .str.replace("image_", "", regex=False)
            .str.replace(".jpg", "", regex=False)
            .str.strip()
        )

        # Normalise text columns to uppercase for consistent merging
        mapping_df["text"] = mapping_df["text"].str.upper()
        ehmi_df["eHMI"] = ehmi_df["eHMI"].str.upper()

        # Merge avg_df with mapping_df and then with ehmi_df
        merged_df = pd.merge(avg_df, mapping_df, left_on="image", right_on="id", how="inner")
        merged_df = pd.merge(merged_df, ehmi_df, left_on="text", right_on="eHMI", how="inner")

        # Identify numeric score columns (excluding image name)
        numeric_columns = [col for col in avg_df.columns if col != "image"]

        # Create one scatter plot per numeric LLM score column
        for col in numeric_columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=merged_df["ehmi_mean"],                  # X-axis: eHMI participant means
                y=merged_df[col],                          # Y-axis: LLM scores
                mode='markers',                            # Scatter markers
                name=self.RENAME_MAP.get(col, col),        # Legend label (renamed if in RENAME_MAP)
                text=merged_df["text"],                    # Hover text
                marker=dict(color='red', size=14)          # Marker style
            ))

            # Customise plot layout
            fig.update_layout(
                title=dict(
                    text="",
                    font=dict(family=font_family, size=font_size + 34)
                ),
                font=dict(family=font_family, size=font_size + 34),
                xaxis=dict(
                    title=dict(
                        text="Mean response from the participants",
                        font=dict(family=font_family, size=font_size + 34)
                    ),
                    tickfont=dict(family=font_family, size=font_size + 34),
                    range=[0, 100]
                ),
                yaxis=dict(
                    title=dict(
                        text=self.RENAME_MAP.get(col, col),
                        font=dict(family=font_family, size=font_size + 34)
                    ),
                    tickfont=dict(family=font_family, size=font_size + 34),
                    range=[0, 100]
                ),
                legend=dict(
                    font=dict(family=font_family, size=font_size + 34)
                )
            )

            # Save each plot to disk if requested
            self.save_plotly_figure(fig, f"scatter_plot_{col}_{memory_type}", save_final=save_final)

    def plot_spearman_correlation(self, mapping_csv_path, ehmi_csv_path, avg_df, memory_type, save_final=False):
        """
        Generate and save a Spearman correlation heatmap between selected features.

        This method:
        1. Merges mapping, eHMI, and averaged LLM score data into a single DataFrame.
        2. Encodes language codes as numeric values ('en' → 0, 'es' → 1).
        3. Selects relevant numeric columns for correlation analysis.
        4. Removes columns with all NaN values or constant values.
        5. Computes Spearman correlations between features.
        6. Renames columns for improved readability in the heatmap.
        7. Generates and saves a Plotly heatmap visualisation.

        Args:
            mapping_csv_path (str): Path to the CSV file containing mapping data.
            ehmi_csv_path (str): Path to the CSV file containing eHMI data.
            avg_df (pandas.DataFrame): Averaged LLM score DataFrame.
            memory_type (str): Identifier appended to the output filename.
            save_final (bool, optional): Whether to save the final figure to disk. Defaults to False.

        Returns:
            None: Saves the generated figure and does not return a DataFrame.
        """
        # Step 1: Merge input DataFrames
        df = self._prepare_merged_df(mapping_csv_path, ehmi_csv_path, avg_df)

        # Step 2: Encode 'lang' column ('en' → 0, 'es' → 1)
        df['lang_encoded'] = df['lang'].map({'en': 0, 'es': 1})

        # Step 3: Select relevant numeric columns for analysis
        selected_columns = [
            'bakllava', 'gpt-4o', 'deepseek-vl2', 'gemma3:12b', 'gemma3:27b', 'granite3.2-vision',
            'llava:13b', 'llava:34b', 'llava-llama3', 'llava-phi3', 'llama3.2-vision', 'moondream',
            'minicpm-v', 'cross', 'wait', 'egocentric', 'allocentric', 'med', 'ehmi_mean', 'lang_encoded'
        ]
        df_selected = df[selected_columns]

        # Step 4: Remove columns with only NaN values
        dropped_cols = df_selected.columns[df_selected.isna().all()].tolist()
        if dropped_cols:
            logger.info("Dropped columns with all NaN values: %s", dropped_cols)
            df_selected = df_selected.drop(columns=dropped_cols)

        # Step 5: Remove columns with constant values across all rows
        constant_cols = [col for col in df_selected.columns if df_selected[col].nunique(dropna=False) <= 1]
        if constant_cols:
            logger.info("Dropped constant-value columns: %s", constant_cols)
            df_selected = df_selected.drop(columns=constant_cols)

        # Step 6: Compute Spearman correlation matrix
        corr_matrix = df_selected.corr(method='spearman')

        # Step 7: Rename columns for readability in the heatmap
        rename_map = self.RENAME_MAP
        existing_rename_map = {k: v for k, v in rename_map.items() if k in corr_matrix.columns}
        corr_matrix = corr_matrix.rename(index=existing_rename_map, columns=existing_rename_map)

        # Step 8: Generate the heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",  # Display correlation values # pyright: ignore[reportArgumentType]
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title='',
            aspect="auto",
        )

        # Step 9: Update layout for consistent styling
        fig.update_layout(
            xaxis_title="",
            yaxis_title="",
            width=1600,
            height=900,
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=0, b=0),  # Remove margins
            font=dict(family=font_family, size=font_size),
            xaxis=dict(
                tickfont=dict(family=font_family, size=font_size),
                tickangle=-45
            ),
            yaxis=dict(
                tickfont=dict(family=font_family, size=font_size)
            )
        )

        # Step 10: Save the heatmap
        self.save_plotly_figure(
            fig,
            f"spearman_correlation_matrix_{memory_type}",
            width=1600,
            height=900,
            save_final=save_final
        )

    def compute_stats_with_text(self, main_csv_path, mapping_csv_path):
        """
        Compute and log statistical summaries for LLM score columns, including associated image
        and text metadata.

        This method:
        1. Reads a main CSV file containing image-based LLM scores and a mapping CSV with
           `id` → `text` mappings.
        2. Extracts `id` values from the `image` filename field (formatted as 'image_{i}').
        3. Merges text descriptions into the main DataFrame.
        4. For each predefined LLM score column:
           - Computes mean, standard deviation, and median.
           - Identifies maximum and minimum values along with corresponding images and texts.
        5. Logs all results in a structured format.

        Args:
            main_csv_path (str): Path to the main CSV file containing image names and LLM scores.
            mapping_csv_path (str): Path to the CSV file mapping `id` to `text` labels.

        Returns:
            None: Logs the statistics; does not return a DataFrame or value.
        """
        # Step 1: Read main and mapping CSVs
        df = pd.read_csv(main_csv_path)
        mapping = pd.read_csv(mapping_csv_path)

        # Step 2: Extract numeric ID from image filename pattern 'image_{i}'
        df['id'] = df['image'].apply(
            lambda x: int(re.search(r'image_(\d+)', x).group(1)) if pd.notnull(x) else None  # type: ignore
        )

        # Step 3: Merge mapping text onto main DataFrame
        df = df.merge(mapping, on='id', how='left')

        # Step 4: Predefined LLM score columns to analyze
        columns = [
            "gpt-4o", "minicpm-v", "llava:13b", "llava:34b", "llava-llama3",
            "llama3.2-vision", "moondream", "bakllava", "granite3.2-vision",
            "llava-phi3", "gemma3:12b", "gemma3:27b", "deepseek-vl2"
        ]

        # Step 5: Iterate through each score column and compute statistics
        for col in columns:
            max_val = df[col].max()
            min_val = df[col].min()

            # Identify rows with max and min values
            max_rows = df[df[col] == max_val][['image', 'text']]
            min_rows = df[df[col] == min_val][['image', 'text']]

            # Format max and min metadata for logging
            max_images = ', '.join(max_rows['image'].tolist())
            max_texts = ', '.join(max_rows['text'].tolist())
            min_images = ', '.join(min_rows['image'].tolist())
            min_texts = ', '.join(min_rows['text'].tolist())

            # Log computed statistics
            logger.info(f"\n==== {col} ====")
            logger.info(f"Mean:   {df[col].mean():.2f}")
            logger.info(f"Std:    {df[col].std():.2f}")
            logger.info(f"Median: {df[col].median():.2f}")

            logger.info(f"Max value: {max_val}")
            logger.info(f"  Images: {max_images}")
            logger.info(f"  Texts:  {max_texts}")

            logger.info(f"Min value: {min_val}")
            logger.info(f"  Images: {min_images}")
            logger.info(f"  Texts:  {min_texts}")


# Example usage
if __name__ == "__main__":
    # Create an Analysis instance
    analysis = Analysis()

    # Loop through configurations (in this case, only "with_memory")
    for memory_type in ["with_memory"]:
        # If configured, process CSV files before analysis
        if common.get_configs("process_csv_files"):
            client.process_csv_files()

        # Define the folder containing pre-analysed CSV outputs
        folder_path = os.path.join(output_path, memory_type, "analysed")

        # Skip processing if the folder doesn't exist or is empty
        if not os.path.isdir(folder_path) or not os.listdir(folder_path):
            analysis.logger.error(f"Skipping {memory_type}: folder is missing or empty.")
            continue

        analysis.logger.info(f"Processing: {memory_type}")

        # Step 1: Compute average LLM results from all CSVs in the folder
        avg_df = analysis.average_llm_results(
            folder_path=folder_path,
            output_csv_path=os.path.join(output_path, f"avg_{memory_type}.csv")
        )

        # Step 2: Create an aggregate scatter plot (eHMI mean vs multiple LLM score columns)
        analysis.plot_ehmi_vs_llm(
            mapping_csv_path=os.path.join(crowdsourced_data, "mapping.csv"),
            ehmi_csv_path=os.path.join(crowdsourced_data, "ehmis.csv"),
            avg_df=avg_df,
            memory_type=memory_type,
            save_final=True
        )

        # Step 3: Create individual scatter plots (one per LLM score column)
        figures = analysis.plot_individual_ehmi_vs_llm(
            mapping_csv_path=os.path.join(crowdsourced_data, "mapping.csv"),
            ehmi_csv_path=os.path.join(crowdsourced_data, "ehmis.csv"),
            avg_df=avg_df,
            memory_type=memory_type,
            save_final=True
        )

        # Step 4: Generate Spearman correlation heatmap between features
        analysis.plot_spearman_correlation(
            mapping_csv_path=os.path.join(crowdsourced_data, "mapping.csv"),
            ehmi_csv_path=os.path.join(crowdsourced_data, "ehmis.csv"),
            avg_df=avg_df,
            memory_type=memory_type,
            save_final=True
        )

        # Step 5: Compute and log statistics for each LLM score column, with associated text labels
        analysis.compute_stats_with_text(
            os.path.join(output_path, f"avg_{memory_type}.csv"),
            os.path.join(crowdsourced_data, "mapping.csv")
        )
