# 🧠 Labeling for Medium Articles with `fenic`

This project demonstrates how to use [**fenic**](https://pypi.org/project/fenic/) and semantic operators on Medium articles to generate content-based labels. These labels are derived using LLMs, embeddings, and heuristic rules — enabling downstream tasks like content discovery and personalized recommendations.

These labels can enrich existing article metadata and guide **exploratory search**, **ranking**, or **recommendation** systems.

---

## 🔧 Setup

### Python Version Requirement

**fenic requires Python 3.10-3.12**. Make sure you're using a compatible Python version before proceeding with installation.

```bash
# Check your Python version
python --version
```

### Dependencies

Install the project dependencies using your preferred package manager:

```bash
# Using pip
pip install -e .

# Or install individual packages
pip install fenic rapidfuzz scikit-learn matplotlib python-dotenv ipykernel
```

### Environment Configuration

Create a `.env` file in the project root with the following API keys:

```env
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

**Required API Keys:**

- **OpenAI API key** - For embedding generation
- **Gemini (Google AI) API key** - For LLM inference

> ⚠️ **Important**: Make sure to keep your `.env` file secure and never commit it to version control.

---

## 📈 Execution Plan

Run the following notebooks in order:

1. `clean.ipynb` – Preprocessing and basic cleaning of article text
2. `is_on_topic.ipynb` – Filter out off-topic articles
3. `extract_features.ipynb` – Generate embeddings and prepare inputs for labeling

From here, the following scripts can be run **in parallel** (they depend only on `extract_features.ipynb`):

4. `technical_terms.ipynb` – Identify domain-specific terms
5. `technical_complexity.ipynb` – Assess the complexity of writing
6. `narrative_intent.ipynb` – Determine the rhetorical purpose of the article
7. `topic_modeling.ipynb` – Group articles into latent topic clusters

⚠️ Due to **Gemini rate limits**, you may prefer to run scripts 4–7 **serially**.

8. `finalize.ipynb` – Join all derived columns by `url` and create the final annotated dataset

---

## 🏷️ Labels Derived

Here are the labels extracted from each analysis stage:

| Label Type               | Description                                                             |
| ------------------------ | ----------------------------------------------------------------------- |
| **Topic Annotation**     | Embedding-based hierarchical K-means clustering of article content      |
| **Technical Terms**      | Canonicalized domain-specific vocabulary extracted using LLMs           |
| **Technical Complexity** | Estimated readability and depth of technical detail                     |
| **Narrative Intent**     | Classifies whether the article is _Inform_, _Persuade_, _Reflect_, etc. |
| **Is On Topic**          | Whether the article is actually about `artificial-intelligence`         |
| **Has Code**             | Whether or not the article has any code in it                           |

---

## 🚀 Getting Started

### Quick Demo (No Setup Required)

Want to explore the results without recreating all the data? Simply run:

```bash
jupyter notebook demo.ipynb
```

This notebook lets you investigate the **materialized tables** stored in DuckDB without needing API keys or running the full pipeline.

### Full Pipeline Setup

1. Get directory
2. Install dependencies: `pip install -e .`
3. Set up your `.env` file with the required API keys
4. Run the notebooks in the specified order
5. Use the final annotated dataset for your content analysis needs

---

## 📦 Project Structure

The project uses `pyproject.toml` for dependency management and follows modern Python packaging standards. All required dependencies are automatically installed when you run `pip install -e .`.
