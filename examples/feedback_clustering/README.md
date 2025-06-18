# Customer Feedback Clustering & Analysis with Fenic

[View in Github](https://github.com/typedef-ai/fenic/blob/main/examples/feedback_clustering/README.md)

This example demonstrates how to use Fenic's `semantic.group_by()` and `semantic.reduce()` to automatically cluster customer feedback into themes and generate intelligent summaries for each discovered category.

## Overview

Customer feedback analysis is a critical business process that traditionally requires manual categorization and analysis. This example shows how semantic clustering can automatically:

- **Discover hidden themes** in unstructured feedback without predefined categories
- **Group similar feedback** based on semantic meaning rather than keywords
- **Generate actionable insights** for each theme using AI-powered summarization
- **Prioritize issues** based on sentiment and frequency

## Key Features Demonstrated

- **Semantic Clustering**: Using `semantic.group_by()` for embedding-based clustering
- **AI Summarization**: Using `semantic.reduce()` for intelligent theme analysis
- **Automatic Theme Discovery**: No manual categorization required
- **Sentiment Analysis**: Understanding positive vs negative feedback patterns
- **Business Intelligence**: Actionable insights for product teams

## How It Works

### Step 1: Data Preparation

Load customer feedback with ratings and metadata:

```python
feedback_data = [
    {
        "feedback_id": "fb_001",
        "customer_name": "Alice Johnson",
        "feedback": "The mobile app crashes every time I try to upload a photo. Very frustrating!",
        "rating": 1,
        "timestamp": "2024-01-15"
    },
    # ... more feedback
]
```

### Step 2: Embedding Creation

Generate semantic embeddings from feedback text:

```python
feedback_with_embeddings = feedback_df.select(
    "*",
    fc.semantic.embed(fc.col("feedback")).alias("feedback_embeddings")
)
```

### Step 3: Semantic Clustering & Summarization

Use both operations together in a single aggregation:

```python
feedback_clusters = feedback_with_embeddings.semantic.group_by(
    fc.col("feedback_embeddings"),
    4  # Number of clusters to discover
).agg(
    fc.count("*").alias("feedback_count"),
    fc.avg("rating").alias("avg_rating"),
    fc.collect_list("customer_name").alias("customer_names"),
    fc.semantic.reduce(
        "Analyze this cluster of customer feedback and provide a concise summary of the main theme, common issues, and sentiment. Feedback: {feedback}"
    ).alias("theme_summary")
)
```

## Sample Results

The system automatically discovered these themes from 12 feedback entries:

### Cluster 0: Positive Features & Support (4.75★)

- **Theme**: Praise for specific features and excellent customer support
- **Key Points**: Dark mode feature, helpful support team, effective search functionality
- **Sentiment**: Predominantly positive with some feature enhancement requests

### Cluster 1: UI/UX Design Issues (2.0★)

- **Theme**: Design consistency and professional appearance concerns
- **Key Points**: Inconsistent button layouts across screens
- **Sentiment**: Negative due to unprofessional user experience

### Cluster 2: Technical Performance Problems (1.75★)

- **Theme**: Critical technical issues affecting core functionality
- **Key Points**: App crashes, slow loading times, frequent freezes
- **Sentiment**: Very negative with high frustration levels

### Cluster 3: Usability & Feature Gaps (2.0★)

- **Theme**: Process complexity and missing functionality
- **Key Points**: Confusing checkout, need for offline mode
- **Sentiment**: Negative about functionality limitations

## Value

### **Automated Insights**

- Identifies themes without manual categorization
- Provides consistent analysis across all feedback
- Scales to thousands of feedback entries

### **Actionable Intelligence**

- **Priority 1**: Fix technical crashes and performance (Cluster 2)
- **Priority 2**: Improve design consistency (Cluster 1)
- **Priority 3**: Simplify user workflows (Cluster 3)
- **Maintain**: Continue excellent support and features (Cluster 0)

### **Resource Optimization**

- Reduces manual analysis time from hours to minutes
- Enables real-time feedback monitoring
- Focuses development efforts on highest-impact issues

## Technical Architecture

### Session Configuration

```python
config = fc.SessionConfig(
    app_name="feedback_clustering",
    semantic=fc.SemanticConfig(
        language_models={
            "mini": fc.OpenAIModelConfig(
                model_name="gpt-4o-mini",
                rpm=500,
                tpm=200_000,
            )
        },
        embedding_models={
            "small": fc.OpenAIModelConfig(
                model_name="text-embedding-3-small",
                rpm=3000,
                tpm=1_000_000
            )
        }
    ),
)
```

### Key Operations

**`semantic.group_by(embedding_column, num_clusters)`**

- Uses K-means clustering on embedding vectors
- Groups semantically similar feedback together
- Assigns `_cluster_id` to each group

**`semantic.reduce(instruction)`**

- Aggregation function that summarizes multiple texts
- Uses LLM to analyze and synthesize insights
- Generates human-readable theme descriptions

## Usage

```bash
# Ensure you have OpenAI API key configured
export OPENAI_API_KEY="your-api-key"

# Run the feedback clustering analysis
python feedback_clustering.py
```

## Expected Output

The script shows:

1. **Raw Feedback Data**: Customer names, feedback text, and ratings
2. **Clustering Progress**: Embedding generation and clustering status
3. **Theme Analysis**: Detailed summaries for each discovered cluster
4. **Business Insights**: Actionable themes ranked by priority

## Use Cases

### **Product Development**

- Identify most requested features
- Understand user pain points
- Prioritize bug fixes and improvements

### **Customer Success**

- Monitor satisfaction trends
- Identify at-risk customer segments
- Improve support processes

### **Marketing Intelligence**

- Understand customer sentiment
- Identify product strengths for messaging
- Track competitive advantages

## Learning Outcomes

This example teaches:

- How to combine embedding-based clustering with AI summarization
- When to use semantic operations for business intelligence
- Patterns for automated text analysis and insight generation
- Integration of multiple semantic operations in data pipelines
