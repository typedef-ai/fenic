# Semantic Joins with Fenic

[View in Github](https://github.com/typedef-ai/fenic/blob/main/examples/semantic_joins/README.md)

This example demonstrates how to use Fenic's semantic joins to perform LLM-powered data matching based on natural language reasoning rather than exact equality or similarity scores.

## Overview

Semantic joins enable you to join DataFrames using natural language predicates that are evaluated by language models. Unlike traditional joins that require exact matches or embedding-based similarity joins, semantic joins can understand complex relationships and make intelligent connections based on meaning and context.

This example showcases two practical use cases:

- **Content Recommendation**: Matching user interests to relevant articles
- **Product Recommendations**: Suggesting complementary products based on purchase history

## Key Features Demonstrated

- **Natural Language Predicates**: Using human-readable join conditions
- **LLM-Powered Reasoning**: Leveraging GPT models for intelligent matching
- **Cross-Domain Understanding**: Connecting concepts across different contexts
- **Zero-Shot Matching**: No training data or examples required

## How Semantic Joins Work

### Basic Syntax

```python
left_df.semantic.join(
    right_df,
    predicate="Natural language predicate with {{left_on}} and {{right_on}}",
    left_on=col("left"),
    right_on=col("right")
)
```

### Join Jinja Predicate Format

- Jinja template variables must be `left_on` (join key on the left dataframe) and `right_on`(join key on the right dataframe)
- Written as a boolean predicate that the LLM evaluates as True/False
- Should be clear and unambiguous for consistent results

## Example 1: Content Recommendation

### Data Setup

**User Profiles:**

- Sarah: "I love cooking Italian food and trying new pasta recipes"
- Mike: "I enjoy working on cars and fixing engines in my spare time"
- Emily: "Gardening is my passion, especially growing vegetables and flowers"
- David: "I'm interested in learning about car maintenance and automotive repair"

**Articles:**

- Cooking Pasta Recipes
- Car Engine Maintenance
- Gardening for Beginners
- Advanced Automotive Repair

### Semantic Join Implementation

```python
users_df.semantic.join(
    articles_df,
    predicate=(
        "A person with interests '{{left_on}}' would be interested in reading about '{{right_on}}'"
    ),
    left_on=fc.col("interests"),
    right_on=fc.col("description")
)
```

### Matching Results

- Sarah → Cooking Pasta Recipes ✅
- Mike → Car Engine Maintenance + Advanced Automotive Repair ✅
- Emily → Gardening for Beginners ✅
- David → Car Engine Maintenance + Advanced Automotive Repair ✅

## Example 2: Product Recommendations

### Sample Data

**Customer Purchases:**

- Alice: Professional DSLR Camera
- Bob: Gaming Laptop
- Carol: Yoga Mat
- Dan: Coffee Maker

**Product Catalog:**

- Camera Lens Kit, Tripod Stand (Photography)
- Gaming Mouse, Mechanical Keyboard (Gaming)
- Yoga Blocks, Exercise Resistance Bands (Fitness)
- Coffee Beans, French Press (Food & Beverage)

### Recommendation Logic

```python
purchases_df.semantic.join(
    products_df,
    predicate=(
        "A customer who bought '{{left_on}}' would also be interested in '{{right_on}}'"
    ),
    left_on=fc.col("purchased_product"),
    right_on=fc.col("product_name")
)
```

### Recommendation Results

- Alice (DSLR Camera) → Camera Lens Kit + Tripod Stand ✅
- Bob (Gaming Laptop) → Gaming Mouse + Mechanical Keyboard ✅
- Carol (Yoga Mat) → Yoga Blocks + Exercise Resistance Bands ✅
- Dan (Coffee Maker) → Coffee Beans + French Press ✅

## Technical Details

### Session Configuration

```python
config = fc.SessionConfig(
    app_name="semantic_joins",
    semantic=fc.SemanticConfig(
        language_models={
            "mini": fc.OpenAILanguageModel(
                model_name="gpt-4o-mini",
                rpm=500,
                tpm=200_000,
            )
        }
    ),
)
```

### Performance Characteristics

- **Complexity**: O(m × n) where m and n are the sizes of the DataFrames
- **LLM Calls**: One API call per potential row pair
- **Rate Limiting**: Respects RPM/TPM limits configured in session
- **Batching**: Efficiently batches requests to optimize API usage

## When to Use Semantic Joins

### **Ideal Use Cases:**

- **Content personalization** and recommendation systems
- **Product cross-selling** and upselling
- **Skill-job matching** in recruitment
- **Entity resolution** across different data sources
- **Question-answer pairing** for knowledge bases
- **Customer-service matching** based on needs

### **Advantages:**

- No training data required (zero-shot)
- Handles complex reasoning and context
- Understands domain-specific relationships
- Works with natural language descriptions
- Flexible and interpretable join conditions

### **Considerations:**

- Higher latency than traditional joins
- API costs for LLM usage
- Rate limiting for large datasets
- Best for moderate-sized datasets (hundreds to low thousands of rows)

## Usage

```bash
# Ensure you have OpenAI API key configured
export OPENAI_API_KEY="your-api-key"

# Run the semantic joins example
python semantic_joins.py
```

## Expected Output

The script demonstrates both use cases with clear before/after data views:

1. **User-Article Matching**: Shows how semantic understanding connects user interests to relevant content
2. **Product Recommendations**: Demonstrates intelligent product relationship detection for cross-selling

## Learning Outcomes

This example teaches:

- How to construct effective natural language join predicates
- When semantic joins are preferable to traditional or similarity-based joins
- Practical applications in recommendation systems and personalization
- Understanding the trade-offs between accuracy, performance, and cost

Perfect for understanding how to leverage LLM reasoning capabilities for intelligent data joining scenarios that go beyond simple keyword matching or embedding similarity.
