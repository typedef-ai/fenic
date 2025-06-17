"""Semantic joins example using fenic.

This example demonstrates how to perform LLM-powered semantic joins that use
natural language reasoning to match data across different DataFrames.
"""

import fenic as fc


def main():
    """Demonstrate semantic join capabilities using LLM reasoning."""
    # Configure session with language models (no embeddings needed)
    config = fc.SessionConfig(
        app_name="semantic_joins",
        semantic=fc.SemanticConfig(
            language_models={
                "mini": fc.OpenAIModelConfig(
                    model_name="gpt-4o-mini",
                    rpm=500,
                    tpm=200_000,
                )
            }
        ),
    )

    # Create session
    session = fc.Session.get_or_create(config)

    print("Semantic Joins Example")
    print("=" * 40)
    print("Demonstrating LLM-powered reasoning joins")
    print()

    # Sample user profiles data
    users_data = [
        {
            "user_id": "user_001",
            "name": "Sarah",
            "interests": "I love cooking Italian food and trying new pasta recipes"
        },
        {
            "user_id": "user_002",
            "name": "Mike",
            "interests": "I enjoy working on cars and fixing engines in my spare time"
        },
        {
            "user_id": "user_003",
            "name": "Emily",
            "interests": "Gardening is my passion, especially growing vegetables and flowers"
        },
        {
            "user_id": "user_004",
            "name": "David",
            "interests": "I'm interested in learning about car maintenance and automotive repair"
        }
    ]

    # Sample content/articles data
    articles_data = [
        {
            "article_id": "art_001",
            "title": "Cooking Pasta Recipes",
            "description": "Delicious pasta recipes including spaghetti carbonara and fettuccine alfredo"
        },
        {
            "article_id": "art_002",
            "title": "Car Engine Maintenance",
            "description": "Essential guide to automobile engine care and troubleshooting"
        },
        {
            "article_id": "art_003",
            "title": "Gardening for Beginners",
            "description": "Start your garden with basic techniques for growing vegetables and flowers"
        },
        {
            "article_id": "art_004",
            "title": "Advanced Automotive Repair",
            "description": "Comprehensive automotive repair instructions for experienced mechanics"
        }
    ]

    # Create DataFrames
    users_df = session.create_dataframe(users_data)
    articles_df = session.create_dataframe(articles_data)

    print("User Profiles:")
    users_df.select("name", "interests").show()
    print()

    print("Available Articles:")
    articles_df.select("title", "description").show()
    print()

    # Step 1: Semantic join to match users with relevant articles
    print("Step 1: Matching users to relevant articles using semantic reasoning...")
    print("-" * 70)

    # Use semantic join to match users with articles based on their interests
    user_article_matches = users_df.semantic.join(
        articles_df,
        join_instruction="A person with interests '{interests:left}' would be interested in reading about '{description:right}'"
    )

    print("User-Article Matches:")
    user_article_matches.select(
        "name",
        "interests",
        "title",
        "description"
    ).show()
    print()

    # Step 2: Product recommendation system using semantic joins
    print("Step 2: Product recommendation system...")
    print("-" * 50)

    # Sample customer purchase history
    purchases_data = [
        {
            "customer_id": "cust_001",
            "customer_name": "Alice",
            "purchased_product": "Professional DSLR Camera"
        },
        {
            "customer_id": "cust_002",
            "customer_name": "Bob",
            "purchased_product": "Gaming Laptop"
        },
        {
            "customer_id": "cust_003",
            "customer_name": "Carol",
            "purchased_product": "Yoga Mat"
        },
        {
            "customer_id": "cust_004",
            "customer_name": "Dan",
            "purchased_product": "Coffee Maker"
        }
    ]

    # Sample product catalog for recommendations
    products_data = [
        {
            "product_id": "prod_001",
            "product_name": "Camera Lens Kit",
            "category": "Photography"
        },
        {
            "product_id": "prod_002",
            "product_name": "Tripod Stand",
            "category": "Photography"
        },
        {
            "product_id": "prod_003",
            "product_name": "Gaming Mouse",
            "category": "Gaming"
        },
        {
            "product_id": "prod_004",
            "product_name": "Mechanical Keyboard",
            "category": "Gaming"
        },
        {
            "product_id": "prod_005",
            "product_name": "Yoga Blocks",
            "category": "Fitness"
        },
        {
            "product_id": "prod_006",
            "product_name": "Exercise Resistance Bands",
            "category": "Fitness"
        },
        {
            "product_id": "prod_007",
            "product_name": "Coffee Beans Premium Blend",
            "category": "Food & Beverage"
        },
        {
            "product_id": "prod_008",
            "product_name": "French Press",
            "category": "Food & Beverage"
        }
    ]

    # Create DataFrames
    purchases_df = session.create_dataframe(purchases_data)
    products_df = session.create_dataframe(products_data)

    print("Customer Purchase History:")
    purchases_df.select("customer_name", "purchased_product").show()
    print()

    print("Available Products for Recommendation:")
    products_df.select("product_name", "category").show()
    print()

    # Use semantic join for product recommendations
    recommendations = purchases_df.semantic.join(
        products_df,
        join_instruction="A customer who bought '{purchased_product:left}' would also be interested in '{product_name:right}'"
    )

    print("Product Recommendations:")
    recommendations.select(
        "customer_name",
        "purchased_product",
        "product_name",
        "category"
    ).show()
    print()

    # Clean up
    session.stop()
    print("Session complete!")


if __name__ == "__main__":
    main()
