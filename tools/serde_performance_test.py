#!/usr/bin/env python3
"""Performance comparison test for CloudPickle vs Proto serialization of logical plans.

This tool generates complex logical plans using various Fenic DataFrame operations
and compares the performance and size characteristics of different serialization methods.
"""

import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

# Add src to path to import fenic
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import fenic as fc
from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core._serde.cloudpickle_serde import CloudPickleSerde
from fenic.core._serde.proto.proto_serde import ProtoSerde
from fenic.core.types import (
    ClassDefinition,
    Paragraph,
)


@dataclass
class SerializationResult:
    """Results from a serialization test."""
    method: str
    serialize_time: float
    deserialize_time: float
    serialized_size: int
    success: bool
    error_message: Optional[str] = None


@dataclass
class TestResults:
    """Complete test results for a logical plan."""
    plan_name: str
    plan_description: str
    cloudpickle: SerializationResult
    proto: SerializationResult


class ComplexLogicalPlanGenerator:
    """Generates complex logical plans for testing serialization performance."""
    
    def __init__(self, session: fc.Session):
        self.session = session
        
    def create_sample_data(self) -> Dict[str, fc.DataFrame]:
        """Create sample datasets for testing."""
        # Customer data
        customers_data = [
            {"customer_id": f"cust_{i:04d}", "customer_name": f"Customer {i}", 
             "email": f"customer{i}@example.com", "age": 20 + (i % 60),
             "location": ["New York", "London", "Tokyo", "Sydney", "Berlin"][i % 5],
             "interests": [
                 "I love technology and gadgets, especially smartphones and laptops",
                 "Passionate about cooking and trying new recipes from different cultures", 
                 "Outdoor enthusiast who enjoys hiking, camping, and nature photography",
                 "Fashion and style blogger interested in sustainable clothing",
                 "Fitness enthusiast focused on strength training and nutrition"
             ][i % 5]}
            for i in range(100)
        ]
        
        # Product data
        products_data = [
            {"product_id": f"prod_{i:04d}", "product_name": f"Product {i}",
             "category": ["Electronics", "Home & Kitchen", "Sports", "Fashion", "Books"][i % 5],
             "price": 10.0 + (i * 5.5), "rating": 3.0 + (i % 3),
             "description": [
                 "Latest smartphone with advanced camera and long battery life",
                 "Premium kitchen appliance for modern cooking enthusiasts",
                 "High-quality sports equipment for outdoor adventures",
                 "Trendy fashion item made from sustainable materials", 
                 "Educational book covering important topics and insights"
             ][i % 5]}
            for i in range(50)
        ]
        
        # Transaction data
        transactions_data = [
            {"transaction_id": f"txn_{i:06d}", 
             "customer_id": f"cust_{(i * 3) % 100:04d}",
             "product_id": f"prod_{(i * 7) % 50:04d}",
             "quantity": 1 + (i % 5), "amount": 15.0 + (i * 2.3),
             "timestamp": f"2024-{1 + (i % 12):02d}-{1 + (i % 28):02d}",
             "payment_method": ["credit_card", "debit_card", "paypal", "apple_pay"][i % 4]}
            for i in range(500)
        ]
        
        # Review data
        reviews_data = [
            {"review_id": f"rev_{i:05d}",
             "product_id": f"prod_{(i * 11) % 50:04d}",
             "customer_id": f"cust_{(i * 13) % 100:04d}",
             "rating": 1 + (i % 5), 
             "review_text": [
                 "Amazing product! Exceeded my expectations in every way. Highly recommend!",
                 "Good quality but could be improved. Worth the price overall.",
                 "Decent product. Does what it's supposed to do, nothing special.",
                 "Not satisfied with this purchase. Quality issues and poor customer service.",
                 "Excellent value for money! Will definitely buy again."
             ][i % 5],
             "helpful_votes": i % 20}
            for i in range(200)
        ]
        
        return {
            "customers": self.session.create_dataframe(customers_data),
            "products": self.session.create_dataframe(products_data), 
            "transactions": self.session.create_dataframe(transactions_data),
            "reviews": self.session.create_dataframe(reviews_data)
        }
    
    def create_simple_plan(self, dfs: Dict[str, fc.DataFrame]) -> LogicalPlan:
        """Create a simple logical plan with basic operations."""
        result = (dfs["customers"]
                 .select("customer_id", "customer_name", "age", "location")
                 .filter(fc.col("age") > 25)
                 .limit(10))
        return result._logical_plan
    
    def create_medium_plan(self, dfs: Dict[str, fc.DataFrame]) -> LogicalPlan:
        """Create a medium complexity logical plan with joins and aggregations."""
        result = (dfs["transactions"]
                 .join(dfs["customers"], on="customer_id", how="inner")
                 .join(dfs["products"], on="product_id", how="inner")
                 .group_by("location", "category")
                 .agg(
                     fc.sum("amount").alias("total_amount"),
                     fc.count("transaction_id").alias("transaction_count"),
                     fc.avg("rating").alias("avg_rating")
                 )
                 .filter(fc.col("total_amount") > 100)
                 .order_by("total_amount", ascending=False))
        return result._logical_plan
    
    def create_complex_plan(self, dfs: Dict[str, fc.DataFrame]) -> LogicalPlan:
        """Create a complex logical plan with semantic operations (avoiding database connections)."""
        
        # Define a response format for extraction
        class CustomerInsights(BaseModel):
            personality_type: str = Field(description="Customer personality type based on interests")
            likely_purchases: List[str] = Field(description="List of product categories they might buy")
            marketing_segment: str = Field(description="Marketing segment classification")
        
        # Complex pipeline with multiple semantic operations (but no semantic.join or extract)
        customer_analysis = (dfs["customers"]
                           .select("customer_id", "customer_name", "interests", "age", "location")
                           .with_column(
                               "interest_category",
                               fc.semantic.classify(
                                   "interests", 
                                   [
                                       ClassDefinition(label="Tech Enthusiast", description="Loves technology and gadgets"),
                                       ClassDefinition(label="Culinary Explorer", description="Passionate about food and cooking"),
                                       ClassDefinition(label="Outdoor Adventurer", description="Enjoys outdoor activities and nature"),
                                       ClassDefinition(label="Fashion Forward", description="Interested in style and trends"),
                                       ClassDefinition(label="Health & Fitness", description="Focused on wellness and fitness")
                                   ]
                               )
                           )
                           .with_column(
                               "interests_summary",
                               fc.semantic.summarize("interests", format=Paragraph(max_words=50))
                           )
                           .with_column(
                               "sentiment_score",
                               fc.semantic.analyze_sentiment("interests")
                           ))
        
        # Join with transaction and review data using regular joins
        enriched_customers = (customer_analysis
                            .join(
                                dfs["transactions"].group_by("customer_id").agg(
                                    fc.sum("amount").alias("total_spent"),
                                    fc.count("transaction_id").alias("purchase_count")
                                ), on="customer_id", how="left"
                            )
                            .join(
                                dfs["reviews"].group_by("customer_id").agg(
                                    fc.avg("rating").alias("avg_review_rating"),
                                    fc.count("review_id").alias("review_count")
                                ), on="customer_id", how="left"
                            ))
        
        # Regular join with products via transactions (no semantic join to avoid DB connections)
        product_data = (enriched_customers
                       .join(dfs["transactions"], on="customer_id", how="inner")
                       .join(dfs["products"], on="product_id", how="inner")
                       .select(
                           "customer_id", "customer_name", "interest_category", "category",
                           "total_spent", "avg_review_rating", "sentiment_score"
                       ))
        
        # Final aggregation and analysis
        result = (product_data
                 .group_by("interest_category", "category")
                 .agg(
                     fc.count("customer_id").alias("customer_count"),
                     fc.avg("total_spent").alias("avg_spending"),
                     fc.avg("avg_review_rating").alias("avg_satisfaction"),
                     fc.collect_list("customer_name").alias("customer_names")
                 )
                 .filter(fc.col("customer_count") >= 2)
                 .order_by("avg_spending", ascending=False))
        
        return result._logical_plan
    
    def create_very_complex_plan(self, dfs: Dict[str, fc.DataFrame]) -> LogicalPlan:
        """Create an extremely complex logical plan with nested operations."""
        
        # Multiple semantic transformations on reviews
        enhanced_reviews = (dfs["reviews"]
                          .select("review_id", "product_id", "customer_id", "review_text")
                          .with_column(
                              "sentiment",
                              fc.semantic.analyze_sentiment("review_text")
                          )
                          .with_column(
                              "sentiment_category",
                              fc.semantic.classify(
                                  "review_text",
                                  ["Positive", "Negative", "Neutral", "Mixed"]
                              )
                          )
                          .with_column(
                              "review_summary",
                              fc.semantic.summarize(
                                  "review_text", 
                                  format=Paragraph(max_words=30)
                              )
                          ))
        
        # Customer analysis with embeddings
        customer_analysis = (dfs["customers"]
                           .select("customer_id", "customer_name", "interests", "age", "location")
                           .with_column(
                               "interest_embedding",
                               fc.semantic.embed("interests")
                           )
                           .with_column(
                               "interest_type",
                               fc.semantic.classify(
                                   "interests",
                                   ["Technology", "Food", "Sports", "Fashion", "Health"]
                               )
                           ))
        
        # Aggregate transaction data
        transaction_summary = (dfs["transactions"]
                             .group_by("customer_id")
                             .agg(
                                 fc.sum("amount").alias("total_spent"),
                                 fc.count("*").alias("transaction_count"),
                                 fc.avg("amount").alias("avg_transaction")
                             ))
        
        # Aggregate review data
        review_summary = (enhanced_reviews
                        .group_by("customer_id")
                        .agg(
                            fc.count("*").alias("review_count"),
                            fc.collect_list("sentiment_category").alias("all_sentiments")
                        ))
        
        # Complex joins and final analysis
        final_result = (customer_analysis
                      .join(transaction_summary, on="customer_id", how="left")
                      .join(review_summary, on="customer_id", how="left")
                      .filter(fc.col("total_spent").is_not_null())
                      .with_column(
                          "customer_value_score",
                          fc.col("total_spent") * fc.col("transaction_count")
                      )
                      .group_by("interest_type", "location")
                      .agg(
                          fc.count("customer_id").alias("customer_count"),
                          fc.avg("customer_value_score").alias("avg_value_score"),
                          fc.sum("total_spent").alias("total_revenue")
                      )
                      .filter(fc.col("customer_count") >= 2)
                      .order_by("avg_value_score", ascending=False)
                      .limit(10))
        
        return final_result._logical_plan


class SerializationTester:
    """Tests serialization performance for different methods."""
    
    def __init__(self):
        self.cloudpickle_serde = CloudPickleSerde()
        self.proto_serde = ProtoSerde()
    
    def test_serialization_method(self, plan: LogicalPlan, method_name: str, serde_impl) -> SerializationResult:
        """Test a single serialization method."""
        try:
            # Test serialization
            start_time = time.perf_counter()
            serialized_data = serde_impl.serialize(plan)
            serialize_time = time.perf_counter() - start_time
            
            # Test deserialization  
            start_time = time.perf_counter()
            _ = serde_impl.deserialize(serialized_data)
            deserialize_time = time.perf_counter() - start_time
            
            return SerializationResult(
                method=method_name,
                serialize_time=serialize_time,
                deserialize_time=deserialize_time,
                serialized_size=len(serialized_data),
                success=True
            )
            
        except Exception as e:
            return SerializationResult(
                method=method_name,
                serialize_time=0.0,
                deserialize_time=0.0,
                serialized_size=0,
                success=False,
                error_message=str(e)
            )
    
    def test_plan(self, plan: LogicalPlan, plan_name: str, plan_description: str) -> TestResults:
        """Test a logical plan with all serialization methods."""
        cloudpickle_result = self.test_serialization_method(plan, "CloudPickle", self.cloudpickle_serde)
        proto_result = self.test_serialization_method(plan, "Proto", self.proto_serde)
        
        return TestResults(
            plan_name=plan_name,
            plan_description=plan_description,
            cloudpickle=cloudpickle_result,
            proto=proto_result
        )


def format_size(size_bytes: int) -> str:
    """Format size in human readable format using standard library."""
    # Python 3.12+ has math.ceil and other improvements, but let's use a simple approach
    # that works across versions
    if size_bytes == 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    unit_index = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    
    return f"{size:.1f} {units[unit_index]}"


def format_time(time_seconds: float) -> str:
    """Format time in human readable format using standard library."""
    if time_seconds >= 60:
        # Use timedelta for longer durations
        td = timedelta(seconds=time_seconds)
        # Format as HH:MM:SS for longer times
        hours, remainder = divmod(int(td.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}.{int((time_seconds % 1) * 1000):03d}"
    elif time_seconds >= 1.0:
        return f"{time_seconds:.3f} s"
    elif time_seconds >= 0.001:
        return f"{time_seconds * 1000:.1f} ms"
    else:
        return f"{time_seconds * 1000000:.1f} μs"


def print_results(results: List[TestResults]):
    """Print formatted test results."""
    print("\n" + "="*100)
    print("SERIALIZATION PERFORMANCE COMPARISON")
    print("="*100)
    
    for result in results:
        print(f"\n📊 {result.plan_name}")
        print(f"   {result.plan_description}")
        print("-" * 80)
        
        # Headers
        print(f"{'Method':<12} {'Status':<8} {'Serialize':<12} {'Deserialize':<12} {'Size':<12} {'Total Time':<12}")
        print("-" * 80)
        
        # CloudPickle results
        cp = result.cloudpickle
        if cp.success:
            total_time = cp.serialize_time + cp.deserialize_time
            print(f"{'CloudPickle':<12} {'✅ OK':<8} {format_time(cp.serialize_time):<12} "
                  f"{format_time(cp.deserialize_time):<12} {format_size(cp.serialized_size):<12} "
                  f"{format_time(total_time):<12}")
        else:
            print(f"{'CloudPickle':<12} {'❌ FAIL':<8} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
            print(f"   Error: {cp.error_message}")
        
        # Proto results
        proto = result.proto
        if proto.success:
            total_time = proto.serialize_time + proto.deserialize_time
            print(f"{'Proto':<12} {'✅ OK':<8} {format_time(proto.serialize_time):<12} "
                  f"{format_time(proto.deserialize_time):<12} {format_size(proto.serialized_size):<12} "
                  f"{format_time(total_time):<12}")
        else:
            print(f"{'Proto':<12} {'❌ FAIL':<8} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
            print(f"   Error: {proto.error_message}")
        
        # Comparison if both succeeded
        if cp.success and proto.success:
            print("\n📈 Comparison:")
            
            # Speed comparison
            cp_total = cp.serialize_time + cp.deserialize_time
            proto_total = proto.serialize_time + proto.deserialize_time
            if cp_total > proto_total:
                speedup = cp_total / proto_total
                print(f"   • Proto is {speedup:.1f}x faster overall")
            else:
                speedup = proto_total / cp_total
                print(f"   • CloudPickle is {speedup:.1f}x faster overall")
            
            # Size comparison
            if cp.serialized_size > proto.serialized_size:
                size_ratio = cp.serialized_size / proto.serialized_size
                savings = (1 - proto.serialized_size / cp.serialized_size) * 100
                print(f"   • Proto is {savings:.1f}% smaller ({size_ratio:.1f}x compression)")
            else:
                size_ratio = proto.serialized_size / cp.serialized_size
                savings = (1 - cp.serialized_size / proto.serialized_size) * 100
                print(f"   • CloudPickle is {savings:.1f}% smaller ({size_ratio:.1f}x compression)")


def main():
    """Main function to run the performance comparison."""
    print("🚀 Starting Fenic Serialization Performance Test")
    print("This test compares CloudPickle vs Proto serialization for complex logical plans.")
    
    # Configure session with semantic capabilities
    config = fc.SessionConfig(
        app_name="serde_performance_test",
        semantic=fc.SemanticConfig(
            language_models={
                "default": fc.OpenAILanguageModel(
                    model_name="gpt-4o-mini",
                    rpm=100,
                    tpm=50_000,
                )
            },
            embedding_models={
                "default": fc.OpenAIEmbeddingModel(
                    model_name="text-embedding-3-small",
                    rpm=1000,
                    tpm=1_000_000,
                )
            }
        ),
    )
    
    session = fc.Session.get_or_create(config)
    
    try:
        # Generate test data and plans
        print("📝 Generating test data and logical plans...")
        generator = ComplexLogicalPlanGenerator(session)
        dfs = generator.create_sample_data()
        
        # Create test plans of varying complexity
        test_plans = [
            (generator.create_simple_plan(dfs), "Simple Plan", 
             "Basic filter and select operations"),
            (generator.create_medium_plan(dfs), "Medium Plan", 
             "Joins, aggregations, and basic transformations"),
            (generator.create_complex_plan(dfs), "Complex Plan", 
             "Semantic operations, classifications, and extractions"),
            (generator.create_very_complex_plan(dfs), "Very Complex Plan", 
             "Advanced semantic operations, clustering, and multi-level joins")
        ]
        
        # Run serialization tests
        print("⚡ Running serialization performance tests...")
        tester = SerializationTester()
        results = []
        
        for i, (plan, name, description) in enumerate(test_plans, 1):
            print(f"   Testing plan {i}/4: {name}")
            result = tester.test_plan(plan, name, description)
            results.append(result)
        
        # Print results
        print_results(results)
        
        print("\n✨ Performance test completed successfully!")
        print(f"   Tested {len(results)} logical plans with 2 serialization methods")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        raise
    finally:
        session.stop()


if __name__ == "__main__":
    main()