#!/usr/bin/env python3
"""News Article Bias Detection.

This script demonstrates how to use fenic's semantic classification capabilities to detect editorial bias and analyze news articles. We'll walk through:

- Language Analysis using `semantic.extract()` to find biased, emotional, or sensationalist language.
- Political Bias Classifcation using `semantic.classify()` grounded in the extracted data.
- News Topic Classification using `semantic.classify()`
- Merging the information together using `semantic.reduce()` to create a 'Media Profile' summary for each analyzed News Source

This is a practical example of how semantic classification can provide insights into media content.

Usage:
    python news_analysis.py
"""

from typing import Optional

from pydantic import BaseModel, Field

import fenic as fc


def main(config: Optional[fc.SessionConfig] = None):
    """Main analysis pipeline for news article bias detection."""
    # Configure session with semantic capabilities
    # Set your `OPENAI_API_KEY` environment variable.
    # Alternatively, you can run the example with an Gemini (`GOOGLE_API_KEY`) model by uncommenting the provided additional model configurations.
    # Using an Anthropic model requires installing fenic with the `anthropic` extra package, and setting the `ANTHROPIC_API_KEY` environment variable
    print("🔧 Configuring fenic session...")
    config = config or fc.SessionConfig(
        app_name="news_analysis",
        semantic=fc.SemanticConfig(
            language_models={
                "openai": fc.OpenAILanguageModel(
                    model_name="gpt-4o-mini",
                    rpm=500,
                    tpm=200_000
                ),
                # "gemini": fc.GoogleDeveloperLanguageModel(
                #     model_name="gemini-2.0-flash",
                #     rpm=500,
                #     tpm=1_000_000
                # ),
                # "anthropic": fc.AnthropicLanguageModel(
                #     model_name="claude-3-5-haiku-latest",
                #     rpm=500,
                #     input_tpm=80_000,
                #     output_tpm=32_000,
                # )
            }
        )
    )

    # Create session
    session = fc.Session.get_or_create(config)

    # Sample news articles - multiple articles per source to show bias patterns
    news_articles = [
        # Global Wire Service (Neutral source, Reuters-style) - 3 articles
        {
            "source": "Global Wire Service",
            "headline": "Federal Reserve Raises Interest Rates by 0.25 Percentage Points",
            "content": "The Federal Reserve announced a quarter-point increase in interest rates Wednesday, bringing the federal funds rate to 5.5%. The decision was unanimous among voting members. Fed Chair Jerome Powell cited persistent inflation concerns and a robust labor market as key factors. The rate hike affects borrowing costs for consumers and businesses. Economic analysts had predicted the move following recent inflation data showing prices remained above the Fed's 2% target."
        },
        {
            "source": "Global Wire Service",
            "headline": "OpenAI Launches GPT-4 Turbo with 128K Context Window",
            "content": "OpenAI today announced GPT-4 Turbo, featuring a 128,000 token context window and updated training data through April 2024. The model offers improved instruction following and reduced likelihood of generating harmful content. Pricing is set at $0.01 per 1K input tokens and $0.03 per 1K output tokens. The release includes enhanced support for JSON mode and function calling. Developer early access begins this week, with general availability planned for December."
        },
        {
            "source": "Global Wire Service",
            "headline": "Climate Summit Reaches Agreement on Fossil Fuel Transition",
            "content": "Delegates at the COP28 climate summit in Dubai reached a consensus agreement calling for a transition away from fossil fuels in energy systems. The deal, approved by nearly 200 countries, marks the first time a COP agreement explicitly mentions fossil fuels. However, the agreement uses the phrase 'transitioning away' rather than 'phasing out,' reflecting compromises necessary to secure broad support. Environmental groups expressed mixed reactions, with some praising the historic mention while others criticized the lack of binding timelines."
        },

        # Progressive Voice (Left-leaning source) - 3 articles
        {
            "source": "Progressive Voice",
            "headline": "Fed's Rate Hike Threatens Working Families as Corporate Profits Soar",
            "content": "Once again, the Federal Reserve has chosen to burden working families with higher borrowing costs while Wall Street celebrates record profits. Wednesday's rate hike to 5.5% will make mortgages, credit cards, and student loans more expensive for millions of Americans already struggling with housing costs. Meanwhile, corporate executives continue awarding themselves massive bonuses. This regressive monetary policy prioritizes the wealthy elite over middle-class families who desperately need relief."
        },
        {
            "source": "Progressive Voice",
            "headline": "Big Tech's AI Surveillance Threatens Democratic Values",
            "content": "OpenAI's latest AI release represents another troubling escalation in Silicon Valley's surveillance capitalism model. These systems hoover up personal data and creative content without meaningful consent from users. Artists, writers, and creators see their work exploited to train AI systems that directly compete with human creativity. Meanwhile, users surrender intimate conversations to corporate servers with little transparency. We need immediate regulation to protect digital rights and prevent tech giants from privatizing human knowledge for profit."
        },
        {
            "source": "Progressive Voice",
            "headline": "Climate Summit's Weak Language Betrays Future Generations",
            "content": "The COP28 agreement represents a devastating failure to confront the climate emergency with the urgency science demands. By choosing vague 'transition' language over concrete 'phase out' commitments, world leaders have once again capitulated to fossil fuel lobbying and corporate interests. Young climate activists who traveled to Dubai seeking real action have been betrayed by politicians who prioritize industry profits over planetary survival. We cannot afford more empty promises while the climate crisis accelerates."
        },

        # Liberty Herald (Right-leaning source) - 3 articles
        {
            "source": "Liberty Herald",
            "headline": "Fed's Prudent Rate Decision Reinforces Economic Stability",
            "content": "The Federal Reserve's measured quarter-point rate increase demonstrates responsible monetary policy that will preserve long-term economic prosperity. By raising rates to 5.5%, Fed officials are taking necessary steps to prevent runaway inflation that would devastate savings and fixed incomes. This disciplined approach protects the purchasing power that American families have worked hard to build. Free market principles and sound fiscal management require tough decisions that ensure sustainable growth for job creators and investors."
        },
        {
            "source": "Liberty Herald",
            "headline": "American AI Innovation Leads Global Technology Revolution",
            "content": "OpenAI's breakthrough demonstrates why American innovation continues to lead the world in transformative technology. This achievement showcases the power of free enterprise and competitive markets to deliver solutions that benefit humanity. While other nations impose heavy-handed regulations that stifle innovation, American companies are unleashing AI capabilities that will create jobs, boost productivity, and solve complex problems. America's technological superiority depends on supporting pioneering companies through pro-growth policies and reduced government interference."
        },
        {
            "source": "Liberty Herald",
            "headline": "Pragmatic Climate Deal Balances Environmental Goals with Economic Reality",
            "content": "The COP28 agreement demonstrates mature leadership by acknowledging environmental concerns while protecting economic stability and energy security. The careful 'transition away' language recognizes that abrupt fossil fuel elimination would devastate working families and developing nations that depend on affordable energy. American energy producers have already reduced emissions through innovation and cleaner technologies, proving that market solutions work better than government mandates. This balanced approach protects jobs while investing in alternatives."
        },

        # National Press Bureau (Neutral source, AP-style) - 3 articles
        {
            "source": "National Press Bureau",
            "headline": "New Alzheimer's Drug Shows Promise in Phase 3 Trial",
            "content": "Pharmaceutical company Biogen announced positive results from a Phase 3 clinical trial of its experimental Alzheimer's treatment, showing a 27% reduction in cognitive decline over 18 months. The drug, which targets amyloid plaques in the brain, was tested on 1,200 participants with early-stage Alzheimer's disease. Side effects included brain swelling in 12% of patients, though most cases were mild. The FDA is expected to review the application within six months. Healthcare economists estimate the drug could cost $50,000 annually per patient."
        },
        {
            "source": "National Press Bureau",
            "headline": "Amazon Reports Record Holiday Sales Despite Economic Uncertainty",
            "content": "Amazon announced record-breaking holiday sales figures, with revenue increasing 14% year-over-year to $170 billion in the fourth quarter. The e-commerce giant attributed growth to expanded Prime membership, faster delivery options, and strong performance in cloud computing services. Amazon Web Services generated $24 billion in revenue, up 13% from the previous year. Despite broader economic concerns about inflation and consumer spending, online shopping demand remained robust throughout the holiday season."
        },
        {
            "source": "National Press Bureau",
            "headline": "Supreme Court Agrees to Hear Case on Social Media Content Moderation",
            "content": "The Supreme Court announced it will review a case challenging state laws that restrict social media platforms' ability to moderate content. The case involves laws passed in Texas and Florida that limit platforms' authority to remove posts or suspend users. Tech companies argue these laws violate their First Amendment rights and could force them to host harmful content. State officials contend the laws prevent political censorship and protect free speech. Legal experts expect the decision to clarify platform liability authority."
        },

        # Social Justice Today (Left-leaning source) - 3 articles
        {
            "source": "Social Justice Today",
            "headline": "Big Pharma's Alzheimer's Scam: Modest Benefits, Massive Profits",
            "content": "Biogen's hyped Alzheimer's breakthrough reveals everything wrong with America's profit-driven healthcare system. While a 27% reduction in cognitive decline sounds impressive, this represents minimal real-world improvement that only wealthy families can afford at $50,000 yearly. Meanwhile, millions of seniors cannot afford basic medications for diabetes and heart disease. The drug's brain swelling side effects raise serious safety concerns that regulatory capture may overlook. Healthcare should be a human right, not a luxury commodity for pharmaceutical profiteering."
        },
        {
            "source": "Social Justice Today",
            "headline": "Amazon's Record Profits Built on Worker Exploitation and Market Abuse",
            "content": "Amazon's $170 billion revenue surge exposes the dark reality behind corporate greed: massive profits extracted through systematic worker exploitation and anti-competitive practices. While executives celebrate record sales, warehouse employees endure dangerous working conditions, unrealistic productivity targets, and poverty wages despite generating billions in value. The company's monopolistic control destroys small businesses while avoiding taxes that could fund public services. Amazon's 'success' represents everything wrong with unregulated capitalism."
        },
        {
            "source": "Social Justice Today",
            "headline": "Supreme Court Case Threatens Online Safety and Democratic Discourse",
            "content": "The Supreme Court's decision to hear the social media case represents a dangerous threat to online safety and democratic values. Texas and Florida's authoritarian laws would force platforms to amplify hate speech, misinformation, and extremist content that threatens vulnerable communities. These Republican-backed measures are designed to protect right-wing disinformation campaigns that undermine elections and public health. The Court must recognize that responsible content policies protect democracy from those who weaponize 'free speech' to spread dangerous lies."
        },

        # Free Market Weekly (Right-leaning source) - 3 articles
        {
            "source": "Free Market Weekly",
            "headline": "Medical Innovation Delivers Hope for Alzheimer's Families",
            "content": "Biogen's successful Alzheimer's trial demonstrates how private sector innovation creates life-changing treatments that government bureaucracy could never deliver. The 27% improvement in cognitive function offers genuine hope to millions of families battling this devastating disease. While critics focus on costs, they ignore the enormous research investments required to develop breakthrough therapies. Competition and patent protection incentivize the risk-taking that produces medical miracles. Insurance markets will make treatments accessible to those who need them most."
        },
        {
            "source": "Free Market Weekly",
            "headline": "Amazon's Success Proves American Capitalism Delivers Value",
            "content": "Amazon's exceptional $170 billion quarterly performance showcases how free market competition creates value for consumers, shareholders, and the broader economy. The company's success stems from relentless focus on customer satisfaction, operational efficiency, and technological innovation that benefits millions worldwide. By creating 100,000 jobs and investing $15 billion in infrastructure, Amazon demonstrates corporate responsibility and economic leadership. Critics who attack successful companies ignore the immense value created through voluntary market transactions."
        },
        {
            "source": "Free Market Weekly",
            "headline": "Historic Case Could Restore Free Speech Rights Online",
            "content": "The Supreme Court's landmark social media case offers hope for restoring constitutional free speech protections in the digital age. For too long, Big Tech monopolies have silenced conservative voices while promoting left-wing propaganda under the guise of 'content moderation.' Texas and Florida's laws recognize that social media platforms function as public forums where Americans exchange ideas and participate in democratic debate. The Court must affirm that the First Amendment applies online, preventing Silicon Valley censorship that threatens our republic."
        },

        # Balanced Tribune (Mixed bias source) - 4 articles with varied political leanings
        {
            "source": "Balanced Tribune",
            "headline": "Fed Rate Hike: Necessary Medicine or Economic Burden?",
            "content": "The Federal Reserve's latest rate increase prioritizes Wall Street stability over working families already crushed by inflation and housing costs. While Powell claims this fights inflation, the real winners are wealthy investors who benefit from higher returns on their capital. Middle-class Americans face even steeper mortgage payments and credit card bills. This hawkish monetary policy reflects the Fed's bias toward protecting asset prices rather than supporting employment and wage growth for ordinary workers."
        },
        {
            "source": "Balanced Tribune",
            "headline": "American Innovation Leadership Drives AI Breakthrough",
            "content": "OpenAI's latest advancement showcases why American free enterprise continues to lead global technology development. While other nations burden their companies with excessive regulations, our competitive market system unleashes the entrepreneurial spirit that creates world-changing innovations. This breakthrough will generate high-paying jobs, boost productivity across industries, and maintain America's technological edge. Smart policy should support these pioneering companies rather than hampering them with bureaucratic red tape."
        },
        {
            "source": "Balanced Tribune",
            "headline": "Climate Summit: Another Round of Empty Corporate Promises",
            "content": "Once again, fossil fuel companies and their political allies have sabotaged meaningful climate action with watered-down language that protects industry profits over planetary survival. The 'transition away' compromise gives oil giants exactly the wiggle room they need to continue expanding production while communities face devastating floods and wildfires. Young activists who demanded real change have been betrayed by politicians who prioritize corporate donors over future generations."
        },
        {
            "source": "Balanced Tribune",
            "headline": "Medical Innovation Delivers Hope Through Free Market Competition",
            "content": "Biogen's Alzheimer's breakthrough demonstrates how private sector research and development creates life-saving treatments that government bureaucracy could never deliver. The 27% improvement offers genuine hope to families facing this devastating disease. While critics complain about costs, they ignore the massive investments required to develop breakthrough therapies. Patent protection and market incentives drive the innovation that produces medical miracles, proving that capitalism delivers results for patients worldwide."
        },

        # Independent Monitor (Mixed bias source) - 3 articles with varied political leanings
        {
            "source": "Independent Monitor",
            "headline": "Big Tech's Monopolistic Stranglehold Destroys Fair Competition",
            "content": "Amazon's relentless expansion represents everything wrong with unregulated corporate power in America. This monopolistic behemoth has systematically destroyed small businesses, exploited warehouse workers, and avoided paying fair taxes while Jeff Bezos accumulates obscene wealth. The company's anti-competitive practices have created a rigged marketplace where innovation dies and consumer choice becomes an illusion. We need aggressive antitrust action to break up these digital monopolies before they completely eliminate economic opportunity for working families."
        },
        {
            "source": "Independent Monitor",
            "headline": "Constitutional Principles Must Prevail Over Silicon Valley Censorship",
            "content": "Big Tech's authoritarian content policies represent a clear and present danger to American democracy and free speech. These unelected Silicon Valley oligarchs have appointed themselves as the arbiters of truth, systematically silencing conservative voices while amplifying left-wing propaganda. Texas and Florida's courageous legislation recognizes that social media platforms function as the modern public square where citizens must be free to express diverse viewpoints. The Supreme Court must defend our constitutional rights against this coordinated assault on free expression."
        },
        {
            "source": "Independent Monitor",
            "headline": "Healthcare Innovation Balances Promise and Accessibility Challenges",
            "content": "Recent pharmaceutical breakthroughs demonstrate the remarkable potential of modern drug development while highlighting persistent questions about treatment accessibility and pricing. Clinical trial results show significant improvements for patients with previously challenging conditions, reflecting decades of scientific research and development investment. However, high treatment costs continue to limit access for many patients, particularly in underserved communities. Healthcare systems are exploring various policy approaches to balance innovation incentives with broader treatment access."
        }
    ]

    # Define Pydantic model for detailed article analysis
    class ArticleAnalysis(BaseModel):
        """Comprehensive analysis of news article content and bias."""
        bias_indicators: str = Field(description="Key words or phrases that indicate political bias")
        emotional_language: str = Field(description="Emotionally charged words or neutral descriptive language")
        opinion_markers: str = Field(description="Words or phrases that signal opinion vs. factual reporting")

    # Create DataFrame from news articles
    df = session.create_dataframe(news_articles)

    print("📰 News Bias Detection Pipeline")
    print("=" * 70)
    print(f"Analyzing {df.count()} news articles from {df.select('source').drop_duplicates(['source']).count()} sources")

    # Show dataset composition
    print("\n📊 Dataset Composition:")
    df.group_by("source").agg(fc.count("*").alias("articles")).order_by("source").show()

    print("\n🔍 Performing semantic bias detection...")
    print("First, we extract key information from each article.\n")

    # Create combined text for context-aware analysis
    combined_content = fc.text.concat(
        fc.col("headline"),
        fc.lit(" | "),
        fc.col("content")
    )

    # We can use Semantic Classification to identify primary topics and content bias for each article.
    # We can use `.cache()` to ensure these expensive LLM operations don't need to be re-run each time we modify
    # the resultant materialized dataframe.
    enriched_df = df.with_column("combined_content", combined_content).select(
        fc.col("source"),
        fc.col("headline"),
        fc.col("content"),
        # Primary topic classification
        fc.semantic.classify(
            fc.col("combined_content"),
            ["politics", "technology", "business", "climate", "healthcare"]
        ).alias("primary_topic"),
        # Content Metadata using semantic.extract
        fc.semantic.extract(
            fc.col("combined_content"),
            ArticleAnalysis,
            max_output_tokens=512,
        ).alias("analysis_metadata"),
    ).unnest("analysis_metadata")

    combined_extracts = fc.text.jinja(
        (
            "Primary Topic: {{primary_topic}}\n"
            "Political Bias Indicators: {{bias_indicators}}\n"
            "Emotional Language Summary: {{emotional_language}}\n"
            "Opinion Markers: {{opinion_markers}}"
        ),
        primary_topic=fc.col("primary_topic"),
        bias_indicators=fc.col("bias_indicators"),
        emotional_language=fc.col("emotional_language"),
        opinion_markers=fc.col("opinion_markers")
    )
    enriched_df = enriched_df.with_column("combined_extracts", combined_extracts)
    print("Then, we classify the political bias of the article based on our extracts.\n")
    results_df = enriched_df.select(
        "*",
        fc.semantic.classify(fc.col("combined_extracts"), ["far_left", "left_leaning", "neutral", "right_leaning", "far_right"]).alias("content_bias"),
        fc.semantic.classify(fc.col("combined_extracts"), ["sensationalist", "informational"]).alias("journalistic_style")
    ).cache()
    print("\n📊 Complete Bias Detection Results:")
    print("=" * 70)
    results_df.show()
    results_df.write.save_as_table("news_bias_analysis", mode="overwrite")

    # Distribution Analysis -- We can combine our classification categories with traditional data engineering to
    # calculate the distribution of bias patterns across articles from a given source and show examples of emotional_language
    # and opinion_markers in biased articles.
    print("\n📊 Overall Distribution Analysis:")
    print("=" * 70)

    print("This shows how consistently each source exhibits bias patterns across articles:")

    source_bias_distribution = results_df.group_by("source", "content_bias").agg(
        fc.count("*").alias("count")
    ).order_by(["source", fc.desc("count")])

    source_bias_distribution.show()

    # Topic distribution
    print("\n📈 Topic Distribution:")
    results_df.group_by("primary_topic").agg(
        fc.count("*").alias("count")
    ).order_by(fc.desc("count")).show()

    # Bias level distribution
    print("\n🎯 Content Bias Distribution:")
    results_df.group_by("content_bias").agg(
        fc.count("*").alias("count")
    ).order_by(fc.desc("count")).show()

    # Topic vs Bias cross-analysis
    print("\n🔍 Topic vs Bias Analysis:")
    results_df.group_by("primary_topic", "content_bias").agg(
        fc.count("*").alias("count")
    ).order_by(cols=[fc.col("primary_topic"), fc.desc("count")]).show()

    # Bias Indicators Analysis
    bias_indicators_df = results_df.select(
        "source",
        "headline",
        "content_bias",
        "bias_indicators",
        "emotional_language",
        "opinion_markers"
    )

    print("\n🔍 Bias Language Analysis:")
    print("=" * 70)

    # Show examples of neutral vs biased language
    print("\n📰 Neutral Articles - Language Patterns:")
    bias_indicators_df.filter(
        fc.col("content_bias") == "neutral"
    ).select("source", "headline", "bias_indicators", "opinion_markers").show(5)

    print("\n📰 Biased Articles - Language Patterns:")
    bias_indicators_df.filter(
        (fc.col("content_bias") != "neutral")
    ).select("source", "headline", "content_bias", "bias_indicators", "emotional_language", "opinion_markers").show(6)

    # Quality Assessment
    print("\n📊 Article Quality Assessment:")
    print("=" * 70)

    # Summary Statistics
    print("\n📈 Summary Statistics:")
    print("-" * 30)
    total_articles = results_df.count()
    neutral_articles = results_df.filter(fc.col("content_bias") == "neutral").count()
    biased_articles = results_df.filter((fc.col("content_bias") != "neutral")).count()

    print(f"Total articles analyzed: {total_articles}")
    print(f"Neutral articles: {neutral_articles} ({neutral_articles/total_articles:.1%})")
    print(f"Biased articles: {biased_articles} ({biased_articles/total_articles:.1%})")

    results_df = results_df.with_column("article_attributes", fc.text.jinja(
        (
            "Primary Topics: {{primary_topic}}\n"
            "Detected Political Bias: {{content_bias}}\n"
            "Detected Bias Indicators: {{bias_indicators}}\n"
            "Opinion Indicators: {{opinion_markers}}\n"
            "Emotional Language: {{emotional_language}}\n"
            "Journalistic Style: {{journalistic_style}}"
        ),
        primary_topic=fc.col("primary_topic"),
        content_bias=fc.col("content_bias"),
        bias_indicators=fc.col("bias_indicators"),
        opinion_markers=fc.col("opinion_markers"),
        emotional_language=fc.col("emotional_language"),
        journalistic_style=fc.col("journalistic_style")
    ))

    # Generate semantic summaries of language patterns for each source
    source_language_profiles = results_df.group_by("source").agg(
        # Use semantic.reduce to produce a media profile for each source, without including the entire original articles.
        # By grounding the model in the extracted information, we can be more confident in the produced results, as they will
        # have built-in justification.
        fc.semantic.reduce(
            """
            You are given a set of article analyses from {{news_outlet}}.
            Create a concise (3-5 sentence) media profile for {{news_outlet}}.
            Summarize the information provided without explicitly referencing it.
            """,
            column=fc.col("article_attributes"),
            group_context = {
                "news_outlet": fc.col("source"),
            },
            max_output_tokens=512,
        ).alias("source_profile"),
    ).select(fc.col("source"), fc.col("source_profile")).cache()

    print("\n🏢 AI-Generated Media Profiles:")
    print("-" * 50)
    source_language_profiles.show()
    source_language_profiles.write.save_as_table("media_profiles", mode="overwrite")

    # Clean up session
    session.stop()

    print("\n✅ News Bias Detection Complete!")
    print("=" * 70)
    print("\n🎯 Key Insights Demonstrated:")
    print("• Content-based bias detection without relying on source name predictions")
    print("• Source consistency analysis across multiple articles")
    print("• Language pattern identification for bias indicators")
    print("• Topic-agnostic bias detection (same source biased across different topics)")
    print("• Quality assessment with confidence scoring")
    print("\n🔍 Applications:")
    print("• Media literacy education showing how bias manifests in language")
    print("• Content moderation for balanced information presentation")
    print("• News aggregation with bias awareness")
    print("• Research on editorial patterns and media analysis")


if __name__ == "__main__":
    main()
