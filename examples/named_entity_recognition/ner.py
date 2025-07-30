import re
from typing import List, Optional

from pydantic import BaseModel, Field

import fenic as fc


def main(config: Optional[fc.SessionConfig] = None):
    # Configure session with semantic capabilities
    config = config or fc.SessionConfig(
        app_name="security_vulnerability_ner",
        semantic=fc.SemanticConfig(
            language_models= {
                "mini" : fc.OpenAILanguageModel(model_name="gpt-4o-mini", rpm=500, tpm=200_000)
            }
        )
    )

    # Create session
    session = fc.Session.get_or_create(config)

    # Sample vulnerability reports data
    vulnerability_reports_data = [
        {
            "report_id": "CVE-2024-001",
            "source": "CVE Database",
            "title": "Critical OpenSSL Buffer Overflow",
            "content": "CVE-2024-3094: Buffer overflow in OpenSSL 3.0.0-3.0.12. Affects Ubuntu 22.04, RHEL 8. CVSS 9.8. IOCs: evil-domain.com, 10.0.0.50:443"
        },
        {
            "report_id": "THREAT-2024-002",
            "source": "Threat Intelligence",
            "title": "APT29 Campaign Targeting Financial Sector",
            "content": "APT29 targeting banks. Exploits CVE-2024-1234, CVE-2024-5678. Malware: SUNBURST 2.0. C2: c2-server.badguys.net (185.159.158.1)"
        },
        {
            "report_id": "SEC-ADV-2024-003",
            "source": "Security Advisory",
            "title": "Zero-Day in Popular WordPress Plugin",
            "content": "CVE-2024-9999: SQL injection in WP Super Cache 1.0.0-1.7.8. CVSS 8.5. Patch in 1.7.9. Related: CVE-2024-9998, CVE-2024-9997"
        },
        {
            "report_id": "INC-2024-004",
            "source": "Incident Report",
            "title": "Ransomware Attack on Healthcare Provider",
            "content": "LockBit 3.0 ransomware via CVE-2024-4444. Used Mimikatz 2.2.0. C2: 45.142.214.99:8443. Affected Windows Server 2016, 2019"
        },
        {
            "report_id": "VULN-2024-005",
            "source": "Bug Bounty Report",
            "title": "Authentication Bypass in Enterprise SaaS Platform",
            "content": "Auth bypass in AuthProvider 2.5.1 at login.platform.com. JWT alg:none vulnerability. Fixed in 2.5.2. Affects /api/v2/admin/*"
        }
    ]

    # Create DataFrame
    reports_df = session.create_dataframe(vulnerability_reports_data)

    print("🔒 Security Vulnerability NER Pipeline")
    print("=" * 70)
    print(f"Processing {reports_df.count()} vulnerability reports\n")

    # Stage 1: Basic NER with zero-shot extraction
    print("🔍 Stage 1: Zero-shot entity extraction...")

    # Define basic NER schema for security entities
    class BasicNERSchema(BaseModel):
        cve_ids: List[str] = Field(
            description="CVE identifiers in format CVE-YYYY-NNNNN"
        )
        software_packages: List[str] = Field(
            description="Software names and versions mentioned"
        )
        ip_addresses: List[str] = Field(
            description="IP addresses (IPv4 or IPv6)"
        )
        domains: List[str] = Field(
            description="Domain names and URLs"
        )
        file_hashes: List[str] = Field(
            description="File hashes (MD5, SHA1, SHA256)"
        )

    # Apply basic extraction
    basic_extraction_df = reports_df.select(
        "report_id",
        "source",
        "title",
        fc.semantic.extract("content", BasicNERSchema).alias("basic_entities")
    )

    # Display sample results
    print("Sample basic extraction results:")
    basic_readable = basic_extraction_df.select(
        "report_id",
        basic_extraction_df.basic_entities.cve_ids.alias("cve_ids"),
        basic_extraction_df.basic_entities.software_packages.alias("software_packages")
    )
    basic_readable.show(2)

    # Stage 2: Enhanced extraction with domain-specific schema
    print("\n🧠 Stage 2: Enhanced domain-specific extraction...")

    # Define enhanced schema with security-specific entities
    class EnhancedNERSchema(BaseModel):
        cve_ids: List[str] = Field(
            description="CVE identifiers in format CVE-YYYY-NNNNN"
        )
        software_packages: List[str] = Field(
            description="Software names with specific version numbers"
        )
        ip_addresses: List[str] = Field(
            description="IP addresses (IPv4 or IPv6)"
        )
        domains: List[str] = Field(
            description="Domain names, subdomains, and URLs"
        )
        file_hashes: List[str] = Field(
            description="File hashes with hash type prefix (MD5:, SHA1:, SHA256:)"
        )
        attack_vectors: List[str] = Field(
            description="Attack methods like buffer overflow, SQL injection, phishing"
        )
        threat_actors: List[str] = Field(
            description="Threat actor names, APT groups, ransomware families"
        )
        cvss_scores: List[str] = Field(
            description="CVSS scores and severity ratings"
        )
        mitre_techniques: List[str] = Field(
            description="MITRE ATT&CK technique IDs (TXXXX format)"
        )
        affected_systems: List[str] = Field(
            description="Operating systems, platforms, or infrastructure affected"
        )

    # Preprocess content for better extraction
    @fc.udf(return_type=fc.StringType)
    def preprocess_udf(content):
        # Standardize CVE format
        content = re.sub(r'CVE\s*-\s*(\d{4})\s*-\s*(\d+)', r'CVE-\1-\2', content)
        # Normalize version ranges
        content = re.sub(r'(\d+\.\d+\.\d+)\s+through\s+(\d+\.\d+\.\d+)', r'\1 to \2', content)
        # Clean up extra whitespace
        content = ' '.join(content.split())
        return content

    # Apply preprocessing and enhanced extraction
    enhanced_df = reports_df.select(
        "report_id",
        "source",
        "title",
        "content",
        preprocess_udf("content").alias("processed_content")
    ).select(
        "report_id",
        "source",
        "title",
        "content",
        fc.semantic.extract("processed_content", EnhancedNERSchema).alias("entities")
    )

    print("Enhanced extraction with security-specific entities:")
    enhanced_readable = enhanced_df.select(
        "report_id",
        enhanced_df.entities.threat_actors.alias("threat_actors"),
        enhanced_df.entities.attack_vectors.alias("attack_vectors"),
        enhanced_df.entities.cvss_scores.alias("cvss_scores")
    )
    enhanced_readable.show(2)

    # Stage 3: Process long documents with chunking
    print("\n📄 Stage 3: Chunking and processing long documents...")

    # Add content length for chunking decisions
    reports_with_length = enhanced_df.select(
        "*",
        fc.text.length(fc.col("content")).alias("content_length")
    )

    # Identify documents needing chunking (>80 characters for demo)
    long_reports = reports_with_length.filter(fc.col("content_length") > 80)
    short_reports = reports_with_length.filter(fc.col("content_length") <= 80)

    print(f"Documents requiring chunking: {long_reports.count()}")
    print(f"Documents processed whole: {short_reports.count()}")

    # Apply chunking to long documents
    chunked_df = long_reports.select(
        "report_id",
        "content",
        fc.text.recursive_word_chunk(
            fc.col("content"),
            chunk_size=50,
            chunk_overlap_percentage=15
        ).alias("chunks")
    ).explode("chunks").select(
        "report_id",
        fc.col("chunks").alias("chunk")
    )

    # Extract entities from each chunk
    chunk_entities_df = chunked_df.select(
        "report_id",
        "chunk",
        fc.semantic.extract("chunk", EnhancedNERSchema).alias("chunk_entities")
    )

    # Aggregate entities across chunks
    aggregated_entities = chunk_entities_df.group_by("report_id").agg(
        fc.collect_list(fc.col("chunk_entities")).alias("all_chunk_entities")
    )

    print("\nChunked extraction completed for long documents")
    print(f"Total chunks processed: {chunk_entities_df.count()}")

    # Show sample of aggregated chunk results
    print("\nSample aggregated entities from chunks:")
    aggregated_sample = aggregated_entities.select(
        "report_id",
        fc.array_size(fc.col("all_chunk_entities")).alias("chunks_with_entities")
    )
    aggregated_sample.show(2)

    # Stage 4: Validation and quality assurance
    print("\n✅ Stage 4: Validating extracted entities...")

    # Create a unified view for validation
    all_entities_df = enhanced_df.select(
        "report_id",
        "source",
        "title",
        "entities"
    )

    # Show extracted CVEs
    print("Extracted CVE IDs:")
    cve_summary = all_entities_df.select(
        "report_id",
        all_entities_df.entities.cve_ids.alias("extracted_cves")
    )
    cve_summary.show(3)

    # Stage 5: Analytics and aggregation
    print("\n📊 Stage 5: Entity analytics and insights...")

    # Flatten entities for analysis
    flattened_cves = all_entities_df.select(
        all_entities_df.entities.cve_ids.alias("cve_id")
    ).explode("cve_id").filter(fc.col("cve_id").is_not_null())

    flattened_software = all_entities_df.select(
        all_entities_df.entities.software_packages.alias("software")
    ).explode("software").filter(fc.col("software").is_not_null())

    flattened_threats = all_entities_df.select(
        all_entities_df.entities.threat_actors.alias("threat_actor")
    ).explode("threat_actor").filter(fc.col("threat_actor").is_not_null())

    # Most common CVEs
    print("\nTop CVEs mentioned:")
    cve_counts = flattened_cves.group_by("cve_id").agg(
        fc.count("*").alias("mentions")
    ).order_by(fc.col("mentions").desc())
    cve_counts.show(5)

    # Most affected software
    print("\nMost affected software:")
    software_counts = flattened_software.group_by("software").agg(
        fc.count("*").alias("mentions")
    ).order_by(fc.col("mentions").desc())
    software_counts.show(5)

    # Active threat actors
    print("\nActive threat actors:")
    threat_counts = flattened_threats.group_by("threat_actor").agg(
        fc.count("*").alias("reports")
    ).order_by(fc.col("reports").desc())
    threat_counts.show(5)

    # Create final comprehensive report
    print("\n📋 Final Security Intelligence Summary:")
    print("=" * 70)

    # Summary statistics
    total_cves = flattened_cves.count()
    unique_cves = flattened_cves.select("cve_id").drop_duplicates().count()
    total_threats = flattened_threats.count()
    unique_threats = flattened_threats.select("threat_actor").drop_duplicates().count()

    print(f"Total CVEs extracted: {total_cves} ({unique_cves} unique)")
    print(f"Total threat actors identified: {total_threats} ({unique_threats} unique)")
    print(f"Reports processed: {reports_df.count()}")

    # Generate actionable intelligence using semantic operations
    print("\n🎯 Actionable Intelligence:")

    # Define Pydantic model for risk assessment

    class ExtractedRiskInfo(BaseModel):
        """
        Directly extracted risk information from the report text.
        If a value is not present in the report, use an empty string.
        """

        severity_rating: str = Field(
            ...,
            description="Explicit severity rating or risk level as stated in the report "
                        "(e.g., 'critical', 'high', 'medium', 'low')"
        )
        cvss_score: str = Field(
            ...,
            description="CVSS score as stated in the report"
        )
        mitigation_steps: str = Field(
            ...,
            description="Quoted mitigation or remediation steps as stated in the report"
        )
        affected_systems: str = Field(
            ...,
            description="Exact systems, platforms, or users mentioned as affected in the report"
        )


    # Assess risk for each report
    risk_assessment_df = enhanced_df.select(
        "report_id",
        "title",
        fc.semantic.extract("content", ExtractedRiskInfo).alias("risk_assessment")
    )

    # Show high-risk items
    high_risk_df = risk_assessment_df.select(
        "report_id",
        "title",
        risk_assessment_df.risk_assessment.severity_rating.alias("risk_level"),
        risk_assessment_df.risk_assessment.mitigation_steps.alias("immediate_action"),
        risk_assessment_df.risk_assessment.affected_systems.alias("affected_scope")
    ).filter(
        (fc.col("risk_level") == "critical") | (fc.col("risk_level") == "high")
    )

    print("\nHigh-Risk Vulnerabilities Requiring Immediate Action:")
    high_risk_df.show()

    # Clean up
    session.stop()

    print("\n✅ Analysis complete!")
    print("\nNext steps for production deployment:")
    print("   - Integrate with vulnerability feeds (NVD, vendor advisories)")
    print("   - Set up real-time processing pipeline")
    print("   - Export to SIEM/SOAR platforms")
    print("   - Create automated incident response workflows")
    print("   - Build threat intelligence knowledge base")


if __name__ == "__main__":
    # Note: Ensure you have set your OpenAI API key:
    # export OPENAI_API_KEY="your-api-key-here"
    main()
