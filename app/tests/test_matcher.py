#!/usr/bin/env python3
"""
Test the job matcher service with real profile and job data.

Run with: python -m app.tests.test_matcher
"""
import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.processors.tokenizer import get_text_processor
from app.processors.chunker import get_chunker
from app.models.embedder import get_embedding_model
from app.processors.matcher import get_matching_engine
from app.storage.memory import get_storage


# ============================================
# YOUR REAL PROFILE DATA
# ============================================

PROFILE = {
    "profile_id": 1,
    "profile_name": "Senior Software Engineer",
    "summary": "15+ years of experience in Application Architecture, Technical Design, Development, and Performance Tuning across diverse industries. Expertise in Java/J2EE, Microservices, API development, and cloud platforms like Kubernetes and Docker.",
    "skills": [
        {"name": "Java", "years": 15, "proficiency": "EXPERT"},
        {"name": "J2EE", "years": 15, "proficiency": "EXPERT"},
        {"name": "Python", "years": 5, "proficiency": "ADVANCED"},
        {"name": "Spring Boot", "years": 10, "proficiency": "EXPERT"},
        {"name": "Micronaut", "years": 3, "proficiency": "ADVANCED"},
        {"name": "Kafka", "years": 8, "proficiency": "EXPERT"},
        {"name": "Kubernetes", "years": 5, "proficiency": "ADVANCED"},
        {"name": "Docker", "years": 6, "proficiency": "ADVANCED"},
        {"name": "AWS", "years": 5, "proficiency": "ADVANCED"},
        {"name": "GCP", "years": 4, "proficiency": "ADVANCED"},
        {"name": "Microservices", "years": 10, "proficiency": "EXPERT"},
        {"name": "GRPC", "years": 5, "proficiency": "ADVANCED"},
        {"name": "Oracle", "years": 10, "proficiency": "ADVANCED"},
        {"name": "Cassandra", "years": 5, "proficiency": "ADVANCED"},
        {"name": "Redis", "years": 6, "proficiency": "ADVANCED"},
        {"name": "GraphQL", "years": 3, "proficiency": "INTERMEDIATE"},
        {"name": "Prometheus", "years": 4, "proficiency": "ADVANCED"},
        {"name": "Grafana", "years": 4, "proficiency": "ADVANCED"},
    ],
    "experiences": [
        {
            "company_name": "Tailored Brands",
            "title": "Principal Software Engineer",
            "description": "Leading migration to Constructor.io. Developed customer-operations agent using MCP with RAG backend.",
            "is_current": True
        },
        {
            "company_name": "AppDynamics (Cisco)",
            "title": "Staff Software Engineer",
            "description": "Led team of 5 engineers. Designed GRPC and HTTP microservices. Developed Kafka Streams data ingestion.",
            "is_current": False
        },
        {
            "company_name": "Macy's Inc",
            "title": "Lead Software Engineer",
            "description": "Led team of 7 engineers. Architected security gateway reducing unauthorized traffic by 25%.",
            "is_current": False
        }
    ],
    "education": [
        {"institution": "University of Louisiana at Lafayette", "degree": "M.S.", "field_of_study": "Computer Science"},
        {"institution": "Minnesota State University", "degree": "B.S.", "field_of_study": "Computer Information Science"}
    ],
    "certifications": ["Oracle Certified Professional Java SE 6 Programmer"],
    "years_of_experience": 15
}


# ============================================
# SAMPLE JOBS TO TEST
# ============================================

JOB_GOOD_MATCH = {
    "job_id": 101,
    "title": "Senior Software Engineer",
    "company_name": "Walmart",
    "description": "Design and develop high-performance microservices using Java and Spring Boot. Lead technical design discussions. Work with Kafka for event-driven architecture. Deploy on Kubernetes.",
    "requirements": "5+ years Java/J2EE. Strong Spring Boot and Microservices experience. Kafka experience. Kubernetes and Docker.",
    "skills_required": "Java, Spring Boot, Kafka, Kubernetes, Docker, Microservices",
    "location": "Sunnyvale, CA",
    "remote_type": "HYBRID"
}

JOB_MODERATE_MATCH = {
    "job_id": 102,
    "title": "Full Stack Engineer",
    "company_name": "Meta",
    "description": "Build user interfaces with React and TypeScript. Develop backend services. Work with GraphQL APIs.",
    "requirements": "3+ years React. TypeScript expertise. Node.js backend experience. GraphQL knowledge.",
    "skills_required": "React, TypeScript, Node.js, GraphQL, PostgreSQL",
    "location": "Menlo Park, CA",
    "remote_type": "REMOTE"
}

JOB_POOR_MATCH = {
    "job_id": 103,
    "title": "iOS Developer",
    "company_name": "Apple",
    "description": "Develop iOS applications using Swift and SwiftUI. Work on Apple ecosystem apps.",
    "requirements": "5+ years iOS development. Swift expert. SwiftUI experience. Published apps on App Store.",
    "skills_required": "Swift, SwiftUI, iOS, Xcode, Objective-C",
    "location": "Cupertino, CA",
    "remote_type": "ONSITE"
}


def test_skill_extraction():
    """Test skill extraction from text"""
    print("\n" + "="*60)
    print("TEST 1: Skill Extraction")
    print("="*60)
    
    processor = get_text_processor()
    processor.load()
    
    text = """
    We need a Java developer with Spring Boot experience.
    Should know Kafka, Kubernetes, and have AWS or GCP experience.
    MongoDB and PostgreSQL are a plus.
    """
    
    skills = processor.extract_skills(text)
    print(f"Extracted skills: {skills}")
    
    assert "java" in skills
    assert "spring boot" in skills or "spring" in skills
    assert "kafka" in skills
    print("✓ Skill extraction working correctly")


def test_chunking():
    """Test document chunking"""
    print("\n" + "="*60)
    print("TEST 2: Document Chunking")
    print("="*60)
    
    chunker = get_chunker()
    
    # Test profile chunking
    profile_chunks = chunker.chunk_profile(PROFILE)
    print(f"Profile ID: {profile_chunks.profile_id}")
    print(f"Skills extracted: {len(profile_chunks.skills_list)}")
    print(f"Years experience: {profile_chunks.years_experience}")
    print(f"Skills text preview: {profile_chunks.skills_text[:100]}...")
    
    # Test job chunking
    job_chunks = chunker.chunk_job(JOB_GOOD_MATCH)
    print(f"\nJob ID: {job_chunks.job_id}")
    print(f"Required skills: {job_chunks.required_skills_list}")
    print(f"Years required: {job_chunks.years_required}")
    
    assert len(profile_chunks.skills_list) > 10
    assert "java" in [s.lower() for s in profile_chunks.skills_list]
    print("✓ Chunking working correctly")


def test_embedding():
    """Test embedding generation"""
    print("\n" + "="*60)
    print("TEST 3: Embedding Generation")
    print("="*60)
    
    embedder = get_embedding_model()
    embedder.load()
    
    # Test single text embedding
    text = "Java developer with Spring Boot experience"
    embedding = embedder.encode(text)
    
    print(f"Embedding dimension: {len(embedding)}")
    print(f"Embedding sample: {embedding[:5]}...")
    
    # Test similarity
    text1 = "Java software engineer"
    text2 = "Python data scientist"
    text3 = "Java backend developer"
    
    emb1 = embedder.encode(text1)
    emb2 = embedder.encode(text2)
    emb3 = embedder.encode(text3)
    
    sim_1_2 = embedder.similarity(emb1, emb2)
    sim_1_3 = embedder.similarity(emb1, emb3)
    
    print(f"\nSimilarity 'Java engineer' vs 'Python scientist': {sim_1_2:.3f}")
    print(f"Similarity 'Java engineer' vs 'Java backend': {sim_1_3:.3f}")
    
    assert sim_1_3 > sim_1_2, "Java-Java should be more similar than Java-Python"
    print("✓ Embedding and similarity working correctly")


def test_full_matching():
    """Test full profile-job matching"""
    print("\n" + "="*60)
    print("TEST 4: Full Profile-Job Matching")
    print("="*60)
    
    engine = get_matching_engine()
    storage = get_storage()
    
    # Embed profile
    print("\nEmbedding profile...")
    profile_emb = engine.embed_profile(PROFILE)
    storage.store_profile(profile_emb)
    print(f"Profile embedded. Skills: {len(profile_emb.chunks.skills_list)}")
    
    # Test each job
    jobs = [
        ("Good Match (Java/Microservices)", JOB_GOOD_MATCH),
        ("Moderate Match (Full Stack)", JOB_MODERATE_MATCH),
        ("Poor Match (iOS)", JOB_POOR_MATCH)
    ]
    
    results = []
    for job_name, job_data in jobs:
        print(f"\n--- {job_name} ---")
        
        job_emb = engine.embed_job(job_data)
        result = engine.match(profile_emb, job_emb)
        results.append(result)
        
        print(f"Job: {job_data['title']} at {job_data['company_name']}")
        print(f"Score: {result.score} ({result.recommendation.value})")
        print(f"Breakdown:")
        print(f"  - Skill Match: {result.breakdown.skill_match}")
        print(f"  - Experience Match: {result.breakdown.experience_match}")
        print(f"  - Semantic Match: {result.breakdown.semantic_match}")
        print(f"  - Requirements Fit: {result.breakdown.requirements_fit}")
        print(f"Matched Skills: {result.matched_skills}")
        print(f"Missing Skills: {result.missing_skills}")
        print(f"Strengths: {result.strengths}")
        print(f"Gaps: {result.gaps}")
        print(f"Processing time: {result.processing_time_ms}ms")
    
    # Verify scoring makes sense
    good_score = results[0].score
    moderate_score = results[1].score
    poor_score = results[2].score
    
    print(f"\n--- Score Comparison ---")
    print(f"Good Match (Java): {good_score}")
    print(f"Moderate Match (Full Stack): {moderate_score}")
    print(f"Poor Match (iOS): {poor_score}")
    
    assert good_score > moderate_score, "Java job should score higher than Full Stack"
    assert moderate_score > poor_score, "Full Stack should score higher than iOS"
    print("\n✓ Scoring ranking is correct!")


def test_api_response_format():
    """Test that match result has correct format"""
    print("\n" + "="*60)
    print("TEST 5: API Response Format")
    print("="*60)
    
    engine = get_matching_engine()
    
    profile_emb = engine.embed_profile(PROFILE)
    job_emb = engine.embed_job(JOB_GOOD_MATCH)
    result = engine.match(profile_emb, job_emb)
    
    # Convert to dict (as JSON response would be)
    result_dict = result.model_dump()
    
    print("Response structure:")
    print(json.dumps(result_dict, indent=2, default=str))
    
    # Verify structure
    assert "score" in result_dict
    assert "recommendation" in result_dict
    assert "breakdown" in result_dict
    assert "matched_skills" in result_dict
    assert "missing_skills" in result_dict
    assert "strengths" in result_dict
    assert "gaps" in result_dict
    
    print("\n✓ API response format is correct!")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("JOB MATCHER SERVICE - TEST SUITE")
    print("="*60)
    
    try:
        test_skill_extraction()
        test_chunking()
        test_embedding()
        test_full_matching()
        test_api_response_format()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
