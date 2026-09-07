"""requirements_fit uses qualification evidence, not degree-vs-job-header."""
import os
import unittest

from app.processors.chunker import DocumentChunker
from app.processors.matcher import MatchingEngine


MATCHING_PROFILE = {
    "profile_id": 1,
    "summary": "Backend engineer focused on Java microservices and AWS.",
    "skills": [
        {"name": "Java"},
        {"name": "AWS"},
        {"name": "Microservices"},
    ],
    "experiences": [
        {
            "company_name": "Acme",
            "title": "Staff Backend Engineer",
            "description": (
                "Built multithreaded Java services, tuned performance, and "
                "ran cloud-native microservices on AWS."
            ),
        }
    ],
    "education": [
        {"institution": "State U", "degree": "B.S.", "field_of_study": "Computer Science"}
    ],
    "certifications": [],
    "years_of_experience": 10,
}

POOR_PROFILE = {
    "profile_id": 2,
    "summary": "Retail store manager with customer service experience.",
    "skills": [{"name": "Excel"}],
    "experiences": [
        {
            "company_name": "Shop",
            "title": "Store Manager",
            "description": "Managed inventory and a team of cashiers.",
        }
    ],
    "education": [
        {"institution": "Community College", "degree": "A.A.", "field_of_study": "Business"}
    ],
    "certifications": [],
    "years_of_experience": 8,
}

JOB = {
    "job_id": 10,
    "title": "Senior Backend Engineer",
    "company_name": "ExampleCorp",
    "location": "Remote",
    "remote_type": "REMOTE",
    "description": (
        "Qualifications\n"
        "10+ years Java backend experience.\n"
        "AWS cloud and microservices required.\n"
        "Computer Science degree.\n"
        "Preferred Qualifications\n"
        "Cybersecurity industry experience.\n"
    ),
}


class RequirementsFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        cls.chunker = DocumentChunker()
        cls.engine = MatchingEngine()
        try:
            cls.engine.embedder.load()
        except Exception as exc:
            raise unittest.SkipTest(f"Embedding model unavailable: {exc}") from exc

    def test_10_evidence_not_degree_vs_header(self):
        profile_chunks = self.chunker.chunk_profile(MATCHING_PROFILE)
        job_chunks = self.chunker.chunk_job(JOB)

        self.assertIn("java", job_chunks.qualifications_text.lower())
        self.assertIn("aws", job_chunks.qualifications_text.lower())
        self.assertNotIn("Position:", job_chunks.qualifications_text)
        self.assertNotIn("ExampleCorp", job_chunks.qualifications_text)
        self.assertNotIn("cybersecurity", job_chunks.qualifications_text.lower())

        evidence = profile_chunks.qualification_evidence_text.lower()
        self.assertIn("java", evidence)
        self.assertIn("10 years", evidence)
        self.assertIn("multithreaded", evidence)
        self.assertNotIn("resume:", evidence)

        matching_emb = self.engine.embed_profile(MATCHING_PROFILE)
        job_emb = self.engine.embed_job(JOB)
        poor_emb = self.engine.embed_profile(POOR_PROFILE)

        matching_fit, _ = self.engine._compute_requirements_fit(
            matching_emb.chunks,
            job_emb.chunks,
            matching_emb.embeddings.get("qualification_evidence"),
            job_emb.embeddings.get("qualifications"),
        )
        education_only, _ = self.engine._compute_requirements_fit(
            matching_emb.chunks,
            job_emb.chunks,
            matching_emb.embeddings.get("education"),
            job_emb.embeddings.get("qualifications"),
        )
        poor_fit, _ = self.engine._compute_requirements_fit(
            poor_emb.chunks,
            job_emb.chunks,
            poor_emb.embeddings.get("qualification_evidence"),
            job_emb.embeddings.get("qualifications"),
        )

        self.assertGreater(matching_fit, education_only)
        self.assertGreater(matching_fit, poor_fit)


if __name__ == "__main__":
    unittest.main()
