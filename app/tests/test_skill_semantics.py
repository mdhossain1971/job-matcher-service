"""Deterministic skill-semantics tests. No embedding model required."""
import unittest

from app.processors.chunker import DocumentChunker
from app.processors.matcher import MatchingEngine
from app.processors.skill_groups import (
    normalize_jd_text,
    parse_skill_groups,
    split_required_preferred,
)
from app.processors.tokenizer import canonicalize_skill, get_text_processor


def _profile(skills):
    return {
        "profile_id": 1,
        "skills": [{"name": s} for s in skills],
        "experiences": [],
        "education": [],
        "certifications": [],
    }


def _job(description, skills_required=None, requirements=None):
    return {
        "job_id": 1,
        "title": "Engineer",
        "description": description,
        "requirements": requirements,
        "skills_required": skills_required,
    }


def _match_lists(profile_skills, job):
    chunker = DocumentChunker()
    engine = MatchingEngine()
    profile = chunker.chunk_profile(_profile(profile_skills))
    job_chunks = chunker.chunk_job(_job(**job) if isinstance(job, dict) else job)
    score, matched, missing, details = engine._compute_skill_match(
        profile, job_chunks, None, None
    )
    return score, matched, missing, details, job_chunks


class CanonicalizeTests(unittest.TestCase):
    def test_alias_table(self):
        self.assertEqual(canonicalize_skill("Go"), "golang")
        self.assertEqual(canonicalize_skill("Golang"), "golang")
        self.assertEqual(canonicalize_skill("Go language"), "golang")
        self.assertEqual(canonicalize_skill("Postgres"), "postgresql")
        self.assertEqual(canonicalize_skill("PostgreSQL"), "postgresql")
        self.assertEqual(canonicalize_skill("K8s"), "kubernetes")
        self.assertEqual(canonicalize_skill("REST API"), "rest")
        self.assertEqual(canonicalize_skill("Google Cloud Platform"), "gcp")
        self.assertEqual(canonicalize_skill("Amazon Web Services"), "aws")

    def test_java_is_not_javascript(self):
        self.assertEqual(canonicalize_skill("Java"), "java")
        self.assertEqual(canonicalize_skill("JavaScript"), "javascript")
        self.assertNotEqual(canonicalize_skill("java"), canonicalize_skill("javascript"))

    def test_aws_is_not_azure(self):
        self.assertEqual(canonicalize_skill("AWS"), "aws")
        self.assertEqual(canonicalize_skill("Azure"), "azure")
        self.assertNotEqual(canonicalize_skill("aws"), canonicalize_skill("azure"))

    def test_extract_skills_canonicalizes(self):
        processor = get_text_processor()
        skills = processor.extract_skills("Experience with Go and PostgreSQL and K8s")
        self.assertIn("golang", skills)
        self.assertNotIn("go", skills)
        self.assertIn("postgresql", skills)
        self.assertNotIn("postgres", skills)
        self.assertIn("kubernetes", skills)
        self.assertNotIn("k8s", skills)


class GroupParseTests(unittest.TestCase):
    def test_slash_language_group(self):
        groups = parse_skill_groups("5+ years developing in Go/C#/Java/C++")
        multi = [g for g in groups if len(g) > 1]
        self.assertTrue(any(set(g) >= {"golang", "java", "c#", "c++"} for g in multi))

    def test_or_cloud_group(self):
        groups = parse_skill_groups("Experience with AWS, GCP, or Azure")
        self.assertTrue(any(set(g) == {"aws", "gcp", "azure"} for g in groups))

    def test_comma_list_without_or_stays_and(self):
        groups = parse_skill_groups("Required: Java, Spring Boot, Kafka, Kubernetes")
        self.assertTrue(all(len(g) == 1 for g in groups))
        flat = {g[0] for g in groups}
        self.assertTrue({"java", "kafka", "kubernetes"} <= flat)

    def test_ci_cd_not_split(self):
        groups = parse_skill_groups("Must have CI/CD experience and Java")
        self.assertFalse(any(set(g) == {"ci", "cd"} for g in groups))
        self.assertTrue(any(g == ["java"] for g in groups))

    def test_html_preferred_header_excluded(self):
        jd = (
            "<strong>Qualifications</strong><br>"
            "• 5+ years developing in Go/C#/Java/C++<br>"
            "<strong>Preferred Qualifications</strong><br>"
            "• Direct experience developing with Golang.<br>"
            "• Kubernetes<br>"
        )
        split = split_required_preferred(normalize_jd_text(jd))
        self.assertTrue(split.found_preferred_header)
        self.assertIn("golang", split.preferred_text.lower())
        self.assertNotIn("preferred", split.required_text.lower())
        preferred_skills = get_text_processor().extract_skills(split.preferred_text)
        self.assertIn("kubernetes", preferred_skills)
        required_groups = parse_skill_groups(split.required_text)
        required_flat = {s for g in required_groups for s in g}
        self.assertNotIn("kubernetes", required_flat)


class SkillMatchSemanticsTests(unittest.TestCase):
    def test_1_language_or_group_java_satisfies(self):
        _, matched, missing, _, chunks = _match_lists(
            ["Java"],
            {"description": "5+ years developing in Go/C#/Java/C++"},
        )
        self.assertIn("java", matched)
        for alt in ("golang", "c#", "c++", "go"):
            self.assertNotIn(alt, missing)
        self.assertTrue(any(len(g) > 1 for g in chunks.required_skill_groups))

    def test_2_cloud_or_group_azure_not_missing(self):
        _, matched, missing, _, _ = _match_lists(
            ["AWS"],
            {"description": "Experience with AWS, GCP, or Azure"},
        )
        self.assertIn("aws", matched)
        self.assertNotIn("azure", missing)
        self.assertNotIn("gcp", missing)

    def test_3_golang_matches_go(self):
        _, matched, missing, _, _ = _match_lists(
            ["Golang"],
            {"description": "Required skills: Go", "skills_required": "Go"},
        )
        self.assertIn("golang", matched)
        self.assertNotIn("go", missing)
        self.assertNotIn("golang", missing)

    def test_4_java_does_not_match_javascript(self):
        _, matched, missing, _, _ = _match_lists(
            ["Java"],
            {"description": "Must know JavaScript", "skills_required": "JavaScript"},
        )
        self.assertNotIn("javascript", matched)
        self.assertIn("javascript", missing)

    def test_5_postgresql_matches_postgres(self):
        _, matched, missing, _, _ = _match_lists(
            ["PostgreSQL"],
            {"description": "Experience with Postgres"},
        )
        self.assertIn("postgresql", matched)
        self.assertNotIn("postgres", missing)

    def test_6_aws_does_not_satisfy_azure(self):
        _, matched, missing, _, _ = _match_lists(
            ["AWS"],
            {"description": "Required: Azure"},
        )
        self.assertNotIn("azure", matched)
        self.assertIn("azure", missing)

    def test_7_preferred_golang_not_missing_required(self):
        jd = (
            "Qualifications\n"
            "Java\n"
            "Preferred Qualifications\n"
            "Golang\n"
        )
        _, matched, missing, _, chunks = _match_lists(
            ["Java"],
            {"description": jd},
        )
        self.assertIn("java", matched)
        self.assertNotIn("golang", missing)
        self.assertIn("golang", chunks.preferred_skills_list)

    def test_8_two_or_groups_both_satisfied(self):
        jd = (
            "Qualifications\n"
            "Experience with Java or Go.\n"
            "Experience with AWS or Azure.\n"
        )
        _, matched, missing, details, _ = _match_lists(
            ["AWS", "Java"],
            {"description": jd},
        )
        self.assertIn("java", matched)
        self.assertIn("aws", matched)
        self.assertEqual(missing, [])
        self.assertEqual(details["exact_matches"], details["total_required"])

    def test_9_language_group_satisfied_kubernetes_missing(self):
        jd = (
            "Qualifications\n"
            "Experience with Java or Go.\n"
            "Kubernetes experience is required.\n"
        )
        _, matched, missing, _, _ = _match_lists(
            ["Java"],
            {"description": jd},
        )
        self.assertIn("java", matched)
        self.assertIn("kubernetes", missing)
        self.assertNotIn("golang", missing)

    def test_unsatisfied_or_group_lists_alternatives_not_independent_ands(self):
        """missing_skills may list OR alternatives of one unsatisfied group."""
        _, matched, missing, details, _ = _match_lists(
            ["Python"],
            {"description": "Experience with AWS, GCP, or Azure"},
        )
        self.assertEqual(matched, [])
        self.assertEqual(set(missing), {"aws", "gcp", "azure"})
        unsatisfied = details["unsatisfied_groups"]
        self.assertEqual(len(unsatisfied), 1)
        self.assertEqual(set(unsatisfied[0]), {"aws", "gcp", "azure"})

    def test_skills_required_flatten_does_not_and_promote_alternatives(self):
        _, matched, missing, _, chunks = _match_lists(
            ["Java", "AWS"],
            {
                "description": (
                    "Qualifications\n"
                    "5+ years developing in Go/C#/Java/C++\n"
                    "Experience with AWS, GCP, or Azure\n"
                    "Preferred Qualifications\n"
                    "Direct Golang experience\n"
                ),
                "skills_required": "aws, azure, gcp, go, java",
            },
        )
        self.assertIn("java", matched)
        self.assertIn("aws", matched)
        self.assertNotIn("azure", missing)
        self.assertNotIn("go", missing)
        self.assertNotIn("golang", missing)
        self.assertTrue(any(len(g) > 1 for g in chunks.required_skill_groups))


if __name__ == "__main__":
    unittest.main()
