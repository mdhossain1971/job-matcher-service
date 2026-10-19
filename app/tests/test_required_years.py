import unittest

from app.processors.chunker import DocumentChunker
from app.processors.tokenizer import TextProcessor


PAYPAL = (
    "PayPal has been revolutionizing commerce globally for more than 25 years. "
    "5+ years relevant experience. "
    "5 plus years' experience with Python."
)


class RequiredYearsTest(unittest.TestCase):
    def setUp(self):
        self.processor = TextProcessor()

    def test_paypal_company_history_does_not_set_the_bar(self):
        self.assertEqual(self.processor.extract_required_years(PAYPAL), 5)

    def test_overall_bar_beats_an_included_specialty(self):
        text = "8 years software development, including 3 years Java"
        self.assertEqual(self.processor.extract_required_years(text), 8)

    def test_alternatives_qualify_at_the_lower_bar(self):
        text = "5 years of Java or 3 years of Go"
        self.assertEqual(self.processor.extract_required_years(text), 3)

    def test_preferred_years_do_not_raise_the_mandatory_bar(self):
        text = "5+ years of experience required. 8+ years preferred."
        self.assertEqual(self.processor.extract_required_years(text), 5)

    def test_preferred_section_is_not_the_job_bar(self):
        text = (
            "Required Qualifications\n"
            "5+ years of experience\n"
            "Preferred Qualifications\n"
            "10+ years of experience\n"
        )
        chunker = DocumentChunker()
        chunks = chunker.chunk_job({
            "title": "Engineer",
            "company_name": "PayPal",
            "description": text,
            "requirements": "",
        })
        self.assertEqual(chunks.years_required, 5)

    def test_company_history_alone_is_none(self):
        text = "PayPal has been revolutionizing commerce globally for more than 25 years."
        self.assertIsNone(self.processor.extract_required_years(text))

    def test_profile_extractor_is_unchanged(self):
        self.assertEqual(self.processor.extract_years_experience("15 years of experience"), 15)
        chunker = DocumentChunker()
        profile = chunker.chunk_profile({
            "years_of_experience": 15,
            "summary": "3 years of experience",
        })
        self.assertEqual(profile.years_experience, 15)


if __name__ == "__main__":
    unittest.main()
