# Job Matcher ML Service

A local ML service for intelligent job-profile matching using NLP and embeddings.

## Features

- **Tokenization & Lemmatization**: Uses spaCy for text processing
- **Semantic Embeddings**: sentence-transformers for dense vector representations
- **Multi-component Scoring**: Skills, experience, semantic, and requirements matching
- **Recommendations**: Strong Apply, Good Fit, Maybe, or Weak Match

## Scoring Weights

| Component | Weight | Description |
|-----------|--------|-------------|
| Skill Match | 40% | Exact + semantic skill matching |
| Experience Match | 30% | Role similarity + years alignment |
| Semantic Match | 20% | Overall resume ↔ job similarity |
| Requirements Fit | 10% | Education, certifications match |

## Quick Start

### Option 1: Local (Recommended for Development)

```bash
# Navigate to service directory
cd job-matcher-service

# Make run script executable
chmod +x run.sh

# Run (creates venv, installs deps, starts server)
./run.sh
```

### Option 2: Docker

```bash
# Build
docker build -t job-matcher-service .

# Run
docker run -p 5000:5000 job-matcher-service
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/embed/profile` | Embed user profile |
| POST | `/api/v1/embed/job` | Embed job posting |
| POST | `/api/v1/match` | Match profile against job |
| POST | `/api/v1/match/inline` | Match without storing |
| POST | `/api/v1/match/batch` | Match against multiple jobs |
| GET | `/api/v1/match/similar-jobs/{id}` | Find best matching jobs |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |

## API Usage

### 1. Embed a Profile

```bash
curl -X POST http://localhost:5000/api/v1/embed/profile \
  -H "Content-Type: application/json" \
  -d '{
    "profile_id": 1,
    "summary": "15+ years Java experience...",
    "skills": [
      {"name": "Java", "years": 15, "proficiency": "EXPERT"},
      {"name": "Spring Boot", "years": 10, "proficiency": "EXPERT"}
    ],
    "experiences": [
      {"company_name": "AppDynamics", "title": "Staff Engineer", "description": "..."}
    ],
    "years_of_experience": 15
  }'
```

### 2. Embed a Job

```bash
curl -X POST http://localhost:5000/api/v1/embed/job \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": 101,
    "title": "Senior Software Engineer",
    "company_name": "Walmart",
    "description": "Build microservices with Java and Spring Boot...",
    "requirements": "5+ years Java, Kafka, Kubernetes...",
    "skills_required": "Java, Spring Boot, Kafka"
  }'
```

### 3. Match Profile to Job

```bash
curl -X POST http://localhost:5000/api/v1/match \
  -H "Content-Type: application/json" \
  -d '{"profile_id": 1, "job_id": 101}'
```

### Response Example

```json
{
  "profile_id": 1,
  "job_id": 101,
  "score": 87,
  "recommendation": "strong_apply",
  "breakdown": {
    "skill_match": 92,
    "experience_match": 85,
    "semantic_match": 82,
    "requirements_fit": 78
  },
  "matched_skills": ["java", "spring boot", "kafka", "kubernetes"],
  "missing_skills": ["mongodb"],
  "strengths": [
    "Strong skill match: java, spring boot, kafka",
    "15+ years of experience",
    "Highly relevant experience background"
  ],
  "gaps": [
    "Missing skills: mongodb"
  ],
  "processing_time_ms": 45
}
```

## Testing

```bash
# Run test suite
python -m app.tests.test_matcher
```

## Configuration

Edit `app/config.py` or set environment variables:

```bash
# Scoring weights
export WEIGHT_SKILL_MATCH=0.40
export WEIGHT_EXPERIENCE_MATCH=0.30
export WEIGHT_SEMANTIC_MATCH=0.20
export WEIGHT_REQUIREMENTS_FIT=0.10

# Thresholds
export THRESHOLD_STRONG_APPLY=85
export THRESHOLD_GOOD_FIT=70
export THRESHOLD_MAYBE=55

# Models
export EMBEDDING_MODEL=all-MiniLM-L6-v2
export SPACY_MODEL=en_core_web_sm
```

## Requirements

- Python 3.9+
- ~2GB RAM for models
- ~500MB disk for sentence-transformers model
- ~80MB disk for spaCy model

## Integration with Job-Assist (Java)

Add to `application.yml`:

```yaml
scoring:
  tier2:
    provider: embedding
    embedding-service-url: http://localhost:5000
```

Then update `TieredScoringService.java` to call this service for Tier 2 scoring.
