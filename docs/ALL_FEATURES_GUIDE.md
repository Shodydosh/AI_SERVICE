# All Features Guide - Hướng Dẫn Đầy Đủ

## 📋 Tổng Quan Tất Cả Features

Hệ thống matching đã được cải tiến với **11 tính năng chính**:

### Core Matching Features
1. ✅ **Hybrid Search** - Semantic + Keyword matching
2. ✅ **Reranking** - Cross-encoder reranking
3. ✅ **Dynamic Filtering** - Auto-adjust filter sizes
4. ✅ **Contextual Embeddings** - Composite embeddings
5. ✅ **Negative Signals** - Deal-breakers & penalties
6. ✅ **Caching** - Redis/in-memory caching

### New Features
7. ✅ **Explainability** - Giải thích tại sao match
8. ✅ **Diversity & Fairness** - Diverse results, fairness metrics
9. ✅ **Multi-Criteria Optimization** - Pareto optimization
10. ✅ **Metrics Dashboard** - Tracking & monitoring
11. ✅ **A/B Testing** - Feature flags & experimentation

## 🚀 Sử Dụng

### Full Featured Service

```python
from src.services.enhanced_matching_with_all_features import EnhancedMatchingWithAllFeatures
from src.database.connection import get_db

db = next(get_db())

service = EnhancedMatchingWithAllFeatures(
    db=db,
    # Core features
    use_hybrid_search=True,
    use_reranking=True,
    use_dynamic_filtering=True,
    use_contextual_embeddings=True,
    use_negative_signals=True,
    use_caching=True,
    # New features
    use_explainability=True,
    use_diversity_fairness=True,
    use_multi_criteria=True,
    use_metrics=True,
    use_ab_testing=True
)

# Find jobs với explanations
response = service.find_jobs_for_candidate(
    candidate_id="15001",
    top_k=10,
    explain=True,
    ensure_diversity=True,
    use_pareto=True
)

# Results
results = response['results']
explanations = response['metadata']['explanations']
diversity_metrics = response['metadata']['diversity_metrics']
```

### Explainability

```python
from src.services.explainability_service import ExplainabilityService

explainer = ExplainabilityService()

explanation = explainer.explain_match(
    candidate=candidate_data,
    job=job_data,
    score=0.75,
    field_similarities={
        'skills': 0.85,
        'experience': 0.70,
        'title': 0.90
    }
)

print(explanation['why_recommended'])
print(explanation['matched_skills'])
print(explanation['missing_skills'])
```

### Diversity & Fairness

```python
from src.services.diversity_fairness_service import DiversityFairnessService

diversity_service = DiversityFairnessService()

diverse_results, metrics = diversity_service.apply_diversity_fairness(
    results=results,
    embeddings=job_embeddings,
    top_k=10
)

print(f"Diversity score: {metrics['diversity_score']}")
print(f"Fairness metrics: {metrics['fairness_metrics']}")
```

### Multi-Criteria Optimization

```python
from src.services.multi_criteria_optimization_service import MultiCriteriaOptimizationService

optimizer = MultiCriteriaOptimizationService()

optimized = optimizer.optimize_multi_criteria(
    candidate=candidate_data,
    matches=results,
    job_data_dict=job_data_dict,
    top_k=10,
    use_pareto=True
)
```

### Metrics Dashboard

```python
from src.services.metrics_dashboard_service import MetricsDashboardService

metrics = MetricsDashboardService()

# Track accuracy
metrics.track_accuracy_metrics(
    recommendations=results,
    ground_truth=['job1', 'job2'],
    k=10
)

# Track latency
metrics.track_latency('matching', 150.5)

# Track engagement
metrics.track_user_engagement(
    candidate_id="15001",
    job_id="job1",
    action="click"
)

# Get dashboard
dashboard = metrics.get_all_metrics()
print(dashboard['latency']['p95'])
print(dashboard['engagement']['ctr'])
```

### A/B Testing

```python
from src.services.ab_testing_service import ABTestingService

ab_service = ABTestingService()

# Register experiment
ab_service.register_experiment(
    experiment_name='hybrid_search',
    control_func=current_pipeline,
    variant_func=new_hybrid_pipeline,
    split_ratio=0.5
)

# Run experiment
result = ab_service.run_experiment(
    experiment_name='hybrid_search',
    user_id="15001",
    candidate_id="15001"
)

# Get metrics
metrics = ab_service.get_experiment_metrics('hybrid_search')
print(f"Improvement: {metrics['improvement']*100:.2f}%")
```

## 📊 Response Format

```python
{
    'results': [
        {
            'job_id': 'job1',
            'title': 'Software Engineer',
            'similarity_score': 0.85,
            'field_similarities': {
                'title': 0.90,
                'skills': 0.85,
                'experience': 0.70
            },
            'objective_scores': [0.85, 0.70, 0.80, 0.75],  # Multi-criteria
            'negative_signals': {
                'total_penalty': 0.05
            }
        }
    ],
    'metadata': {
        'candidate_id': '15001',
        'total_results': 10,
        'latency_ms': 150.5,
        'diversity_metrics': {
            'diversity_score': 0.65
        },
        'explanations': [
            {
                'job_id': 'job1',
                'overall_score': 0.85,
                'breakdown': {
                    'skills_match': 0.85,
                    'experience_fit': 0.70,
                    'title_relevance': 0.90
                },
                'matched_skills': ['Python', 'FastAPI'],
                'missing_skills': ['AWS'],
                'why_recommended': 'Strong skills alignment...'
            }
        ]
    }
}
```

## ⚙️ Configuration

Tất cả features có thể bật/tắt độc lập:

```python
service = EnhancedMatchingWithAllFeatures(
    db=db,
    # Core
    use_hybrid_search=True,
    use_reranking=False,  # Tắt reranking
    # New
    use_explainability=True,
    use_diversity_fairness=False,  # Tắt diversity
    use_multi_criteria=True,
    use_metrics=True,
    use_ab_testing=False  # Tắt A/B testing
)
```

## 📈 Expected Improvements

- **Accuracy**: +20-25% với tất cả features
- **User Experience**: +30% với explainability
- **Fairness**: +15% với diversity & fairness
- **Performance**: +40% với caching
- **Insights**: Metrics dashboard cho monitoring

## 🔄 Migration

1. Test từng feature một
2. Monitor metrics
3. Gradually enable features
4. A/B test để so sánh

