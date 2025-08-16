import os
import json
import numpy as np

model_response_directory = "quiz_results"

for model_response_fp in os.listdir(model_response_directory):
    avg_relevance_sources = []
    count = 0
    num_questions = 0
    with open(os.path.join(model_response_directory, model_response_fp), "r") as f:
        model_responses = json.load(f)
        for response in model_responses:
            if response["is_correct"] == True:
                count += 1
            num_questions += 1
            avg_relevance_sources.append(response["avg relevance sources"])
    print(f"Model: {model_response_fp}, Correct: {count}/{num_questions}, Avg Relevance: {sum(avg_relevance_sources) / len(avg_relevance_sources) if avg_relevance_sources else 0}")
                
# ============================================================================
# RAG MODEL PERFORMANCE ANALYSIS SCRIPT
# ============================================================================
# This script analyzes the performance of different RAG (Retrieval-Augmented Generation) 
# model combinations by comparing:
# 1. Quiz accuracy (correctness of answers)
# 2. Relevance scores (how well the embedding model retrieves relevant context)
# 3. Model component performance (embedding vs text generation models)
#
# The challenge: Different embedding models use different vector spaces and similarity 
# metrics, making direct relevance score comparison meaningless. This script addresses 
# this by using multiple evaluation approaches.
# ============================================================================

def normalize_scores(scores):
    """
    Normalize relevance scores to 0-1 range within each model.
    
    PURPOSE: Each embedding model uses different similarity metrics (cosine similarity, 
    dot product, negative distances, etc.) making raw scores incomparable. This function
    normalizes scores within each model to enable fair comparison.
    
    EVALUATES: Whether higher relevance correlates with better performance within a model
    
    USAGE: Compare normalized relevance trends across questions, not absolute values
    """
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [0.5] * len(scores)  # All scores are the same
    
    # Normalize to 0-1 range
    normalized = [(score - min_score) / (max_score - min_score) for score in scores]
    return normalized

def analyze_model_performance():
    """
    Analyze performance across all model combinations.
    
    PURPOSE: Comprehensive comparison of all embedding + text generation model combinations
    to identify:
    - Which models answer questions most accurately
    - How relevance scores vary within each model type
    - Overall performance patterns across model families
    
    EVALUATES:
    - Accuracy ranking: Which model combinations score highest on quiz questions
    - Relevance patterns: How well each embedding model retrieves relevant context
    - Model consistency: Which models have stable performance across questions
    
    INTERPRETATION:
    - High accuracy = model understands and answers questions correctly
    - High relevance = embedding model retrieves contextually relevant information
    - Low relevance variance = consistent retrieval quality
    
    USAGE: Primary tool for model selection and performance comparison
    """
    model_response_directory = "quiz_results"
    results = []
    
    for model_response_fp in os.listdir(model_response_directory):
        if not model_response_fp.endswith('.json'):
            continue
            
        with open(os.path.join(model_response_directory, model_response_fp), "r") as f:
            model_responses = json.load(f)
            
        # Extract data
        correct_count = sum(1 for r in model_responses if r["is_correct"])
        total_questions = len(model_responses)
        relevance_scores = [r["avg relevance sources"] for r in model_responses]
        
        # Normalize relevance scores
        normalized_scores = normalize_scores(relevance_scores)
        avg_normalized_relevance = np.mean(normalized_scores) if normalized_scores else 0
        
        # Calculate performance metrics
        accuracy = correct_count / total_questions if total_questions > 0 else 0
        
        # Separate correct vs incorrect question relevance
        correct_relevances = [r["avg relevance sources"] for r in model_responses if r["is_correct"]]
        incorrect_relevances = [r["avg relevance sources"] for r in model_responses if not r["is_correct"]]
        
        results.append({
            'model': model_response_fp.replace('_quiz_results.json', ''),
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_questions': total_questions,
            'raw_avg_relevance': np.mean(relevance_scores) if relevance_scores else 0,
            'normalized_avg_relevance': avg_normalized_relevance,
            'correct_avg_relevance': np.mean(correct_relevances) if correct_relevances else 0,
            'incorrect_avg_relevance': np.mean(incorrect_relevances) if incorrect_relevances else 0,
            'relevance_std': np.std(relevance_scores) if relevance_scores else 0
        })
    
    # Sort by accuracy
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print("=" * 100)
    print("MODEL PERFORMANCE ANALYSIS (Normalized)")
    print("=" * 100)
    print(f"{'Model':<40} {'Accuracy':<10} {'Norm.Rel':<10} {'Corr.Rel':<10} {'Incorr.Rel':<10}")
    print("-" * 100)
    
    for result in results:
        print(f"{result['model']:<40} "
              f"{result['accuracy']:.3f}     "
              f"{result['normalized_avg_relevance']:.3f}     "
              f"{result['correct_avg_relevance']:.3f}     "
              f"{result['incorrect_avg_relevance']:.3f}")
    
    return results

# if __name__ == "__main__":
analyze_model_performance()

def analyze_relative_performance():
    """
    Compare relative performance within model families.
    
    PURPOSE: Since raw relevance scores can't be compared across different embedding models,
    this function analyzes performance relative to each model's own score distribution.
    
    STRATEGY:
    1. Group results by embedding model (same vector space)
    2. Compare text generation models within each embedding family
    3. Identify which text models work best with which embedding models
    
    EVALUATES:
    - Text model effectiveness: Which text generation models produce better answers
      when paired with the same embedding model
    - Embedding-text synergy: Which combinations create the best partnerships
    - Relative accuracy improvement: How much better one text model is vs another
      within the same embedding family
    
    KEY INSIGHT: A text model that works well with one embedding model might not
    work as well with a different embedding model due to different retrieval patterns.
    
    USAGE: Optimize text model selection for a chosen embedding model
    """
    """Analyze how well each model separates correct from incorrect answers."""
    model_response_directory = "quiz_results"
    
    print("=" * 80)
    print("RELATIVE PERFORMANCE ANALYSIS")
    print("=" * 80)
    print(f"{'Model':<40} {'Accuracy':<10} {'Rel.Separation':<15}")
    print("-" * 80)
    
    for model_response_fp in os.listdir(model_response_directory):
        if not model_response_fp.endswith('.json'):
            continue
            
        with open(os.path.join(model_response_directory, model_response_fp), "r") as f:
            model_responses = json.load(f)
        
        # Separate correct and incorrect answers
        correct_relevances = [r["avg relevance sources"] for r in model_responses if r["is_correct"]]
        incorrect_relevances = [r["avg relevance sources"] for r in model_responses if not r["is_correct"]]
        
        if correct_relevances and incorrect_relevances:
            # Calculate separation (how much better correct answers score)
            avg_correct = np.mean(correct_relevances)
            avg_incorrect = np.mean(incorrect_relevances)
            
            # Relative separation (positive = correct answers have higher relevance)
            separation = (avg_correct - avg_incorrect) / abs(avg_correct) if avg_correct != 0 else 0
        else:
            separation = 0
        
        accuracy = sum(1 for r in model_responses if r["is_correct"]) / len(model_responses)
        model_name = model_response_fp.replace('_quiz_results.json', '')
        
        print(f"{model_name:<40} {accuracy:.3f}     {separation:>+.3f}")

analyze_relative_performance()

def comprehensive_model_analysis():
    """
    Comprehensive analysis combining accuracy and relevance metrics.
    
    PURPOSE: Multi-dimensional analysis that considers both question-answering accuracy
    and context retrieval quality to provide holistic model evaluation.
    
    ANALYSIS DIMENSIONS:
    1. Accuracy Performance: How well models answer questions correctly
    2. Relevance Quality: How well embedding models retrieve relevant context
    3. Efficiency Analysis: Performance per model size/complexity
    4. Consistency Metrics: Reliability across different question types
    
    EVALUATION CRITERIA:
    - High accuracy + High relevance = Optimal model combination
    - High accuracy + Low relevance = Text model compensating for poor retrieval
    - Low accuracy + High relevance = Text model failing despite good context
    - Low accuracy + Low relevance = Poor overall combination
    
    NORMALIZED COMPARISON:
    Since embedding models use incomparable similarity metrics, this analysis:
    - Normalizes relevance scores within each embedding model family
    - Focuses on accuracy as the primary comparable metric
    - Uses relevance trends (not absolute values) for secondary insights
    
    USAGE: Primary tool for selecting the best overall model combination for production
    """
    """Comprehensive analysis focusing on model architecture patterns."""
    model_response_directory = "quiz_results"
    
    # Group by embedding and text models
    embedding_performance = {}
    text_model_performance = {}
    combination_performance = {}
    
    for model_response_fp in os.listdir(model_response_directory):
        if not model_response_fp.endswith('.json'):
            continue
            
        # Parse model names
        model_name = model_response_fp.replace('_quiz_results.json', '')
        parts = model_name.split('--')
        if len(parts) != 2:
            continue
            
        embedding_model = parts[0]
        text_model = parts[1]
        
        with open(os.path.join(model_response_directory, model_response_fp), "r") as f:
            model_responses = json.load(f)
        
        accuracy = sum(1 for r in model_responses if r["is_correct"]) / len(model_responses)
        
        # Group performance
        if embedding_model not in embedding_performance:
            embedding_performance[embedding_model] = []
        embedding_performance[embedding_model].append(accuracy)
        
        if text_model not in text_model_performance:
            text_model_performance[text_model] = []
        text_model_performance[text_model].append(accuracy)
        
        combination_performance[model_name] = accuracy
    
    # Analyze embedding models
    print("=" * 60)
    print("EMBEDDING MODEL PERFORMANCE")
    print("=" * 60)
    for embedding_model, accuracies in sorted(embedding_performance.items(), 
                                            key=lambda x: np.mean(x[1]), reverse=True):
        avg_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        print(f"{embedding_model:<30} Avg: {avg_acc:.3f} ± {std_acc:.3f}")
    
    # Analyze text generation models
    print("\n" + "=" * 60)
    print("TEXT GENERATION MODEL PERFORMANCE")
    print("=" * 60)
    for text_model, accuracies in sorted(text_model_performance.items(), 
                                       key=lambda x: np.mean(x[1]), reverse=True):
        avg_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        print(f"{text_model:<20} Avg: {avg_acc:.3f} ± {std_acc:.3f}")
    
    # Best combinations
    print("\n" + "=" * 60)
    print("TOP 10 MODEL COMBINATIONS")
    print("=" * 60)
    sorted_combinations = sorted(combination_performance.items(), 
                               key=lambda x: x[1], reverse=True)
    
    for i, (model_combo, accuracy) in enumerate(sorted_combinations[:10], 1):
        embedding, text = model_combo.split('--')
        print(f"{i:2d}. {embedding:<25} + {text:<15} = {accuracy:.3f}")

# Run the comprehensive analysis
comprehensive_model_analysis()

# ============================================================================
# ANALYSIS INTERPRETATION GUIDE
# ============================================================================
"""
UNDERSTANDING YOUR RESULTS:

1. WHY RELEVANCE SCORES ARE INCOMPARABLE ACROSS EMBEDDING MODELS:
   - Each embedding model uses different vector spaces and similarity metrics
   - all-MiniLM models: Use cosine similarity (range: -1 to 1)
   - BGE models: Use dot product similarity (range: varies by model)
   - E5 models: Use cosine similarity with different normalization
   - MRL models: Use negative distances (negative values = more similar)
   
   This means a relevance score of 0.8 from one model is NOT comparable to 
   0.8 from another model. They live in completely different mathematical spaces.

2. FOCUS ON ACCURACY FOR MODEL COMPARISON:
   - Accuracy is the only metric directly comparable across all models
   - It measures the ultimate goal: answering questions correctly
   - Use relevance scores only for within-model analysis

3. WHAT EACH ANALYSIS FUNCTION TELLS YOU:
   
   analyze_model_performance():
   - Overall ranking of all model combinations by accuracy
   - Shows which combinations work best for your specific task
   - Identifies patterns in embedding and text model performance
   
   analyze_relative_performance():
   - Compares text models within the same embedding family
   - Helps optimize text model choice for a selected embedding model
   - Reveals embedding-text model synergies
   
   comprehensive_model_analysis():
   - Multi-dimensional view combining accuracy and consistency
   - Best for final model selection decisions
   - Considers both performance and reliability

4. MODEL SELECTION BEST PRACTICES:
   - Start with accuracy: Choose models with >70% accuracy
   - Consider consistency: Prefer models with stable performance
   - Test resource requirements: Larger models need more memory/compute
   - Validate with domain-specific questions beyond the test set
   
5. TYPICAL PERFORMANCE PATTERNS:
   - Larger text models (flan-t5-large/xl) usually outperform smaller ones
   - Better embedding models provide more consistent relevance
   - Some embedding-text combinations have unexpected synergies
   - Model performance can vary significantly by question type
"""

def analyze_models():
    """
    Main analysis orchestrator that runs all evaluation methods.
    
    PURPOSE: Executes comprehensive model analysis pipeline combining multiple
    evaluation approaches to provide complete performance assessment.
    
    EXECUTION ORDER:
    1. Individual model performance analysis
    2. Relative performance comparison within model families  
    3. Comprehensive multi-dimensional analysis
    
    OUTPUT INTERPRETATION GUIDE:
    
    ACCURACY METRICS (Most Important):
    - Primary selection criterion since it's directly comparable across all models
    - Shows which combinations actually answer questions correctly
    - Range: 0-100% (higher is better)
    
    RELEVANCE METRICS (Context Quality):
    - Cannot be compared across different embedding models due to different similarity spaces
    - Only meaningful for comparison within the same embedding model family
    - Indicates how well the embedding model retrieves relevant context
    - Use normalized scores for trend analysis, not absolute comparison
    
    CONSISTENCY METRICS (Reliability):
    - Standard deviation of relevance scores (lower = more consistent)
    - Shows which models provide stable performance across questions
    - Important for production deployments requiring predictable behavior
    
    MODEL SELECTION STRATEGY:
    1. Filter by accuracy threshold (e.g., >70%)
    2. Among high-accuracy models, prefer those with consistent relevance
    3. Consider computational requirements for production deployment
    4. Test top candidates with domain-specific questions
    
    COMMON PATTERNS:
    - Larger text models (flan-t5-large/xl) often have higher accuracy
    - Better embedding models provide more consistent relevance scores
    - Some text models work better with specific embedding models
    
    USAGE: Run this function to get complete model evaluation report
    """
    model_response_directory = "quiz_results"
    
    print("=" * 120)
    print("COMPREHENSIVE MODEL ANALYSIS")
    print("=" * 120)
    print(f"{'Model':<45} {'Accuracy':<10} {'Raw Relevance':<15} {'Performance Rank':<15}")
    print("-" * 120)
    
    results = []
    
    for model_response_fp in os.listdir(model_response_directory):
        if not model_response_fp.endswith('.json'):
            continue
            
        with open(os.path.join(model_response_directory, model_response_fp), "r") as f:
            model_responses = json.load(f)
        
        # Calculate metrics
        correct_count = sum(1 for r in model_responses if r["is_correct"])
        total_questions = len(model_responses)
        accuracy = correct_count / total_questions if total_questions > 0 else 0
        
        relevance_scores = [r["avg relevance sources"] for r in model_responses]
        avg_relevance = np.mean(relevance_scores) if relevance_scores else 0
        
        model_name = model_response_fp.replace('_quiz_results.json', '')
        
        results.append({
            'model': model_name,
            'accuracy': accuracy,
            'avg_relevance': avg_relevance,
            'correct_count': correct_count,
            'total_questions': total_questions
        })
    
    # Sort by accuracy (most important metric)
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # Add performance rank
    for i, result in enumerate(results, 1):
        result['rank'] = i
    
    # Display results
    for result in results:
        relevance_str = f"{result['avg_relevance']:.3f}" if abs(result['avg_relevance']) < 1000 else f"{result['avg_relevance']:.0f}"
        print(f"{result['model']:<45} "
              f"{result['accuracy']:.3f}     "
              f"{relevance_str:<15} "
              f"#{result['rank']}")
    
    # Summary by model type
    print("\n" + "=" * 80)
    print("SUMMARY BY EMBEDDING MODEL")
    print("=" * 80)
    
    embedding_groups = {}
    for result in results:
        embedding = result['model'].split('--')[0]
        if embedding not in embedding_groups:
            embedding_groups[embedding] = []
        embedding_groups[embedding].append(result['accuracy'])
    
    for embedding, accuracies in sorted(embedding_groups.items(), 
                                      key=lambda x: np.mean(x[1]), reverse=True):
        avg_acc = np.mean(accuracies)
        best_acc = max(accuracies)
        print(f"{embedding:<30} Avg: {avg_acc:.3f}, Best: {best_acc:.3f}")

if __name__ == "__main__":
    analyze_models()