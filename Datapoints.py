import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import textwrap

# Load the transformed data
df = pd.read_csv('transformed_quiz_data.csv')

def calculate_overall_accuracy(df):
    attempt_groups = df.groupby(['user_id', 'quiz_id', 'quiz_attempts'])
    perfect_attempts = attempt_groups['is_correct'].agg(lambda x: x.all()).reset_index()
    accuracy_rate = perfect_attempts['is_correct'].mean() * 100
    return round(accuracy_rate, 2)

def calculate_question_difficulty(df):
    question_stats = df.groupby(['question_id', 'question_text'])['is_correct'].agg(
        correct_rate='mean',
        total_attempts='count'
    ).reset_index()
    most_missed = question_stats.loc[question_stats['correct_rate'].idxmin()]
    easiest = question_stats.loc[question_stats['correct_rate'].idxmax()]
    return most_missed, easiest

def calculate_average_score(df):
    unique_attempts = df.drop_duplicates(subset=['user_id', 'quiz_id', 'quiz_attempts'])
    return round(unique_attempts['total_score'].mean(), 2)

def calculate_perfect_scorers(df):
    perfect_attempts = df.groupby(['user_id', 'quiz_id', 'quiz_attempts'])['is_correct'].all().reset_index()
    perfect_attempts = perfect_attempts[perfect_attempts['is_correct']]
    return perfect_attempts['user_id'].nunique()

def calculate_zero_scorers(df):
    attempt_groups = df.groupby(['user_id', 'quiz_id', 'quiz_attempts'])
    zero_attempts = attempt_groups['is_correct'].agg(lambda x: not x.any()).reset_index()
    zero_scorers = zero_attempts[zero_attempts['is_correct']]['user_id'].nunique()
    total_students = df['user_id'].nunique()
    return zero_scorers, total_students

def calculate_partial_correctness(df):
    multi_select_mask = df['correct_answer'].str.contains(r'\{\{.*\}\}', na=False)
    multi_select_df = df[multi_select_mask].copy()
    
    def is_partial_correct(row):
        if not pd.isna(row['selected_option']) and not pd.isna(row['correct_answer']):
            selected = set([s.strip() for s in row['selected_option'].split('{{}}')])
            correct = set([c.strip() for c in row['correct_answer'].split('{{}}')])
            return bool(selected & correct) and not (selected == correct)
        return False
    
    multi_select_df['partial_correct'] = multi_select_df.apply(is_partial_correct, axis=1)
    student_partial = multi_select_df.groupby('user_id')['partial_correct'].any()
    partial_percentage = (student_partial.mean() * 100) if not student_partial.empty else 0
    return round(partial_percentage, 2), multi_select_df.shape[0]

def plot_score_distribution(df):
    unique_attempts = df.drop_duplicates(subset=['user_id', 'quiz_id', 'quiz_attempts'])
    plt.figure(figsize=(12, 6))
    sns.histplot(data=unique_attempts, x='total_score', bins=11, discrete=True, kde=False)
    plt.title('Distribution of Quiz Scores', fontsize=16)
    plt.xlabel('Total Score (out of 10)', fontsize=12)
    plt.ylabel('Number of Attempts', fontsize=12)
    plt.xticks(range(0, 11))
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()
    return unique_attempts['total_score'].value_counts().sort_index().reindex(range(0,11), fill_value=0)

def analyze_difficulty_mismatch(df):
    difficulty_stats = df.groupby('difficulty_level')['is_correct'].agg(
        mean_accuracy='mean',
        question_count='count'
    ).reset_index()
    
    difficulty_order = ['EASY', 'MEDIUM', 'HARD']
    if set(difficulty_stats['difficulty_level']).issuperset(difficulty_order):
        difficulty_stats['difficulty_level'] = pd.Categorical(
            difficulty_stats['difficulty_level'], 
            categories=difficulty_order,
            ordered=True
        )
        difficulty_stats = difficulty_stats.sort_values('difficulty_level')
    return difficulty_stats

def find_polarizing_question(df):
    question_stats = df.groupby(['question_id', 'question_text'])['is_correct'].agg(
        correct_rate='mean',
        total_attempts='count'
    ).reset_index()
    question_stats['distance_from_50'] = abs(question_stats['correct_rate'] - 0.5)
    polarizing_question = question_stats.loc[question_stats['distance_from_50'].idxmin()]
    polarizing_question['correct_percentage'] = round(polarizing_question['correct_rate'] * 100, 2)
    polarizing_question['incorrect_percentage'] = 100 - polarizing_question['correct_percentage']
    return polarizing_question

def analyze_question_specific(df):
    insights = {}
    
    # Q1: Workbook Completion
    q1 = df[df['question_id'] == 1]    
    q1_truthful = q1[q1['is_correct']].shape[0] / q1.shape[0] * 100
    insights['q1_truthful'] = round(q1_truthful, 2)
    
    # Q2: Global Warming Effects
    q2 = df[df['question_id'] == 2]
    options = ["Increase in disasters like floods", "Poor air quality and health issues",
               "Increase in heat and forest fires", "Damages crops and increase in food scarcity"]
    missed_counts = {opt: 0 for opt in options}
    
    for _, row in q2.iterrows():
        selected = set([s.strip() for s in row['selected_option'].split('{{}}')])
        correct = set([c.strip() for c in row['correct_answer'].split('{{}}')])
        missed = correct - selected
        for opt in missed:
            if opt in missed_counts:
                missed_counts[opt] += 1
    
    insights['q2_missed'] = max(missed_counts, key=missed_counts.get)
    
    # Q3: SDG Innovations
    q3 = df[df['question_id'] == 3]
    incorrect_options = ["Remote controlled fan with lights", "Eco-Friendly wall paint that absorbs pollution",
                         "A device that stops too much watering of fields"]
    freq = q3['selected_option'].value_counts().to_dict()
    for opt in incorrect_options[1:]:
        if opt in freq:
            del freq[opt]
    insights['q3_incorrect'] = max(freq, key=freq.get) if freq else "N/A"
    
    # Q4: Multipurpose Bags
    q4 = df[df['question_id'] == 4]
    insights['q4_correct'] = round(q4['is_correct'].mean() * 100, 2)
    
    # Q5: SDG for Drones
    q5 = df[df['question_id'] == 5]
    insights['q5_correct'] = round(q5['is_correct'].mean() * 100, 2)
    
    # Q6: Solar School Bags
    q6 = df[df['question_id'] == 6]
    features = ["Uses solar energy", "Bag is made of old used plastic"]
    missed_counts = {feat: 0 for feat in features}
    
    for _, row in q6.iterrows():
        selected = set([s.strip() for s in row['selected_option'].split('{{}}')])
        correct = set([c.strip() for c in row['correct_answer'].split('{{}}')])
        missed = correct - selected
        for feat in missed:
            if feat in missed_counts:
                missed_counts[feat] += 1
    
    insights['q6_missed'] = max(missed_counts, key=missed_counts.get)
    
    # Q7: Sustainability Definition
    q7 = df[df['question_id'] == 7]
    incorrect_responses = q7[~q7['is_correct']]['selected_option'].value_counts()
    insights['q7_misconception'] = incorrect_responses.idxmax() if not incorrect_responses.empty else "N/A"
    
    # Q8: SDG Importance
    q8 = df[df['question_id'] == 8]
    insights['q8_correct'] = round(q8['is_correct'].mean() * 100, 2)
    
    # Q9: Innovator Actions
    q9 = df[df['question_id'] == 9]
    option_counts = q9['selected_option'].value_counts(normalize=True) * 100
    insights['q9_understand'] = round(option_counts.get("Understand different problems in the community", 0), 2)
    
    # Q10: Social Innovation
    q10 = df[df['question_id'] == 10]
    incorrect_statements = []
    correct_options = set([
        "Social Innovation is a new idea to solve a problem faced by a group of people",
        "Social Innovations make people's lives better by reducing problems",
        "A device that can provide clean drinking water at a low cost is an example of Social Innovation"
    ])
    
    for _, row in q10.iterrows():
        selected = set([s.strip() for s in row['selected_option'].split('{{}}')])
        incorrect = selected - correct_options
        incorrect_statements.extend(incorrect)
    
    if incorrect_statements:
        insights['q10_incorrect'] = pd.Series(incorrect_statements).value_counts().idxmax()
    else:
        insights['q10_incorrect'] = "N/A"
    
    return insights

def analyze_attempt_behavior(df):
    insights = {}
    
    # 21. Average Attempts per Question
    insights['avg_question_attempts'] = round(df.groupby('question_id')['question_attempts'].mean().mean(), 2)
    
    # 22. First-Attempt Accuracy
    first_attempts = df[df['question_attempts'] == 1]
    subsequent_attempts = df[df['question_attempts'] > 1]
    
    first_accuracy = first_attempts['is_correct'].mean() * 100
    subsequent_accuracy = subsequent_attempts['is_correct'].mean() * 100 if not subsequent_attempts.empty else 0
    insights['first_attempt_accuracy'] = round(first_accuracy, 2)
    insights['subsequent_accuracy'] = round(subsequent_accuracy, 2)
    
    # 23. Most Re-attempted Question
    retry_counts = df[df['question_attempts'] > 1].groupby('question_id').size()
    insights['most_retried'] = retry_counts.idxmax() if not retry_counts.empty else "N/A"
    
    # 24. Retry Improvement
    insights['retry_improvement'] = round(subsequent_accuracy - first_accuracy, 2)
    
    # 25. Guessing Patterns (Most common incorrect options)
    incorrect_responses = df[~df['is_correct']]
    guessing_patterns = {}
    for qid in df['question_id'].unique():
        q_incorrect = incorrect_responses[incorrect_responses['question_id'] == qid]
        if not q_incorrect.empty:
            most_common = q_incorrect['selected_option'].value_counts().idxmax()
            guessing_patterns[qid] = most_common
    insights['guessing_patterns'] = guessing_patterns
    
    # 27. Drop-off Rate
    expected_questions = df.groupby(['user_id', 'quiz_id', 'quiz_attempts']).size()
    drop_off_rate = (expected_questions < 10).mean() * 100
    insights['drop_off_rate'] = round(drop_off_rate, 2)
    
    # 28. Most Skipped Question
    skipped = df[df['selected_option'].isna() | (df['selected_option'] == '')]
    insights['most_skipped'] = skipped['question_id'].value_counts().idxmax() if not skipped.empty else "N/A"
    
    return insights

def analyze_question_design(df):
    insights = {}
    
    # 36. Multi-Select vs Single-Select Difficulty
    multi_select_mask = df['correct_answer'].str.contains(r'\{\{.*\}\}', na=False)
    multi_select = df[multi_select_mask]
    single_select = df[~multi_select_mask]
    
    insights['multi_select_accuracy'] = round(multi_select['is_correct'].mean() * 100, 2)
    insights['single_select_accuracy'] = round(single_select['is_correct'].mean() * 100, 2)
    
    # 38. Question Wording Impact (Complexity vs Accuracy)
    df['question_length'] = df['question_text'].str.len()
    wording_impact = df.groupby('question_id').agg(
        length=('question_length', 'mean'),
        accuracy=('is_correct', 'mean')
    ).corr().iloc[0, 1]
    insights['wording_impact_corr'] = round(wording_impact, 3)
    
    # 39. Most Misleading Distractor
    misleading_distractors = {}
    for qid in df['question_id'].unique():
        q_data = df[df['question_id'] == qid]
        if q_data['is_correct'].mean() < 0.8:  # Only consider questions where many get it wrong
            incorrect = q_data[~q_data['is_correct']]
            if not incorrect.empty:
                most_common = incorrect['selected_option'].value_counts().idxmax()
                misleading_distractors[qid] = most_common
    insights['misleading_distractors'] = misleading_distractors
    
    # 40. Learning Curve Evidence
    learning_curve = df.groupby('question_number')['is_correct'].mean().reset_index()
    insights['learning_curve'] = learning_curve
    insights['learning_curve_corr'] = round(learning_curve['question_number'].corr(learning_curve['is_correct']), 3)
    
    return insights

def advanced_statistical_analysis(df):
    insights = {}
    
    # 41. Point-Biserial Correlation
    question_discrimination = {}
    for qid in df['question_id'].unique():
        q_data = df[df['question_id'] == qid]
        if not q_data.empty:
            point_biserial = stats.pointbiserialr(
                q_data['is_correct'],
                q_data['total_score'] - q_data['question_score']
            )
            question_discrimination[qid] = round(point_biserial[0], 3)
    insights['question_discrimination'] = question_discrimination
    
    # 42. Cronbach's Alpha
    def cronbach_alpha(items):
        items_count = items.shape[1]
        variance_sum = items.var(axis=0).sum()
        total_var = items.sum(axis=1).var()
        return (items_count / (items_count - 1)) * (1 - variance_sum / total_var)
    
    # Create a pivot table for responses
    response_matrix = df.pivot_table(
        index=['user_id', 'quiz_attempts'],
        columns='question_number',
        values='is_correct',
        fill_value=0
    )
    insights['cronbach_alpha'] = round(cronbach_alpha(response_matrix), 3)
    
    # 44. K-means Clustering
    scaler = StandardScaler()
    scaled_responses = scaler.fit_transform(response_matrix)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled_responses)
    response_matrix['cluster'] = clusters
    
    cluster_profiles = response_matrix.groupby('cluster').mean()
    insights['clusters'] = cluster_profiles
    
    return insights

def plot_performance_metrics(results):
    """Visualizations for performance metrics (1-10)"""
    # 1. Overall Accuracy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie([results['overall_accuracy'], 100 - results['overall_accuracy']],
           labels=['Perfect Attempts', 'Imperfect Attempts'],
           autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#F44336'])
    ax.set_title('Overall Perfect Attempt Rate')
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # 2/3. Question Difficulty
    question_stats = df.groupby('question_id')['is_correct'].mean().sort_values().reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(question_stats['question_id'].astype(str), 
                 question_stats['is_correct'] * 100,
                 color=np.where(question_stats['is_correct'] < 0.5, '#F44336', '#4CAF50'))
    ax.axhline(50, color='gray', linestyle='--', alpha=0.7)
    ax.set_title('Question Accuracy Rates')
    ax.set_xlabel('Question ID')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 100)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # 4. Average Score
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(['Average Score'], [results['avg_score']], color='#2196F3')
    ax.axhline(5, color='gray', linestyle='--', alpha=0.7)
    ax.set_title(f'Average Score: {results["avg_score"]}/10')
    ax.set_ylim(0, 10)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # 5/6. Scorers Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    scorers = pd.Series({
        'Perfect Scorers': results['perfect_scorers'],
        'Zero Scorers': results['zero_scorers'],
        'Other Students': results['total_students'] - results['perfect_scorers'] - results['zero_scorers']
    })
    scorers.plot(kind='bar', color=['#4CAF50', '#F44336', '#2196F3'], ax=ax)
    ax.set_title('Student Performance Categories')
    ax.set_ylabel('Number of Students')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # 7. Partial Correctness
    if results['partial_correct'] > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie([results['partial_correct'], 100 - results['partial_correct']],
               labels=['Partial Credit', 'No Partial Credit'],
               autopct='%1.1f%%', startangle=90, colors=['#FFC107', '#2196F3'])
        ax.set_title('Partial Correctness in Multi-Select Questions')
        plt.tight_layout()
        plt.show()
        plt.close()
    
    # 8. Score Distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    score_dist = results['score_distribution']
    score_dist.plot(kind='bar', color='#2196F3', ax=ax)
    ax.set_title('Distribution of Quiz Scores')
    ax.set_xlabel('Total Score (out of 10)')
    ax.set_ylabel('Number of Attempts')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # 9. Difficulty Effectiveness
    fig, ax = plt.subplots(figsize=(10, 6))
    results['difficulty_mismatch'].plot.bar(
        x='difficulty_level', 
        y='mean_accuracy', 
        color=['#4CAF50', '#FFC107', '#F44336'],
        ax=ax
    )
    ax.set_title('Accuracy by Difficulty Level')
    ax.set_xlabel('Difficulty Level')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # 10. Polarizing Question
    polar = results['polarizing_question']
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie([polar['correct_percentage'], polar['incorrect_percentage']],
           labels=['Correct', 'Incorrect'],
           autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#F44336'])
    ax.set_title(f'Polarizing Question (ID: {polar["question_id"]})')
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_question_specific(results):
    """Visualizations for question-specific insights (11-20)"""
    qs = results['question_specific']
    
    # Q1: Workbook Completion
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie([qs['q1_truthful'], 100 - qs['q1_truthful']],
           labels=['Admitted Finishing', 'Did Not Admit'],
           autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#F44336'])
    ax.set_title('Q1: Workbook Completion Truthfulness')
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # Q2: Global Warming Effects
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(['Most Missed Effect'], [1], color='#F44336')
    ax.text(0.5, 0, qs['q2_missed'], ha='center', va='center', fontsize=12)
    ax.set_title('Q2: Most Missed Global Warming Effect')
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # Q3: SDG Innovations
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(['Most Selected Incorrect'], [1], color='#F44336')
    ax.text(0.5, 0, qs['q3_incorrect'], ha='center', va='center', fontsize=12)
    ax.set_title('Q3: Most Selected Incorrect Innovation')
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # Q4: Multipurpose Bags
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(['Correct Answers'], [qs['q4_correct']], color='#4CAF50')
    ax.set_title(f'Q4: Multipurpose Bags\n{qs["q4_correct"]}% Correct')
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # Q5: SDG for Drones
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(['Correct Answers'], [qs['q5_correct']], color='#4CAF50')
    ax.set_title(f'Q5: SDG for Drones\n{qs["q5_correct"]}% Correct')
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # Q6: Solar Bags
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(['Most Overlooked Feature'], [1], color='#FFC107')
    ax.text(0.5, 0, qs['q6_missed'], ha='center', va='center', fontsize=12)
    ax.set_title('Q6: Most Overlooked Solar Bag Feature')
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # Q7: Sustainability Definition
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(['Common Misconception'], [1], color='#F44336')
    wrapped_text = textwrap.fill(qs['q7_misconception'], 40)
    ax.text(0.5, 0, wrapped_text, ha='center', va='center', fontsize=12)
    ax.set_title('Q7: Most Common Misconception')
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # Q8: SDG Importance
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(['Correct Answers'], [qs['q8_correct']], color='#4CAF50')
    ax.set_title(f'Q8: SDG Importance\n{qs["q8_correct"]}% Correct')
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # Q9: Innovator Actions
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie([qs['q9_understand'], 100 - qs['q9_understand']],
           labels=['Selected "Understand"', 'Other Options'],
           autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#9E9E9E'])
    ax.set_title('Q9: Innovator Actions Understanding')
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # Q10: Social Innovation
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(['Most Frequent Incorrect'], [1], color='#F44336')
    wrapped_text = textwrap.fill(qs['q10_incorrect'], 40)
    ax.text(0.5, 0, wrapped_text, ha='center', va='center', fontsize=12)
    ax.set_title('Q10: Most Frequent Incorrect Statement')
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_attempt_behavior(results):
    """Visualizations for attempt behavior (21-30)"""
    ab = results['attempt_behavior']
    
    # 21. Average Attempts per Question
    fig, ax = plt.subplots(figsize=(10, 6))
    attempts = df.groupby('question_id')['question_attempts'].mean()
    attempts.plot(kind='bar', color='#2196F3', ax=ax)
    ax.set_title('Average Attempts per Question')
    ax.set_xlabel('Question ID')
    ax.set_ylabel('Average Attempts')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # 22. First vs Subsequent Accuracy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(['First Attempt', 'Subsequent Attempts'], 
           [ab['first_attempt_accuracy'], ab['subsequent_accuracy']],
           color=['#2196F3', '#4CAF50'])
    ax.set_title('Accuracy by Attempt Type')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # 23. Most Re-attempted Question
    fig, ax = plt.subplots(figsize=(10, 6))
    retries = df[df['question_attempts'] > 1].groupby('question_id').size().sort_values(ascending=False)
    retries.head(5).plot(kind='bar', color='#FF9800', ax=ax)
    ax.set_title('Most Re-attempted Questions')
    ax.set_xlabel('Question ID')
    ax.set_ylabel('Number of Retries')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # 24. Retry Improvement
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(['Accuracy Improvement'], [ab['retry_improvement']], 
           color=('#4CAF50' if ab['retry_improvement'] > 0 else '#F44336'))
    ax.set_title(f'Accuracy Improvement After Retry: {ab["retry_improvement"]}%')
    ax.axhline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # 27. Drop-off Rate
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie([ab['drop_off_rate'], 100 - ab['drop_off_rate']],
           labels=['Incomplete Attempts', 'Complete Attempts'],
           autopct='%1.1f%%', startangle=90, colors=['#F44336', '#4CAF50'])
    ax.set_title('Quiz Completion Rate')
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # 28. Most Skipped Question
    fig, ax = plt.subplots(figsize=(10, 6))
    skipped = df[df['selected_option'].isna() | (df['selected_option'] == '')]
    skipped_counts = skipped['question_id'].value_counts().sort_values(ascending=False)
    
    # Add check for empty data
    if not skipped_counts.empty:
        skipped_counts.head(5).plot(kind='bar', color='#9C27B0', ax=ax)
        ax.set_title('Most Skipped Questions')
        ax.set_xlabel('Question ID')
        ax.set_ylabel('Number of Skips')
    else:
        ax.text(0.5, 0.5, 'No questions were skipped', 
                ha='center', va='center', fontsize=12)
        ax.set_title('Most Skipped Questions')
        ax.axis('off')
    
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_question_design(results):
    """Visualizations for question design (36-40)"""
    qd = results['question_design']
    
    # 36. Multi-Select vs Single-Select
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(['Multi-Select', 'Single-Select'], 
           [qd['multi_select_accuracy'], qd['single_select_accuracy']],
           color=['#FF9800', '#2196F3'])
    ax.set_title('Accuracy by Question Type')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # 38. Question Wording Impact
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(
        x='length', 
        y='accuracy', 
        data=df.groupby('question_id').agg(
            length=('question_length', 'mean'),
            accuracy=('is_correct', 'mean')
        ).reset_index(),
        ax=ax,
        scatter_kws={'color': '#2196F3'},
        line_kws={'color': '#F44336'}
    )
    ax.set_title('Question Length vs Accuracy')
    ax.set_xlabel('Question Length (Characters)')
    ax.set_ylabel('Accuracy')
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # 40. Learning Curve
    fig, ax = plt.subplots(figsize=(10, 6))
    qd['learning_curve'].plot(
        x='question_number', 
        y='is_correct', 
        marker='o',
        ax=ax,
        color='#4CAF50'
    )
    ax.set_title('Learning Curve (Accuracy by Question Order)')
    ax.set_xlabel('Question Number')
    ax.set_ylabel('Accuracy')
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_advanced_stats(results):
    """Visualizations for advanced stats (41-44)"""
    adv = results['advanced_stats']
    
    # 41. Question Discrimination
    fig, ax = plt.subplots(figsize=(12, 6))
    discrimination = pd.Series(adv['question_discrimination'])
    discrimination.plot(kind='bar', color=discrimination.map(
        lambda x: '#4CAF50' if x > 0.3 else '#FFC107' if x > 0.1 else '#F44336'
    ), ax=ax)
    ax.axhline(0.3, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(0.1, color='gray', linestyle='--', alpha=0.7)
    ax.set_title('Question Discrimination Index')
    ax.set_xlabel('Question ID')
    ax.set_ylabel('Point-Biserial Correlation')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    plt.close()
    
     # 44. Student Clusters - Modified to handle missing cluster column
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Check if cluster column exists before trying to drop it
    clusters_df = adv['clusters']
    if 'cluster' in clusters_df.columns:
        clusters_df = clusters_df.drop(columns='cluster')
    
    sns.heatmap(
        clusters_df,
        annot=True, 
        cmap='coolwarm',
        ax=ax
    )
    ax.set_title('Student Cluster Profiles (by Question Accuracy)')
    ax.set_xlabel('Question Number')
    ax.set_ylabel('Cluster')
    plt.tight_layout()
    plt.show()
    plt.close()

def main_analysis():
    print("Starting comprehensive quiz analysis...")
    results = {}
    results['total_students'] = df['user_id'].nunique()
    
    # Section 1: Performance & Accuracy
    print("Calculating Performance & Accuracy Insights...")
    results['overall_accuracy'] = calculate_overall_accuracy(df)
    results['most_missed'], results['easiest'] = calculate_question_difficulty(df)
    results['avg_score'] = calculate_average_score(df)
    results['perfect_scorers'] = calculate_perfect_scorers(df)
    results['zero_scorers'], _ = calculate_zero_scorers(df)
    results['partial_correct'], _ = calculate_partial_correctness(df)
    results['score_distribution'] = plot_score_distribution(df)
    results['difficulty_mismatch'] = analyze_difficulty_mismatch(df)
    results['polarizing_question'] = find_polarizing_question(df)
    
    # Section 2: Question-Specific
    print("Calculating Question-Specific Insights...")
    results['question_specific'] = analyze_question_specific(df)
    
    # Section 3: Attempt & Behavioral
    print("Calculating Attempt & Behavioral Insights...")
    results['attempt_behavior'] = analyze_attempt_behavior(df)
    
    # Section 5: Question Design
    print("Calculating Question Design Insights...")
    results['question_design'] = analyze_question_design(df)
    
    # Section 6: Advanced Statistical
    print("Calculating Advanced Statistical Insights...")
    results['advanced_stats'] = advanced_statistical_analysis(df)
    
    # Generate Visualizations
    print("\nDisplaying visualizations...")
    plot_performance_metrics(results)
    plot_question_specific(results)
    plot_attempt_behavior(results)
    plot_question_design(results)
    plot_advanced_stats(results)
    
    return results

if __name__ == "__main__":
    analysis_results = main_analysis()