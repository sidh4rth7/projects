def complexity_distribution(question_counts, complexities):
    """
    Distributes complexities among different question types based on the provided counts.

    Args:
        question_counts (dict): A dictionary containing the number of questions for each type.
        complexities (list): A list of complexity levels to distribute.

    Returns:
        dict: A dictionary with question types as keys and lists of complexities as values.
    """
    distribution = {}
    total_questions = sum(question_counts.values())

    for question_type, count in question_counts.items():
        if count > 0:
            # Distribute complexities evenly based on the number of questions
            distribution[question_type] = [complexities[i % len(complexities)] for i in range(count)]
        else:
            distribution[question_type] = []

    return distribution