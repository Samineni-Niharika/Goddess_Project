from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def print_evaluation_summary(y_true, y_pred, class_names, title="Goddess Classification"):
    """Print and return multiclass metrics for held-out test predictions."""
    if not class_names:
        raise ValueError("At least one class name is required for evaluation.")

    labels = list(range(len(class_names)))
    print("\n" + "=" * 76)
    print(f"{title.upper()} MODEL EVALUATION")
    print("=" * 76)
    print("Dataset: Test Dataset")
    print(f"Number of Classes: {len(class_names)}")
    print(f"Class Names: {', '.join(class_names)}")

    if len(y_true) == 0:
        print("Total Test Samples: 0")
        print("Evaluation skipped: the test dataset contains no samples.")
        print("=" * 76)
        return None

    if len(y_true) != len(y_pred):
        raise ValueError("Ground-truth labels and predictions must have equal length.")

    accuracy = accuracy_score(y_true, y_pred)
    averages = {}
    for average in ("macro", "weighted"):
        averages[average] = {
            "precision": precision_score(
                y_true,
                y_pred,
                labels=labels,
                average=average,
                zero_division=0,
            ),
            "recall": recall_score(
                y_true,
                y_pred,
                labels=labels,
                average=average,
                zero_division=0,
            ),
            "f1": f1_score(
                y_true,
                y_pred,
                labels=labels,
                average=average,
                zero_division=0,
            ),
        }
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        digits=4,
        output_dict=True,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        digits=4,
    )

    print(f"Total Test Samples: {len(y_true)}")
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    for average, display_name in (("macro", "Macro"), ("weighted", "Weighted")):
        print(f"{display_name} Precision: {averages[average]['precision']:.4f}")
        print(f"{display_name} Recall:    {averages[average]['recall']:.4f}")
        print(f"{display_name} F1-Score:  {averages[average]['f1']:.4f}")

    label_width = max(16, max(len(name) for name in class_names) + 2)
    print("\n" + "-" * 22 + " CONFUSION MATRIX " + "-" * 36)
    print("Rows: actual class; columns: predicted class.")
    header = "Actual / Predicted"
    print(f"{header:<{label_width}}"
        + "".join(f"{name:>{label_width}}" for name in class_names))
    for class_name, row in zip(class_names, matrix):
        print(f"{class_name:<{label_width}}" + "".join(
            f"{count:>{label_width}d}" for count in row
        ))

    print("\n" + "-" * 20 + " PER-CLASS METRICS " + "-" * 37)
    print(f"{'Class':<{label_width}}{'Precision':>12}{'Recall':>12}{'F1-score':>12}{'Support':>10}")
    for class_name in class_names:
        class_metrics = report_dict[class_name]
        print(
            f"{class_name:<{label_width}}"
            f"{class_metrics['precision']:>12.4f}"
            f"{class_metrics['recall']:>12.4f}"
            f"{class_metrics['f1-score']:>12.4f}"
            f"{int(class_metrics['support']):>10d}"
        )

    print("\n" + "-" * 19 + " CLASSIFICATION REPORT " + "-" * 34)
    print(report_text)
    print("=" * 76)

    return {
        "accuracy": accuracy,
        "macro": averages["macro"],
        "weighted": averages["weighted"],
        "per_class": {
            class_name: report_dict[class_name]
            for class_name in class_names
        },
        "confusion_matrix": matrix,
        "classification_report": report_text,
        "test_samples": len(y_true),
        "class_names": list(class_names),
    }
