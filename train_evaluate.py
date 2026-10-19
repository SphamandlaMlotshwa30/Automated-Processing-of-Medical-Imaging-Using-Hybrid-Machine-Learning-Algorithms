import tensorflow as tf
from tensorflow import keras
from model import ResUNetSVM
from data_preprocessing import preprocess_data, IMAGE_SIZE, DEFAULT_DATA_DIR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support, RocCurveDisplay
import seaborn as sns
import os

# Create a directory for saving results
results_dir = "training_results"
os.makedirs(results_dir, exist_ok=True)

# Download (when needed), split, and preprocess the Kaggle dataset.
data_dir = DEFAULT_DATA_DIR
train_data, val_data, test_data, num_classes = preprocess_data(data_dir)

print(f"Number of classes: {num_classes}")
print(f"Class names: {train_data.class_indices}")

# Create the model using the analogy-based architecture
# Input shape is (H, W, 1) for grayscale X-ray images
model = ResUNetSVM(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1), num_classes=num_classes)

# Compile the model
model.compile(
    optimizer="adam",
    loss="categorical_hinge",
    metrics=["accuracy"]
)

# Store training history for plotting
training_history = {
    'accuracy': [],
    'val_accuracy': [],
    'loss': [],
    'val_loss': []
}

# Custom callback to save model and plots after each epoch
class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        # Store metrics in training history
        training_history["accuracy"].append(logs.get("accuracy", 0))
        training_history["val_accuracy"].append(logs.get("val_accuracy", 0))
        training_history["loss"].append(logs.get("loss", 0))
        training_history["val_loss"].append(logs.get("val_loss", 0))
        
        # Save model after each epoch
        # Using a neutral naming convention
        model_path = os.path.join(results_dir, f"multi_cancer_cnn_model_epoch_{epoch+1}.keras")
        self.model.save(model_path)
        print(f"\nModel saved to {model_path}")

        # Save training history plots after each epoch
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(training_history["accuracy"], label="Training Accuracy")
        plt.plot(training_history["val_accuracy"], label="Validation Accuracy")
        plt.title(f"Model Accuracy - Epoch {epoch+1}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(training_history["loss"], label="Training Loss")
        plt.plot(training_history["val_loss"], label="Validation Loss")
        plt.title(f"Model Loss - Epoch {epoch+1}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"training_history_epoch_{epoch+1}.png"), dpi=600, bbox_inches="tight")
        # Also save as SVG for scale-independent resolution
        plt.savefig(os.path.join(results_dir, f"training_history_epoch_{epoch+1}.svg"), format="svg", bbox_inches="tight")
        plt.close()
        print(f"Training history plot saved for epoch {epoch+1}")

def save_metrics_to_file(history, test_accuracy, precision, recall, f1, auc_roc, output_path="Metrics.txt"):
    """Write the final evaluation metrics to a text file for the Gradio UI."""
    training_accuracy = history.history.get("accuracy", [None])[-1]
    validation_accuracy = history.history.get("val_accuracy", [None])[-1]

    with open(output_path, "w", encoding="utf-8") as metrics_file:
        metrics_file.write("Detailed Metrics:\n")
        metrics_file.write(f"Training Accuracy : {training_accuracy * 100:.2f}\n" if training_accuracy is not None else "Training Accuracy : N/A\n")
        metrics_file.write(f"Validation Accuracy: {validation_accuracy * 100:.2f}\n" if validation_accuracy is not None else "Validation Accuracy: N/A\n")
        metrics_file.write(f"Precision: {precision * 100:.2f}\n")
        metrics_file.write(f"Recall: {recall * 100:.2f}\n")
        metrics_file.write(f"F1-Score: {f1 * 100:.2f}\n")
        metrics_file.write(f"AUC-ROC: {auc_roc * 100:.2f}\n")


# Train the model
print("Starting training...")
history = model.fit(
    train_data,
    epochs=50, # Reduced to 50 as per manuscript alignment
    validation_data=val_data,
    verbose=1,
    callbacks=[CustomSaver()]
)

# Evaluate the model on the test set after training completes
print("Evaluating on test data...")
test_loss, test_accuracy = model.evaluate(test_data, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Generate predictions for detailed metrics
print("Generating predictions...")
predictions = model.predict(test_data)
predicted_classes = np.argmax(predictions, axis=1)
probabilities = tf.nn.softmax(predictions, axis=1).numpy()

# Get true labels
true_labels = test_data.classes

# Calculate metrics
auc_roc = roc_auc_score(
    true_labels,
    probabilities,
    labels=np.arange(num_classes),
    multi_class="ovr",
    average="weighted",
)

precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels,
    predicted_classes,
    average="weighted",
    zero_division=0,
)

print(f"\nDetailed Metrics:")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")

save_metrics_to_file(history, test_accuracy, precision, recall, f1, auc_roc)

# Classification report
class_names = list(train_data.class_indices.keys())
print(f"\nClassification Report:")
print(classification_report(true_labels, predicted_classes, target_names=class_names, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(true_labels, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.tight_layout()
confusion_matrix_path = os.path.join(results_dir, "confusion_matrix.png")
confusion_matrix_svg_path = os.path.join(results_dir, "confusion_matrix.svg")
plt.savefig(confusion_matrix_path, dpi=600, bbox_inches="tight")
plt.savefig(confusion_matrix_svg_path, format="svg", bbox_inches="tight")
plt.close()

# One-vs-rest AUC-ROC curves
plt.figure(figsize=(8, 6))
for class_index, class_name in enumerate(class_names):
    RocCurveDisplay.from_predictions(
        (true_labels == class_index).astype(int),
        probabilities[:, class_index],
        name=class_name,
    )
plt.title(f"One-vs-Rest ROC Curves (Weighted AUC = {auc_roc:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True)
auc_roc_path = os.path.join(results_dir, "auc_roc_curve.svg")
auc_roc_png_path = os.path.join(results_dir, "auc_roc_curve.png")
plt.savefig(auc_roc_path, format="svg", bbox_inches="tight")
plt.savefig(auc_roc_png_path, dpi=600, bbox_inches="tight")
plt.close()
print(f"AUC-ROC curves saved as \"{auc_roc_path}\"")

# Final training history plot
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(training_history["accuracy"], 'b-', label="Training Accuracy", linewidth=2)
plt.plot(training_history["val_accuracy"], 'r-', label="Validation Accuracy", linewidth=2)
plt.title("Model Accuracy Over Time")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(training_history["loss"], 'b-', label="Training Loss", linewidth=2)
plt.plot(training_history["val_loss"], 'r-', label="Validation Loss", linewidth=2)
plt.title("Model Loss Over Time")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.bar(class_names, [cm[i, i] for i in range(len(class_names))], color="steelblue")
plt.title("Correct Predictions by Class")
plt.xlabel("Class")
plt.ylabel("Number of Correct Predictions")
plt.xticks(rotation=45)

plt.tight_layout()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "final_training_summary.png"), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(results_dir, "final_training_summary.svg"), format="svg", bbox_inches="tight")
plt.close()

print("\nTraining and evaluation complete!")
print(f"Final Results:")
print(f"- Test Accuracy: {test_accuracy:.4f}")
print(f"- Precision: {precision:.4f}")
print(f"- Recall: {recall:.4f}")
print(f"- F1-Score: {f1:.4f}")
print(f"- AUC-ROC: {auc_roc:.4f}")
print(f"- Confusion Matrix saved as \"{confusion_matrix_path}\"")
print(f"- Training History plots saved in {results_dir}")
print(f"- Final summary plot saved as \"final_training_summary.png\"")

