# Model Interpretability: Complete Guide

## Table of Contents
1. [Introduction to Interpretability](#introduction-to-interpretability)
2. [Feature Importance](#feature-importance)
3. [SHAP Values](#shap-values)
4. [LIME](#lime)
5. [Partial Dependence Plots](#partial-dependence-plots)
6. [Permutation Importance](#permutation-importance)
7. [Attention Visualization](#attention-visualization)
8. [Model-Agnostic Methods](#model-agnostic-methods)
9. [Interpretability for Deep Learning](#interpretability-for-deep-learning)
10. [Practical Examples](#practical-examples)
11. [Best Practices](#best-practices)

---

## Introduction to Interpretability

Model interpretability helps understand how models make predictions, which is crucial for:
- **Trust**: Building trust in AI systems
- **Debugging**: Identifying model issues
- **Compliance**: Meeting regulatory requirements
- **Insights**: Understanding data patterns
- **Fairness**: Detecting bias

### Types of Interpretability

- **Global**: How model works overall
- **Local**: Why specific prediction was made
- **Model-specific**: Methods for specific model types
- **Model-agnostic**: Works with any model

---

## Feature Importance

### Tree-based Feature Importance

```python
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

print(feature_importance)
```

### Permutation-based Importance

```python
from sklearn.inspection import permutation_importance

# Calculate permutation importance
perm_importance = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

# Create DataFrame
perm_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

print(perm_df)
```

---

## SHAP Values

### Installation

```bash
pip install shap
```

### Tree SHAP

```python
import shap

# Tree explainer (for tree-based models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Bar plot (mean absolute SHAP values)
shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar")

# Waterfall plot for single prediction
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_test.iloc[0],
        feature_names=feature_names
    )
)
```

### Kernel SHAP (Model-agnostic)

```python
# Kernel SHAP for any model
explainer = shap.KernelExplainer(model.predict_proba, X_train[:100])
shap_values = explainer.shap_values(X_test[:10])

# Force plot
shap.force_plot(
    explainer.expected_value[0],
    shap_values[0][0],
    X_test.iloc[0],
    feature_names=feature_names
)
```

### Deep SHAP (for Neural Networks)

```python
import tensorflow as tf

# Deep explainer
explainer = shap.DeepExplainer(model, X_train[:100])
shap_values = explainer.shap_values(X_test[:10])

# Visualize
shap.summary_plot(shap_values, X_test[:10])
```

### SHAP for Text Models

```python
# For transformer models
import shap

# Create explainer
explainer = shap.Explainer(model, tokenizer)

# Explain prediction
shap_values = explainer(["Your text here"])

# Visualize
shap.plots.text(shap_values)
```

---

## LIME

### Installation

```bash
pip install lime
```

### Tabular Data

```python
from lime import lime_tabular

# Create explainer
explainer = lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification'
)

# Explain single prediction
explanation = explainer.explain_instance(
    X_test.iloc[0].values,
    model.predict_proba,
    num_features=10
)

# Show explanation
explanation.show_in_notebook(show_table=True)

# Get explanation as list
exp_list = explanation.as_list()
print(exp_list)
```

### Text Data

```python
from lime import lime_text
from lime.lime_text import LimeTextExplainer

# Text explainer
explainer = LimeTextExplainer(class_names=class_names)

# Explain text prediction
explanation = explainer.explain_instance(
    text_sample,
    model.predict_proba,
    num_features=10
)

# Show explanation
explanation.show_in_notebook()
```

### Image Data

```python
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Image explainer
explainer = lime_image.LimeImageExplainer()

# Explain image
explanation = explainer.explain_instance(
    image,
    model.predict_proba,
    top_labels=5,
    hide_color=0,
    num_samples=1000
)

# Get explanation
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],
    positive_only=True,
    num_features=5,
    hide_rest=True
)

# Visualize
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()
```

---

## Partial Dependence Plots

### Basic PDP

```python
from sklearn.inspection import PartialDependenceDisplay

# Partial dependence plot
PartialDependenceDisplay.from_estimator(
    model,
    X_train,
    features=[0, 1, (0, 1)],  # Individual and interaction
    feature_names=feature_names,
    grid_resolution=20
)
plt.show()
```

### ICE Plots

```python
from sklearn.inspection import PartialDependenceDisplay

# Individual Conditional Expectation (ICE)
PartialDependenceDisplay.from_estimator(
    model,
    X_train,
    features=[0],
    kind='individual',  # Show ICE curves
    feature_names=feature_names
)
plt.show()
```

---

## Permutation Importance

### ELI5 Permutation Importance

```python
import eli5
from eli5.sklearn import PermutationImportance

# Permutation importance
perm = PermutationImportance(model, random_state=42).fit(X_test, y_test)
eli5.show_weights(perm, feature_names=feature_names)

# Show prediction explanation
eli5.show_prediction(model, X_test.iloc[0], feature_names=feature_names)
```

---

## Attention Visualization

### Transformer Attention

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)

# Get attention
inputs = tokenizer("Your text here", return_tensors="pt")
outputs = model(**inputs)
attentions = outputs.attentions

# Visualize attention
import matplotlib.pyplot as plt

def plot_attention(attention, tokens):
    """Plot attention weights"""
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(attention, cmap='Blues')
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45)
    ax.set_yticklabels(tokens)
    plt.colorbar(im)
    plt.show()

# Plot first layer attention
plot_attention(attentions[0][0, 0].detach().numpy(), tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
```

---

## Model-Agnostic Methods

### Surrogate Models

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Train surrogate model
surrogate = DecisionTreeClassifier(max_depth=3)
surrogate.fit(X_train, model.predict(X_train))

# Visualize
plt.figure(figsize=(20, 10))
tree.plot_tree(surrogate, feature_names=feature_names, filled=True)
plt.show()
```

### Feature Interaction

```python
from sklearn.inspection import plot_partial_dependence

# Feature interactions
plot_partial_dependence(
    model,
    X_train,
    features=[(0, 1)],  # Interaction between features 0 and 1
    feature_names=feature_names,
    grid_resolution=20
)
plt.show()
```

---

## Interpretability for Deep Learning

### Integrated Gradients

```python
import tensorflow as tf
import numpy as np

def integrated_gradients(model, input_image, baseline, steps=50):
    """Calculate integrated gradients"""
    # Create path from baseline to input
    alphas = np.linspace(0, 1, steps)
    gradients = []
    
    for alpha in alphas:
        interpolated = baseline + alpha * (input_image - baseline)
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            prediction = model(interpolated)
        
        gradient = tape.gradient(prediction, interpolated)
        gradients.append(gradient)
    
    # Average gradients
    avg_gradients = np.mean(gradients, axis=0)
    
    # Integrated gradients
    integrated_grads = (input_image - baseline) * avg_gradients
    
    return integrated_grads

# Usage
baseline = np.zeros_like(image)
attributions = integrated_gradients(model, image, baseline)

# Visualize
plt.imshow(attributions)
plt.show()
```

### Grad-CAM

```python
import tensorflow as tf
from tensorflow import keras

def grad_cam(model, image, layer_name, class_idx):
    """Generate Grad-CAM heatmap"""
    grad_model = keras.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, class_idx]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.maximum(tf.reduce_max(heatmap), 1e-7)
    
    return heatmap.numpy()

# Usage
heatmap = grad_cam(model, image, 'conv_layer', class_idx=1)

# Overlay on image
import cv2
heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
plt.imshow(superimposed)
plt.show()
```

---

## Practical Examples

### Example 1: Complete Interpretability Pipeline

```python
import shap
import lime
from sklearn.inspection import PartialDependenceDisplay

def interpret_model(model, X_train, X_test, y_test, feature_names):
    """Complete interpretability analysis"""
    
    # 1. Feature Importance
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("Feature Importance:")
        print(importance_df)
    
    # 2. SHAP Values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:100])
    shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names)
    
    # 3. LIME
    lime_explainer = lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=feature_names,
        mode='classification'
    )
    explanation = lime_explainer.explain_instance(
        X_test.iloc[0].values,
        model.predict_proba
    )
    explanation.show_in_notebook()
    
    # 4. Partial Dependence
    PartialDependenceDisplay.from_estimator(
        model,
        X_train,
        features=[0, 1],
        feature_names=feature_names
    )
    plt.show()

# Use
interpret_model(model, X_train, X_test, y_test, feature_names)
```

### Example 2: Explain Specific Predictions

```python
def explain_prediction(model, instance, feature_names):
    """Explain a single prediction"""
    
    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(instance.reshape(1, -1))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=instance,
            feature_names=feature_names
        )
    )
    
    # LIME
    lime_explainer = lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=feature_names
    )
    explanation = lime_explainer.explain_instance(
        instance,
        model.predict_proba
    )
    explanation.show_in_notebook()

# Use
explain_prediction(model, X_test.iloc[0].values, feature_names)
```

---

## Best Practices

1. **Use Multiple Methods**: Combine different interpretability techniques
2. **Understand Limitations**: Each method has assumptions
3. **Visualize**: Use plots and visualizations
4. **Document**: Document interpretations
5. **Validate**: Verify interpretations make sense
6. **Consider Context**: Interpretability depends on use case
7. **Balance**: Trade-off between accuracy and interpretability

---

## Resources

- **SHAP**: github.com/slundberg/shap
- **LIME**: github.com/marcotcr/lime
- **ELI5**: eli5.readthedocs.io
- **Papers**: 
  - SHAP (2017)
  - LIME (2016)
  - Integrated Gradients (2017)

---

## Conclusion

Model interpretability is essential for trustworthy AI. Key takeaways:

1. **Use SHAP**: For comprehensive feature attribution
2. **Use LIME**: For local explanations
3. **Visualize**: Make interpretations accessible
4. **Combine Methods**: Different methods provide different insights
5. **Document**: Keep records of interpretations

Remember: Interpretability builds trust and enables better models!

