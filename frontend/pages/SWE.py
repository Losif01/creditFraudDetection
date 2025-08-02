import streamlit as st



#introduction
st.markdown('''
## Design patterns used in this project
In this project, i used multiple design patterns, and i advise if you are not from a software engineering/ IT background, to not read this all.
Basically a design pattern is a way to write code and organize it in classes, combined with the SOLID principles, which makes maintainability much easier for the rest of the team, it is a way of communicating code between team members working on the same project
Consider this a small brief, if you are interested, you are more than welcome to see the project breakdown in detail, just scroll a bit
I would like to give big thanks to `Hend Ramadan`, my teammate, who helped me with this big project that was done for educational and practitioning purposes

### Important Note: This page is too big, (roughly 1900 lines were written)
it explains the following files design patterns:
- `data_loader.py`
- `base_model.py`
- `model_factory.py`
- `model_logistic.py`
- `model_mlp.py`
- `model_xgb.py`
- `evaluation.py`
- `trainer.py`
- `logger.py`
''')

# data_loader.py
st.markdown('''
            

`data_loader.py`
### ✅ **1. Strategy Pattern**

#### 🔧 What it is:
The **Strategy Pattern** allows you to define a family of algorithms, encapsulate each one, and make them interchangeable. The context (e.g., `DataLoader`) can switch between strategies at runtime.

#### 📌 Code Snippet 1: Scaler Selection

```python
def _setup_scaler(self):
    """Setup the appropriate scaler based on configuration."""
    scaler_type = self.config['preprocessing']['scaling']['method']
    
    if scaler_type == "robust":
        self.scaler = RobustScaler()
    elif scaler_type == "standard":
        self.scaler = StandardScaler()
    elif scaler_type == "minmax":
        self.scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
```

> This selects a scaling algorithm dynamically based on config.

#### 📌 Code Snippet 2: Sampling Strategy

```python
if sampling_method == "undersample":
    # Manual undersampling logic
    fraud = X[y == 1]
    not_fraud = X[y == 0].sample(n=len(fraud), random_state=random_state)
    ...
elif sampling_method == "oversample":
    sampler = SMOTE(random_state=random_state)
    X_balanced, y_balanced = sampler.fit_resample(X, y)
else:
    return X, y
```

> Chooses between different data balancing techniques.

#### 💡 Impact on Code Quality:
- **Interchangeable behavior**: You can change scalers or sampling methods via config without touching the core logic.
- **Loose coupling**: The pipeline doesn’t depend on concrete classes — only on interfaces (e.g., `.fit_transform()`).
- **Easy extension**: Adding a new scaler (like `QuantileScaler`) only requires updating this block.
- **Runtime flexibility**: The system adapts behavior based on configuration, not hardcoded choices.

---

### ✅ **2. Template Method Pattern**

#### 🔧 What it is:
The **Template Method Pattern** defines the skeleton of an algorithm in a method, deferring some steps to subclasses. It lets subclasses redefine certain steps without changing the algorithm’s structure.

#### 📌 Code Snippet: Complete Preprocessing Pipeline

```python
def load_and_preprocess(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Complete data loading and preprocessing pipeline."""
    logger.info("Starting complete data preprocessing pipeline...")
    
    df = self.load_raw_data()
    df = self.clean_data(df)
    df = self.scale_features(df)
    X, y = self.prepare_features_target(df)
    X, y = self.balance_dataset(X, y)
    X_train, X_test, y_train, y_test = self.split_data(X, y)
    
    logger.info("Data preprocessing completed successfully!")
    
    return X_train, X_test, y_train, y_test
```

> This method defines the **fixed sequence** of data processing steps.

#### 💡 Impact on Code Quality:
- **Consistent workflow**: Every time this method runs, the same steps are executed in the same order.
- **Modular design**: Each step (`clean_data`, `scale_features`, etc.) is a separate method, making it easy to understand and test.
- **Extensibility**: Subclasses can override individual steps (e.g., custom cleaning) while reusing the overall flow.
- **High readability**: The high-level logic reads like a story — very easy to follow.

---

### ✅ **3. Configuration Object Pattern**

#### 🔧 What it is:
The **Configuration Object Pattern** centralizes all settings in a single object, making the system configurable and decoupled from hardcoded values.

#### 📌 Code Snippet: Loading and Resolving Config

```python
def __init__(self, config_path: str = "config/config.yaml"):
    config_abs_path = os.path.abspath(config_path)
    with open(config_abs_path, 'r') as f:
        self.config = yaml.safe_load(f)

    config_dir = os.path.dirname(config_abs_path)
    raw_path = self.config['data']['raw_path']
    self.config['data']['raw_path'] = os.path.join(config_dir, raw_path)
    self.config['data']['raw_path'] = os.path.normpath(self.config['data']['raw_path'])
```

> Loads config from a YAML file and resolves paths relative to the config location.

#### 💡 Impact on Code Quality:
- **Decoupling**: No hardcoded paths or parameters — everything comes from config.
- **Portability**: Works across environments (local, server, CI) by changing just the config file.
- **Maintainability**: All settings are in one place (`config.yaml`), reducing errors.
- **Reusability**: Same class can be reused with different datasets or preprocessing rules.

---

### ✅ **4. Logging (Cross-Cutting Concern)**

#### 🔧 What it is:
While not a classic "design pattern", structured **logging** is a software engineering best practice for managing cross-cutting concerns like monitoring, debugging, and auditing.

#### 📌 Code Snippet: Use of Logger

```python
logger = logging.getLogger(__name__)

# Example usage
logger.info(f"Loading data from {raw_path}")
logger.warning(f"Found {missing_values} missing values")
logger.info(f"Removed {duplicates} duplicates")
```

> Logs provide visibility into what the system is doing at each stage.

#### 💡 Impact on Code Quality:
- **Observability**: You can trace the entire data pipeline execution from logs.
- **Debugging support**: Warnings and info messages help identify issues (e.g., missing data, class imbalance).
- **Production readiness**: Logs enable monitoring in deployed systems.
- **Non-intrusive**: Logging doesn’t interfere with business logic and can be disabled when not needed.

---

### 🏁 Summary

| Design Pattern | Code Snippet | Code Quality Impact |
|---------------|-------------|---------------------|
| **Strategy** | `_setup_scaler`, `balance_dataset` | Enables interchangeable algorithms; improves flexibility and maintainability |
| **Template Method** | `load_and_preprocess` | Defines a fixed workflow with customizable steps; ensures consistency and clarity |
| **Configuration Object** | `__init__` + `config.yaml` loading | Decouples code from settings; enhances portability and reusability |
| **Logging** | `logger.info(...)`, `logger.warning(...)` | Provides visibility into execution; supports debugging and monitoring |

These patterns work together to make your `DataLoader` class **robust, reusable, and production-grade** — a strong foundation for any machine learning system.

''')

# base_model.py
st.markdown('''
            `base_model.py`
### ✅ **1. Abstract Base Class (ABC) Pattern / Template Method Pattern**

#### 🔧 What it is:
This is the core of your design — an **abstract base class** that defines a common interface and shared behavior for all machine learning models. It enforces structure while allowing flexibility.

#### 📌 Code Snippet: Abstract Methods

```python
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract base class for all machine learning models."""
    
    @abstractmethod
    def build_model(self) -> Any:
        """Build and return the model instance."""
        pass
    
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'BaseModel':
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass
```

> These `@abstractmethod` decorators **force all subclasses** (like `LogisticRegressionModel`, `XGBoostModel`) to implement these methods.

#### 📌 Code Snippet: Template Method with Shared Logic

```python
def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate model performance."""
    if not self.is_trained:
        raise ValueError("Model must be trained before evaluation")
    
    y_pred = self.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    return metrics
```

> This method uses the **Template Method Pattern** — it defines the algorithm skeleton (evaluate model), but relies on `predict()` and `predict_proba()`, which may be implemented differently per model.

#### 💡 Impact on Code Quality:
- **Consistency**: All models follow the same interface (`fit`, `predict`, `evaluate`, etc.).
- **Enforced Structure**: Prevents missing key methods in subclasses.
- **Code Reuse**: Common logic (evaluation, saving, logging) is written once.
- **Extensibility**: Easy to add new models (e.g., SVM, Neural Net) by extending `BaseModel`.

---

### ✅ **2. Strategy Pattern (via Inheritance & Polymorphism)**

#### 🔧 What it is:
Each concrete model (e.g., Logistic Regression, XGBoost) implements the same interface but with different internal algorithms — a classic use of **Strategy via polymorphism**.

#### 📌 How It Works:
Even though not shown directly in this file, `BaseModel` enables the **Strategy Pattern** by allowing:
- Different models to be used interchangeably in training/prediction pipelines.
- The rest of the system (e.g., `Trainer`, `ModelFactory`) to work with any model through the same interface.

> Example (conceptual):
```python
model = LogisticRegressionModel(config)
# vs
model = XGBoostModel(config)

# Both can be used like this:
model.fit(X_train, y_train)
metrics = model.evaluate(X_test, y_test)
```

This interchangeability is the essence of the **Strategy Pattern**.

#### 💡 Impact on Code Quality:
- **Interchangeable Models**: You can swap models without changing the training or evaluation code.
- **Loose Coupling**: Higher-level components depend on the abstraction (`BaseModel`), not concrete implementations.
- **Testability**: Each model can be tested using the same evaluation logic.
- **Scalability**: Adding a new model doesn’t require rewriting the pipeline.

---

### ✅ **3. Template Method Pattern (in `evaluate`, `save_model`, etc.)**

#### 📌 Code Snippet: Shared Evaluation Workflow

```python
def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    if not self.is_trained:
        raise ValueError("Model must be trained before evaluation")
    
    y_pred = self.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted')
    }
    
    try:
        y_proba = self.predict_proba(X_test)[:, 1]
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
    except (NotImplementedError, AttributeError):
        logger.warning(f"{self.model_name} does not support ROC AUC calculation")
    
    return metrics
```

> This defines a **standard evaluation workflow** that all models follow, with variation only in how predictions are made.

#### 💡 Impact on Code Quality:
- **Uniform Metrics**: Every model reports the same set of performance metrics.
- **Reliability**: Prevents inconsistencies in evaluation logic.
- **Maintainability**: Fix or improve evaluation once, and it applies to all models.

---

### ✅ **4. Template Method + Hook Methods: `predict_proba`, `get_feature_importance`**

These are **optional extensions** that models can support — they act as **hook methods** in the Template Method Pattern.

#### 📌 Code Snippet: Conditional Behavior via Hooks

```python
def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
    if hasattr(self.model, 'predict_proba'):
        return self.model.predict_proba(X)
    else:
        raise NotImplementedError(f"{self.model_name} does not support probability predictions")
```

```python
def get_feature_importance(self) -> np.ndarray:
    if hasattr(self.model, 'feature_importances_'):
        return self.model.feature_importances_
    elif hasattr(self.model, 'coef_'):
        return np.abs(self.model.coef_[0])
    else:
        raise NotImplementedError(f"{self.model_name} does not support feature importance")
```

> These methods provide **default logic** that checks for model capabilities at runtime.

#### 💡 Impact on Code Quality:
- **Graceful Degradation**: If a model doesn’t support probabilities or feature importance, it fails cleanly with a meaningful message.
- **Adaptability**: Works with both sklearn-style models and custom ones.
- **Abstraction**: Hides implementation differences behind a unified API.

---

### ✅ **5. Logging (Cross-Cutting Concern)**

#### 📌 Code Snippet:

```python
logger = logging.getLogger(__name__)

# Used in:
logger.info(f"Model saved to {filepath}")
logger.warning(f"{self.model_name} does not support ROC AUC calculation")
```

> Provides visibility into model behavior during training, evaluation, and saving.

#### 💡 Impact on Code Quality:
- **Observability**: You can track when models are saved/loaded and why certain metrics are missing.
- **Debugging**: Warnings help identify unsupported operations early.
- **Auditability**: Logs create a trace of model lifecycle events.

---

### ✅ **6. Encapsulation & State Management**

#### 📌 Code Snippet:

```python
def __init__(self, config: Dict[str, Any]):
    self.config = config
    self.model = None
    self.is_trained = False
    self.model_name = self.__class__.__name__
```

> Centralizes model state: configuration, trained status, name.

#### 💡 Impact on Code Quality:
- **Clear State Tracking**: `is_trained` prevents evaluating or saving untrained models.
- **Self-Awareness**: `model_name` enables better logging and representation.
- **Config-Driven**: All models are initialized with config, enabling consistent behavior.

---

### 🏁 Summary

| Design Pattern | Code Snippet | Code Quality Impact |
|---------------|-------------|---------------------|
| **Abstract Base Class** | `class BaseModel(ABC):` + `@abstractmethod` | Enforces consistent interface across models |
| **Template Method** | `evaluate()`, `save_model()`, `get_feature_importance()` | Reuses common logic; defines algorithm skeleton |
| **Strategy Pattern** | Subclasses implement `fit`, `predict` differently | Enables interchangeable models in pipelines |
| **Hook Methods** | `predict_proba()`, `get_feature_importance()` | Allows optional extensions with default behavior |
| **Logging** | `logger.info()`, `logger.warning()` | Adds visibility and debugging support |
| **Encapsulation** | `self.is_trained`, `self.model`, `self.config` | Manages internal state safely and clearly |

---

These patterns make `BaseModel` a **powerful foundation** for your machine learning system. It ensures that all models behave consistently, are easy to extend, and integrate smoothly into larger workflows like training, evaluation, and deployment.


---
''')

# model_factory.py
st.markdown('''
`model_factory.py`
### ✅ **1. Factory Pattern (Core Design Pattern)**

#### 🔧 What it is:
The **Factory Pattern** centralizes object creation, hiding the complexity of instantiation behind a simple interface. It allows you to create objects without exposing the instantiation logic.

#### 📌 Code Snippet: Model Creation via Factory

```python
class ModelFactory:
    """Factory class for creating machine learning models."""
    
    _models = {
        'logistic_regression': LogisticRegressionModel,
        'xgboost': XGBoostModel,
        'mlp': MLPModel,
        # ... and aliases
    }
    
    @classmethod
    def create_model(cls, model_type: str, config: Dict[str, Any]) -> BaseModel:
        model_type_lower = model_type.lower()
        
        if model_type_lower not in cls._models:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model_class = cls._models[model_type_lower]
        logger.info(f"Creating {model_class.__name__} model")
        
        return model_class(config)
```

> This method takes a string (e.g., `'xgboost'`) and returns a fully initialized model instance — no need for the caller to know which class to instantiate.

#### 💡 Impact on Code Quality:
- **Encapsulation of Creation Logic**: The calling code doesn’t need to know about class names or import paths.
- **Single Point of Control**: All model instantiation happens through one method.
- **Simplifies Usage**: Users just say `"xgboost"` and get the right model.
- **Reduces Coupling**: Code that uses models doesn’t depend directly on concrete classes.

---

### ✅ **2. Factory Pattern Extension: Dynamic Registration (Plugin Architecture)**

#### 📌 Code Snippet: Registering New Models at Runtime

```python
@classmethod
def register_model(cls, model_name: str, model_class: type):
    if not issubclass(model_class, BaseModel):
        raise ValueError("Model class must inherit from BaseModel")
    
    cls._models[model_name.lower()] = model_class
    logger.info(f"Registered new model: {model_name} -> {model_class.__name__}")
```

> This allows new models to be added **after the code is written**, even from outside the module.

#### 💡 Impact on Code Quality:
- **Extensibility**: You can plug in new models without modifying the factory itself.
- **Open/Closed Principle**: Open for extension, closed for modification.
- **Supports Ecosystem Growth**: Ideal for libraries or systems where third-party models may be added later.

---

### ✅ **3. Abstract Factory / Wrapper Pattern (via `create_sklearn_model_wrapper`)**

#### 🔧 What it is:
This is a powerful combination of the **Factory Pattern** and **Adapter/Wrapper Pattern** — it dynamically creates wrapper classes that conform to your `BaseModel` interface.

#### 📌 Code Snippet: Sklearn Model Wrapper Generator

```python
def create_sklearn_model_wrapper(sklearn_model_class, default_params=None):
    """Create a wrapper for sklearn models to work with our BaseModel interface."""
    
    class SklearnModelWrapper(BaseModel):
        def __init__(self, config: Dict[str, Any]):
            super().__init__(config)
            self.sklearn_class = sklearn_model_class
            self.default_params = default_params or {}
            self.model = self.build_model()
        
        def build_model(self):
            params = {**self.default_params, **model_config}
            return self.sklearn_class(**params)
        
        def fit(self, X_train, y_train):
            self.model.fit(X_train, y_train)
            self.is_trained = True
            return self
        
        def predict(self, X):
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            return self.model.predict(X)
    
    return SklearnModelWrapper
```

> This function **generates a new class on the fly** that wraps any scikit-learn classifier and makes it compatible with your system.

#### 📌 Example Use:

```python
ModelFactory.register_model('random_forest', create_sklearn_model_wrapper(
    RandomForestClassifier, {'random_state': 42}))
```

> Now `random_forest` can be created via the factory just like any native model.

#### 💡 Impact on Code Quality:
- **Uniform Interface**: All models (custom or sklearn) behave the same way.
- **Massive Reusability**: You instantly gain access to all scikit-learn models.
- **Abstraction Over Diversity**: Hides differences between model APIs behind a common contract.
- **Reduces Boilerplate**: No need to write a full wrapper class for each sklearn model.

---

### ✅ **4. Configuration-Driven Object Creation**

#### 📌 Code Snippet: Merging Config and Defaults

```python
model_config = self.config.get('models', {}).get(model_name, {})
params = {**self.default_params, **model_config}
return self.sklearn_class(**params)
```

> Uses configuration from `config.yaml` to override default parameters.

#### 💡 Impact on Code Quality:
- **Flexibility**: Model hyperparameters can be changed without code edits.
- **Experimentation Support**: Easy to test different settings via config.
- **Environment-Specific Tuning**: Dev, test, prod can use different configs.

---

### ✅ **5. Logging (Cross-Cutting Concern)**

#### 📌 Code Snippet:

```python
logger.info(f"Creating {model_class.__name__} model")
logger.info(f"Registered new model: {model_name} -> {model_class.__name__}")
logger.warning(f"Some sklearn models not available: {e}")
```

> Logs provide visibility into what models are being created and registered.

#### 💡 Impact on Code Quality:
- **Traceability**: You can see which models are loaded and when.
- **Debugging Aid**: Helps catch issues during registration or import.
- **Audit Trail**: Useful in production to confirm expected models are available.

---

### ✅ **6. Error Handling & Validation**

#### 📌 Code Snippet:

```python
if not issubclass(model_class, BaseModel):
    raise ValueError("Model class must inherit from BaseModel")
```

> Ensures only valid models are registered.

#### 💡 Impact on Code Quality:
- **Robustness**: Prevents invalid types from breaking the system.
- **Fail Fast**: Catches errors early during registration, not at runtime.
- **Clear Feedback**: Error messages guide users on correct usage.

---

### 🏁 Summary

| Design Pattern | Code Snippet | Code Quality Impact |
|---------------|-------------|---------------------|
| **Factory Pattern** | `ModelFactory.create_model()` | Centralizes and simplifies object creation |
| **Dynamic Registration** | `register_model()` | Enables runtime extensibility and plugin-like behavior |
| **Wrapper / Adapter Pattern** | `create_sklearn_model_wrapper()` | Adapts external models (sklearn) to your interface |
| **Abstract Factory (Conceptual)** | Wrapper generator returns classes | Creates families of compatible objects |
| **Configuration-Driven Creation** | `params = {**defaults, **config}` | Decouples code from hardcoded settings |
| **Logging** | `logger.info()` in creation/registering | Adds observability and debugging support |
| **Validation** | `issubclass()` check | Improves reliability and user feedback |

---

### ✅ Overall Impact on Code Quality

Your `ModelFactory` is a **masterclass in scalable design**:
- It turns model selection into a **config-driven**, **type-safe**, and **extensible** process.
- It enables **hundreds of models** (via sklearn) to be used with minimal code.
- It follows **SOLID principles**:  
  - **Single Responsibility**: Factory only creates models.  
  - **Open/Closed**: Extendable without modification.  
  - **Liskov Substitution**: All models can be used interchangeably.  
  - **Interface Segregation**: Through `BaseModel`.  
  - **Dependency Inversion**: High-level code depends on abstractions.

This is **production-grade architecture** — clean, flexible, and future-proof.

''')

# model_logistic.py
st.markdown('''
            `model_logistic.py`
### ✅ **1. Inheritance (Is-A Relationship)**

#### 🔧 What it is:
The **Inheritance Pattern** allows a class to inherit attributes and methods from a parent class, promoting code reuse and interface consistency.

#### 📌 Code Snippet: Extending `BaseModel`

```python
from .base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    """Logistic Regression model implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_config = config.get('models', {}).get('logistic_regression', {})
        self.model = self.build_model()
```

> This class inherits core functionality (like `evaluate`, `save_model`, `is_trained`) from `BaseModel`.

#### 💡 Impact on Code Quality:
- **Code Reuse**: No need to re-implement evaluation, saving, logging, or state tracking.
- **Consistency**: Follows the same interface as all other models (`fit`, `predict`, etc.).
- **Maintainability**: Changes to shared logic (e.g., in `BaseModel.evaluate`) apply automatically.
- **Polymorphism Support**: Can be used anywhere a `BaseModel` is expected (e.g., in `Trainer`, `ModelFactory`).

---

### ✅ **2. Constructor Initialization with Configuration**

#### 📌 Code Snippet: Config-Driven Setup

```python
def __init__(self, config: Dict[str, Any]):
    super().__init__(config)
    self.model_config = config.get('models', {}).get('logistic_regression', {})
    self.model = self.build_model()
```

> Loads model-specific parameters from the central config file.

#### 💡 Impact on Code Quality:
- **Decoupling**: No hardcoded hyperparameters — everything comes from config.
- **Flexibility**: Easy to experiment with different values (e.g., `C=0.01`, `solver='liblinear'`) without changing code.
- **Uniformity**: All models follow the same pattern for config access.

---

### ✅ **3. Template Method Pattern (via Abstract Methods)**

#### 📌 Code Snippet: Implementing Abstract Methods

```python
def build_model(self) -> LogisticRegression:
    params = {
        'C': self.model_config.get('C', 0.01),
        'penalty': self.model_config.get('penalty', 'l2'),
        'solver': self.model_config.get('solver', 'lbfgs'),
        'random_state': self.model_config.get('random_state', 42)
    }
    logger.info(f"Building Logistic Regression with params: {params}")
    return LogisticRegression(**params)
```

```python
def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'LogisticRegressionModel':
    logger.info("Training Logistic Regression model...")
    self.model.fit(X_train, y_train)
    self.is_trained = True
    return self
```

```python
def predict(self, X: pd.DataFrame) -> np.ndarray:
    if not self.is_trained:
        raise ValueError("Model must be trained before making predictions")
    return self.model.predict(X)
```

> These methods fulfill the contract defined in `BaseModel`, completing the **Template Method** design.

#### 💡 Impact on Code Quality:
- **Enforced Structure**: Ensures every model implements essential methods.
- **Clear Workflow**: The sequence `build → fit → predict` is explicit and consistent.
- **Extensibility**: Subclasses define *how*, while the base class defines *what* must be done.

---

### ✅ **4. Strategy Pattern (Polymorphic Behavior)**

Even though not directly visible here, this class is a **concrete strategy** in the larger **Strategy Pattern** ecosystem.

#### 📌 How It Fits In:
- `LogisticRegressionModel` provides one specific algorithm for classification.
- It can be swapped with `XGBoostModel`, `MLPModel`, etc., in pipelines like training or prediction.

> Example:
```python
model = ModelFactory.create_model("logistic_regression", config)
# vs
model = ModelFactory.create_model("xgboost", config)

# Both used identically:
model.fit(X_train, y_train)
metrics = model.evaluate(X_test, y_test)
```

#### 💡 Impact on Code Quality:
- **Interchangeability**: The rest of the system doesn’t care which model it’s using.
- **Loose Coupling**: High-level components depend on abstraction (`BaseModel`), not concrete types.
- **Scalability**: Adding new models doesn’t require rewriting training or evaluation logic.

---

### ✅ **5. Logging (Cross-Cutting Concern)**

#### 📌 Code Snippet:

```python
logger.info(f"Building Logistic Regression with params: {params}")
logger.info("Training Logistic Regression model...")
logger.info(f"Training accuracy: {train_score:.4f}")
```

> Provides visibility into model creation and training.

#### 💡 Impact on Code Quality:
- **Observability**: You can trace when and how the model was trained.
- **Debugging Support**: Logs help identify misconfigurations (e.g., wrong solver).
- **Audit Trail**: Useful in production or automated runs to verify behavior.

---

### ✅ **6. Encapsulation & State Management**

#### 📌 Code Snippet:

```python
self.is_trained = True
```

Used in `fit()` to track model state.

#### 💡 Impact on Code Quality:
- **Safe Access**: Prevents calling `predict()` or `evaluate()` before training.
- **Self-Awareness**: Model knows its own lifecycle status.
- **Error Prevention**: Raises clear errors if used incorrectly.

---

### ✅ **7. Method Overriding with Enhanced Functionality**

#### 📌 Code Snippet: Adding Model-Specific Features

```python
def get_coefficients(self) -> np.ndarray:
    if not self.is_trained:
        raise ValueError("Model must be trained to get coefficients")
    return self.model.coef_[0]

def get_intercept(self) -> float:
    if not self.is_trained:
        raise ValueError("Model must be trained to get intercept")
    return self.model.intercept_[0]
```

> Adds domain-specific capabilities beyond the base interface.

#### 💡 Impact on Code Quality:
- **Rich Interface**: Enables interpretation of logistic regression results (coefficients = feature importance).
- **Extensibility**: Base class doesn’t need to know about coefficients — only concrete models that support it expose them.
- **Domain Relevance**: Supports explainability, which is crucial in fraud detection.

---

### 🏁 Summary

| Design Pattern | Code Snippet | Code Quality Impact |
|---------------|-------------|---------------------|
| **Inheritance** | `class LogisticRegressionModel(BaseModel)` | Reuses shared logic; ensures interface consistency |
| **Template Method** | `build_model`, `fit`, `predict` | Completes algorithm skeleton defined in base class |
| **Strategy Pattern** | Used via `ModelFactory` and `BaseModel` | Enables interchangeable use with other models |
| **Configuration-Driven** | `self.model_config = config.get(...)` | Decouples code from hardcoded parameters |
| **Logging** | `logger.info()` during build and train | Adds traceability and debugging support |
| **Encapsulation** | `is_trained` flag, private state | Prevents invalid usage; manages lifecycle safely |
| **Method Extension** | `get_coefficients`, `get_intercept` | Exposes model-specific insights for interpretability |

---

### ✅ Final Thoughts

`LogisticRegressionModel` is a **perfect example of well-structured, maintainable ML code**:
- It **respects the base class contract**.
- It **adds value with model-specific features**.
- It’s **configurable, observable, and safe to use**.
- It **integrates seamlessly** into the larger factory and training system.

This approach ensures that your machine learning components are not just functional, but **professional, scalable, and production-ready**.
''')

#model_mlp.py
st.markdown('''
            `model_mlp.py`

### ✅ **1. Inheritance (Is-A Relationship)**

#### 🔧 What it is:
The **Inheritance Pattern** allows `MLPModel` to reuse functionality from `BaseModel`, ensuring consistency across all models.

#### 📌 Code Snippet: Extending `BaseModel`

```python
from .base_model import BaseModel

class MLPModel(BaseModel):
    """Multi-Layer Perceptron model implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_config = config.get('models', {}).get('mlp', {})
        self.model = self.build_model()
```

> Inherits core behavior like `evaluate()`, `save_model()`, `is_trained`, and `predict_proba()`.

#### 💡 Impact on Code Quality:
- **Code Reuse**: No need to re-implement evaluation, saving, or state tracking.
- **Uniform Interface**: Behaves like all other models (`fit`, `predict`, etc.).
- **Polymorphism**: Can be used interchangeably with any `BaseModel` in pipelines.
- **Maintainability**: Shared logic lives in one place (`BaseModel`), reducing duplication.

---

### ✅ **2. Template Method Pattern**

#### 📌 Code Snippet: Implementing Required Abstract Methods

```python
def build_model(self) -> MLPClassifier:
    params = {
        'hidden_layer_sizes': tuple(self.model_config.get('hidden_layer_sizes', [50, 50])),
        'max_iter': self.model_config.get('max_iter', 1000),
        'alpha': self.model_config.get('alpha', 0.0001),
        'solver': self.model_config.get('solver', 'adam'),
        'random_state': self.model_config.get('random_state', 42)
    }
    logger.info(f"Building MLP with params: {params}")
    return MLPClassifier(**params)
```

```python
def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'MLPModel':
    logger.info("Training MLP model...")
    self.model.fit(X_train, y_train)
    self.is_trained = True
    return self
```

```python
def predict(self, X: pd.DataFrame) -> np.ndarray:
    if not self.is_trained:
        raise ValueError("Model must be trained before making predictions")
    return self.model.predict(X)
```

> These methods complete the algorithm skeleton defined in `BaseModel`.

#### 💡 Impact on Code Quality:
- **Enforced Structure**: Ensures every model implements essential steps.
- **Clear Workflow**: The sequence `build → fit → predict` is consistent across models.
- **Extensibility**: Subclasses define *how*, while the base class defines *what* must be done.

---

### ✅ **3. Strategy Pattern (Polymorphic Behavior)**

Even though not visible directly in this file, `MLPModel` is a **concrete strategy** in the larger system.

#### 📌 How It Fits In:
- It provides one specific algorithm (neural network) for classification.
- Can be swapped with `LogisticRegressionModel`, `XGBoostModel`, etc., via `ModelFactory`.

> Example:
```python
model = ModelFactory.create_model("mlp", config)
model.fit(X_train, y_train)
metrics = model.evaluate(X_test, y_test)
```

#### 💡 Impact on Code Quality:
- **Interchangeability**: The rest of the system doesn’t care which model it’s using.
- **Loose Coupling**: High-level components depend on abstraction (`BaseModel`), not concrete types.
- **Scalability**: Adding new models doesn’t require rewriting training or evaluation logic.

---

### ✅ **4. Configuration-Driven Initialization**

#### 📌 Code Snippet:

```python
self.model_config = config.get('models', {}).get('mlp', {})
```

> Loads hyperparameters from config (e.g., `hidden_layer_sizes`, `max_iter`) instead of hardcoding.

#### 💡 Impact on Code Quality:
- **Flexibility**: Easy to tune or experiment via config file.
- **Decoupling**: No need to modify code to change network architecture.
- **Reproducibility**: Full setup is version-controlled in config.

---

### ✅ **5. Logging (Cross-Cutting Concern)**

#### 📌 Code Snippet:

```python
logger.info(f"Building MLP with params: {params}")
logger.info(f"Training accuracy: {train_score:.4f}")
logger.info(f"Converged after {self.model.n_iter_} iterations")
logger.warning("Model did not converge. Consider increasing max_iter.")
```

> Logs key events during model lifecycle.

#### 💡 Impact on Code Quality:
- **Observability**: You can see training progress and convergence status.
- **Debugging Aid**: Warnings help catch issues like non-convergence early.
- **Audit Trail**: Useful in automated runs or production monitoring.

---

### ✅ **6. Encapsulation & State Management**

#### 📌 Code Snippet:

```python
self.is_trained = True
```

Used in `fit()` to track training state.

#### 💡 Impact on Code Quality:
- **Safe Usage**: Prevents calling `predict()` or `get_loss_curve()` before training.
- **Self-Awareness**: Model knows its own status.
- **Error Prevention**: Raises clear errors if used incorrectly.

---

### ✅ **7. Method Extension with Model-Specific Features**

#### 📌 Code Snippet: Adding Neural Network-Specific Insights

```python
def get_loss_curve(self) -> np.ndarray:
    """Get the loss curve during training."""
    if not self.is_trained:
        raise ValueError("Model must be trained to get loss curve")
    return self.model.loss_curve_
```

```python
def plot_loss_curve(self):
    """Plot the training loss curve."""
    try:
        import matplotlib.pyplot as plt
        plt.plot(self.model.loss_curve_)
        plt.title('MLP Training Loss Curve')
        plt.show()
    except ImportError:
        logger.warning("matplotlib not available for plotting")
```

```python
def get_network_info(self) -> Dict[str, Any]:
    """Get information about the trained network."""
    return {
        'layers': self.model.hidden_layer_sizes,
        'n_layers': self.model.n_layers_,
        'n_iter': self.model.n_iter_,
        'loss': self.model.loss_,
        'output_activation': self.model.out_activation_
    }
```

> Adds rich, model-specific functionality beyond the base interface.

#### 💡 Impact on Code Quality:
- **Explainability**: Loss curve helps diagnose training issues (e.g., overfitting, slow convergence).
- **User Experience**: `plot_loss_curve()` provides instant visual feedback (great for notebooks).
- **Robustness**: Graceful fallback if `matplotlib` is missing.
- **Insightfulness**: `get_network_info()` gives a quick summary of trained network properties.

---

### 🏁 Summary

| Design Pattern | Code Snippet | Code Quality Impact |
|---------------|-------------|---------------------|
| **Inheritance** | `class MLPModel(BaseModel)` | Reuses shared logic; ensures interface consistency |
| **Template Method** | `build_model`, `fit`, `predict` | Completes algorithm skeleton defined in base class |
| **Strategy Pattern** | Used via `ModelFactory` and `BaseModel` | Enables interchangeable use with other models |
| **Configuration-Driven** | `self.model_config = config.get(...)` | Decouples code from hardcoded parameters |
| **Logging** | `logger.info()` during build and train | Adds traceability and debugging support |
| **Encapsulation** | `is_trained` flag, private state | Prevents invalid usage; manages lifecycle safely |
| **Method Extension** | `get_loss_curve`, `plot_loss_curve`, `get_network_info` | Exposes model-specific insights for monitoring and interpretation |

---

### ✅ Final Thoughts

`MLPModel` is **not just functional — it’s thoughtful and user-centric**:
- It follows the same clean architecture as other models (thanks to `BaseModel`).
- It goes further by **adding value-specific features** like loss visualization and convergence checks.
- It’s **configurable, observable, and safe to use**.
- It supports **debugging, explainability, and usability** — critical in ML systems.

This is how **professional-grade ML engineering** should look: consistent, extensible, and insightful.
''')

#model_xgb.py
st.markdown('''
`model_xgb.py`
### ✅ **1. Inheritance (Is-A Relationship)**

#### 🔧 What it is:
The **Inheritance Pattern** allows `XGBoostModel` to reuse functionality from `BaseModel`, ensuring consistency across all models.

#### 📌 Code Snippet: Extending `BaseModel`

```python
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    """XGBoost model implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_config = config.get('models', {}).get('xgboost', {})
        self.model = self.build_model()
```

> Inherits shared logic like `evaluate()`, `save_model()`, `predict_proba()`, and `is_trained`.

#### 💡 Impact on Code Quality:
- **Code Reuse**: No need to re-implement evaluation, saving, or state tracking.
- **Uniform Interface**: Behaves just like `LogisticRegressionModel` or `MLPModel`.
- **Polymorphism**: Can be used interchangeably in training, prediction, or factory systems.
- **Maintainability**: Shared behavior lives in one place (`BaseModel`), reducing duplication.

---

### ✅ **2. Template Method Pattern**

#### 📌 Code Snippet: Implementing Required Abstract Methods

```python
def build_model(self) -> XGBClassifier:
    params = {
        'n_estimators': self.model_config.get('n_estimators', 100),
        'max_depth': self.model_config.get('max_depth', 6),
        'learning_rate': self.model_config.get('learning_rate', 0.1),
        'random_state': self.model_config.get('random_state', 42),
        'eval_metric': 'logloss'
    }
    logger.info(f"Building XGBoost with params: {params}")
    return XGBClassifier(**params)
```

```python
def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'XGBoostModel':
    logger.info("Training XGBoost model...")
    self.model.fit(X_train, y_train)
    self.is_trained = True
    return self
```

```python
def predict(self, X: pd.DataFrame) -> np.ndarray:
    if not self.is_trained:
        raise ValueError("Model must be trained before making predictions")
    return self.model.predict(X)
```

> These methods complete the algorithm skeleton defined in `BaseModel`.

#### 💡 Impact on Code Quality:
- **Enforced Structure**: Every model must implement `build_model`, `fit`, and `predict`.
- **Consistent Workflow**: The training pipeline is predictable and reusable.
- **Extensibility**: Subclasses define *how*, while the base class defines *what* must be done.

---

### ✅ **3. Strategy Pattern (Polymorphic Behavior)**

Even though not directly visible here, `XGBoostModel` is a **concrete strategy** in the larger **Strategy Pattern** ecosystem.

#### 📌 How It Fits In:
- It provides one specific algorithm (gradient boosting) for classification.
- Can be swapped with `LogisticRegressionModel`, `MLPModel`, etc., via `ModelFactory`.

> Example:
```python
model = ModelFactory.create_model("xgboost", config)
# vs
model = ModelFactory.create_model("mlp", config)

# Both used identically:
model.fit(X_train, y_train)
metrics = model.evaluate(X_test, y_test)
```

#### 💡 Impact on Code Quality:
- **Interchangeability**: The rest of the system doesn’t care which model it’s using.
- **Loose Coupling**: High-level components depend on abstraction (`BaseModel`), not concrete types.
- **Scalability**: Adding new models doesn’t require rewriting training or evaluation logic.

---

### ✅ **4. Configuration-Driven Initialization**

#### 📌 Code Snippet:

```python
self.model_config = config.get('models', {}).get('xgboost', {})
```

> Loads hyperparameters like `n_estimators`, `max_depth`, `learning_rate` from config.

#### 💡 Impact on Code Quality:
- **Flexibility**: Easy to tune or experiment via config file.
- **Decoupling**: No need to modify code to change model settings.
- **Reproducibility**: Full setup is version-controlled in config.
- **Environment-Specific Tuning**: Dev vs production models can have different parameters.

---

### ✅ **5. Logging (Cross-Cutting Concern)**

#### 📌 Code Snippet:

```python
logger.info(f"Building XGBoost with params: {params}")
logger.info("Training XGBoost model...")
logger.info(f"Training accuracy: {train_score:.4f}")
```

> Logs key events during model creation and training.

#### 💡 Impact on Code Quality:
- **Observability**: You can trace which parameters were used and how the model performed.
- **Debugging Aid**: Helps identify misconfigurations (e.g., wrong `max_depth`).
- **Audit Trail**: Useful in automated pipelines or production monitoring.

---

### ✅ **6. Encapsulation & State Management**

#### 📌 Code Snippet:

```python
self.is_trained = True
```

Set in `fit()` to track model lifecycle.

#### 💡 Impact on Code Quality:
- **Safe Access**: Prevents calling `predict()` or `get_feature_importance()` before training.
- **Self-Awareness**: Model knows its own status.
- **Error Prevention**: Raises clear errors if used incorrectly.

---

### ✅ **7. Method Extension with Model-Specific Features**

#### 📌 Code Snippet: Overriding `get_feature_importance`

```python
def get_feature_importance(self) -> np.ndarray:
    """Get feature importance scores."""
    if not self.is_trained:
        raise ValueError("Model must be trained to get feature importance")
    return self.model.feature_importances_
```

> Overrides the base method to provide XGBoost’s built-in importance scores.

#### 📌 Code Snippet: Adding Visualization Support

```python
def plot_importance(self, max_num_features: int = 20):
    """Plot feature importance."""
    try:
        from xgboost import plot_importance
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_importance(self.model, ax=ax, max_num_features=max_num_features)
        plt.title("XGBoost Feature Importance")
        plt.tight_layout()
        return fig
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None
```

> Adds rich, model-specific functionality for **interpretability and debugging**.

#### 💡 Impact on Code Quality:
- **Explainability**: Feature importance is critical in fraud detection to understand what drives predictions.
- **User Experience**: Built-in plotting makes it easy to visualize insights (great for notebooks or dashboards).
- **Robustness**: Graceful handling of missing dependencies (`matplotlib`, `xgboost.plotting`).
- **Value-Added**: Goes beyond basic prediction to support analysis and trust in the model.

---

### 🏁 Summary

| Design Pattern | Code Snippet | Code Quality Impact |
|---------------|-------------|---------------------|
| **Inheritance** | `class XGBoostModel(BaseModel)` | Reuses shared logic; ensures interface consistency |
| **Template Method** | `build_model`, `fit`, `predict` | Completes algorithm skeleton defined in base class |
| **Strategy Pattern** | Used via `ModelFactory` and `BaseModel` | Enables interchangeable use with other models |
| **Configuration-Driven** | `self.model_config = config.get(...)` | Decouples code from hardcoded parameters |
| **Logging** | `logger.info()` during build and train | Adds traceability and debugging support |
| **Encapsulation** | `is_trained` flag, private state | Prevents invalid usage; manages lifecycle safely |
| **Method Extension** | `get_feature_importance`, `plot_importance` | Exposes model-specific insights for interpretability |

---

### ✅ Final Thoughts

`XGBoostModel` is a **strong example of clean, professional ML engineering**:
- It integrates seamlessly into your existing architecture via `BaseModel` and `ModelFactory`.
- It leverages XGBoost’s strengths — especially **feature importance** — to enhance model transparency.
- It adds **practical, user-focused features** like plotting, which are invaluable during development and explanation.

This approach ensures that your models are not just accurate, but also **understandable, maintainable, and production-ready**.

''')

#evaluation.py
st.markdown('''
`evaluation.py`
### ✅ **1. Strategy Pattern (via Configurable Metrics)**

#### 🔧 What it is:
The **Strategy Pattern** allows you to dynamically select which metrics to compute, based on configuration.

#### 📌 Code Snippet: Conditional Metric Calculation

```python
def evaluate_single_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    metrics = {}
    
    if 'accuracy' in self.metrics_config:
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
    
    if 'f1_weighted' in self.metrics_config:
        metrics['f1_weighted'] = f1_score(y_test, y_pred, average='weighted')
    
    if 'roc_auc' in self.metrics_config:
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        except (NotImplementedError, AttributeError):
            logger.warning(f"ROC AUC not available for {model.model_name}")
```

> The evaluator **selects which metrics to calculate** based on config — a classic use of Strategy.

#### 💡 Impact on Code Quality:
- **Flexibility**: You can enable/disable metrics without changing code.
- **Extensibility**: Easy to add new metrics (e.g., `precision_micro`) by updating config.
- **Performance Control**: Skip expensive metrics (like ROC AUC) if not needed.
- **User-Centric**: Evaluation adapts to the use case (e.g., fraud detection may prioritize recall).

---

### ✅ **2. Template Method Pattern**

#### 🔧 What it is:
The **Template Method Pattern** defines a fixed algorithm structure, with customizable steps.

#### 📌 Code Snippet: Standardized Evaluation Workflow

```python
def evaluate_single_model(self, model, X_test, y_test):
    logger.info(f"Evaluating {model.model_name}")
    y_pred = model.predict(X_test)
    metrics = {}
    # ... calculate selected metrics
    return metrics
```

> This method defines a **standard evaluation pipeline** that all models go through.

Similarly:
```python
def compare_models(self, model_results):
    comparison_df = pd.DataFrame(model_results).T
    comparison_df = comparison_df.round(4)
    comparison_df = comparison_df.sort_values(by='f1_weighted', ascending=False)
    return comparison_df
```

> Defines a consistent way to compare models.

#### 💡 Impact on Code Quality:
- **Consistency**: Every model is evaluated the same way.
- **Reusability**: Same logic applies across experiments.
- **Maintainability**: Fix or improve evaluation once, and it applies everywhere.
- **Clarity**: High-level flow is easy to follow.

---

### ✅ **3. Configuration Object Pattern**

#### 📌 Code Snippet: Config-Driven Behavior

```python
def __init__(self, config: Dict[str, Any]):
    self.config = config
    self.eval_config = config.get('evaluation', {})
    self.metrics_config = self.eval_config.get('metrics', ['accuracy', 'f1_weighted'])
```

> Uses config to control:
- Which metrics to compute
- Whether to generate plots
- Cross-validation settings

#### 💡 Impact on Code Quality:
- **Decoupling**: No hardcoded logic — behavior comes from config.
- **Portability**: Same code works across environments (dev, prod, notebook).
- **Experimentation Support**: Easily test different evaluation setups.
- **Maintainability**: All evaluation rules are defined in one place.

---

### ✅ **4. Single Responsibility Principle (SRP)**

#### 📌 Code Snippet: One Class, One Purpose

```python
class ModelEvaluator:
    """Comprehensive model evaluation utilities."""
    
    def evaluate_single_model(...)
    def cross_validate_model(...)
    def plot_confusion_matrix(...)
    def plot_roc_curve(...)
    def compare_models(...)
    def plot_model_comparison(...)
    def evaluate_feature_importance(...)
    def save_evaluation_results(...)
```

> Each method has a **single, well-defined purpose**.

#### 💡 Impact on Code Quality:
- **Modularity**: Each function does one thing well.
- **Testability**: Easy to unit test individual methods.
- **Readability**: Clear separation of concerns (metrics vs. plotting vs. saving).
- **Maintainability**: Changes to one feature (e.g., ROC curve) don’t affect others.

---

### ✅ **5. Dependency on Abstractions (Polymorphism)**

#### 📌 Code Snippet: Works with Any Model

```python
def evaluate_single_model(self, model, X_test, y_test):
    y_pred = model.predict(X_test)  # relies on BaseModel interface
    y_proba = model.predict_proba(X_test)  # optional, but safely handled
```

> The evaluator doesn’t care *what kind* of model it is — only that it follows the `BaseModel` contract.

#### 💡 Impact on Code Quality:
- **Loose Coupling**: Doesn’t depend on concrete model classes.
- **Interchangeability**: Works with `LogisticRegressionModel`, `XGBoostModel`, etc.
- **Scalability**: Add new models? No need to change the evaluator.

---

### ✅ **6. Template + Hook Methods (Optional Features)**

#### 📌 Code Snippet: Conditional Plotting

```python
if not self.eval_config.get('plot_confusion_matrix', True):
    return None
```

```python
if not self.eval_config.get('plot_roc_curve', True):
    return None
```

> These act as **hook methods** — optional extensions that can be turned on/off.

#### 💡 Impact on Code Quality:
- **Control**: Users can disable plots in headless environments (e.g., server).
- **Performance**: Skip expensive visualization when not needed.
- **Graceful Degradation**: System still works even if plotting is disabled.

---

### ✅ **7. Logging (Cross-Cutting Concern)**

#### 📌 Code Snippet:

```python
logger.info(f"Evaluating {model.model_name}")
logger.warning(f"ROC AUC not available for {model.model_name}")
logger.info(f"\nModel Comparison Results:\n{comparison_df}")
```

> Provides visibility into evaluation flow and issues.

#### 💡 Impact on Code Quality:
- **Observability**: You can trace which models were evaluated and what happened.
- **Debugging Aid**: Warnings help identify missing capabilities (e.g., no `predict_proba`).
- **Audit Trail**: Useful for reproducibility and reporting.

---

### ✅ **8. Data Encapsulation & Safe Serialization**

#### 📌 Code Snippet: JSON-Safe Output

```python
def save_evaluation_results(self, results: Dict[str, Any], filepath: str):
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serializable_results[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v 
                for k, v in value.items()
            }
        else:
            serializable_results[key] = value.tolist() if isinstance(value, np.ndarray) else value
```

> Handles non-serializable types (like `numpy.ndarray`) safely.

#### 💡 Impact on Code Quality:
- **Robustness**: Prevents `TypeError` when saving results.
- **Reproducibility**: Full evaluation results are persisted.
- **Interoperability**: Saved JSON can be used by other systems or dashboards.

---

### 🏁 Summary

| Design Pattern | Code Snippet | Code Quality Impact |
|---------------|-------------|---------------------|
| **Strategy Pattern** | `if 'accuracy' in self.metrics_config` | Enables configurable, flexible evaluation |
| **Template Method** | `evaluate_single_model`, `compare_models` | Defines consistent evaluation workflow |
| **Configuration Object** | `self.eval_config = config.get('evaluation', {})` | Decouples behavior from hardcoded logic |
| **Single Responsibility** | One method per task (metrics, plots, saving) | Improves modularity and testability |
| **Polymorphism** | Works with any `BaseModel` subclass | Enables model-agnostic evaluation |
| **Hook Methods** | `plot_confusion_matrix`, `plot_roc_curve` | Optional features controlled by config |
| **Logging** | `logger.info()`, `logger.warning()` | Adds traceability and debugging support |
| **Encapsulation + Serialization** | `save_evaluation_results` with type checks | Ensures safe, portable output |

---

### ✅ Final Thoughts

`ModelEvaluator` class is a **powerful, professional-grade evaluation engine**:
- It’s **configurable**, so it adapts to different needs.
- It’s **consistent**, so results are comparable.
- It’s **insightful**, with rich visualizations and reports.
- It’s **robust**, handling missing features and data types gracefully.

Together, these patterns make your evaluation system **not just functional, but scalable, maintainable, and production-ready**.

''')

#trainer.py
st.markdown('''
`trainer.py`
### ✅ **1. Strategy Pattern (via Configurable Training & Evaluation)**

#### 🔧 What it is:
The **Strategy Pattern** allows different training and evaluation behaviors (e.g., model selection, hyperparameter tuning) to be selected dynamically.

#### 📌 Code Snippet: Model-Agnostic Training

```python
def train_single_model(self, model_type: str, X_train, y_train, X_test, y_test):
    model = ModelFactory.create_model(model_type, self.config)
    model.fit(X_train, y_train)
    metrics = self.evaluator.evaluate_single_model(model, X_test, y_test)
    return model, metrics
```

> The same method trains *any* model — logistic regression, XGBoost, MLP — without knowing internal details.

#### 💡 Impact on Code Quality:
- **Interchangeability**: Any model from `ModelFactory` can be trained.
- **Loose Coupling**: `ModelTrainer` depends on the `BaseModel` abstraction, not concrete classes.
- **Extensibility**: Add new models? No changes needed in trainer logic.

---

### ✅ **2. Template Method Pattern**

#### 🔧 What it is:
The **Template Method Pattern** defines a fixed algorithm structure, with steps that can vary.

#### 📌 Code Snippet: Standardized Training Workflow

```python
def train_single_model(self, model_type, X_train, y_train, X_test, y_test):
    logger.info(f"Training {model_type} model...")
    model = ModelFactory.create_model(model_type, self.config)
    model.fit(X_train, y_train)
    metrics = self.evaluator.evaluate_single_model(model, X_test, y_test)
    self.trained_models[model_type] = model
    self.evaluation_results[model_type] = metrics
    return model, metrics
```

> This defines a **consistent training pipeline**:
1. Create model
2. Train
3. Evaluate
4. Store results

#### 💡 Impact on Code Quality:
- **Consistency**: Every model follows the same evaluation and storage logic.
- **Reusability**: Used by `train_multiple_models`, hyperparameter tuning, etc.
- **Maintainability**: Fix or enhance training once, and it applies everywhere.

---

### ✅ **3. Composition Over Inheritance**

#### 📌 Code Snippet: Using `ModelEvaluator`

```python
def __init__(self, config_path: str = "config/config.yaml"):
    with open(config_file, 'r') as f:
        self.config = yaml.safe_load(f)
    self.evaluator = ModelEvaluator(self.config)
    self.trained_models = {}
    self.evaluation_results = {}
```

> Instead of inheriting evaluation logic, `ModelTrainer` **uses** a `ModelEvaluator` instance.

#### 💡 Impact on Code Quality:
- **Modularity**: Training and evaluation are separate concerns.
- **Reusability**: `ModelEvaluator` can be used independently (e.g., in notebooks).
- **Flexibility**: You could inject a different evaluator (e.g., for logging to a database).
- **Single Responsibility**: `ModelTrainer` focuses on training, not plotting or metrics.

This is a **best practice** in object-oriented design.

---

### ✅ **4. Factory Pattern (via `ModelFactory`)**

#### 📌 Code Snippet: Centralized Model Creation

```python
model = ModelFactory.create_model(model_type, self.config)
```

> Delegates object creation to `ModelFactory`, hiding complexity.

#### 💡 Impact on Code Quality:
- **Encapsulation**: No need to know class names or import paths.
- **Configuration-Driven**: Model type comes from string input (e.g., `"xgboost"`).
- **Extensibility**: New models can be added via registration, not code changes.
- **Uniform Interface**: All models are created the same way.

This enables the **Strategy Pattern** at scale.

---

### ✅ **5. Configuration Object Pattern**

#### 📌 Code Snippet: Config-Driven Behavior

```python
if not self.config.get('hyperparameter_tuning', {}).get('enabled', False):
    logger.info("Hyperparameter tuning is disabled")
    return ...
```

> Uses config to control:
- Whether to run hyperparameter tuning
- Which models to train
- Cross-validation settings

#### 💡 Impact on Code Quality:
- **Decoupling**: Behavior is defined in config, not hardcoded.
- **Portability**: Same code works across environments.
- **Experimentation**: Easy to test different setups without code changes.
- **Maintainability**: All rules are in one place.

---

### ✅ **6. Single Responsibility Principle (SRP)**

#### 📌 Code Snippet: One Class, Multiple Focused Methods

```python
def train_single_model(...)
def train_multiple_models(...)
def cross_validate_models(...)
def hyperparameter_tuning(...)
def train_with_kfold_comparison(...)
def train_ensemble_models(...)
def perform_grid_search_comparison(...)
def save_trained_models(...)
def load_trained_models(...)
def get_best_model(...)
```

> Each method has a **single, well-defined purpose**.

#### 💡 Impact on Code Quality:
- **Clarity**: Easy to understand what each method does.
- **Testability**: Each can be tested independently.
- **Maintainability**: Changes to one feature (e.g., saving) don’t affect others.
- **Reusability**: Methods like `get_best_model()` can be used in deployment.

---

### ✅ **7. Dependency on Abstractions (Polymorphism)**

#### 📌 Code Snippet: Works with Any Model

```python
def get_best_model(self, metric: str = 'f1_weighted') -> Tuple[str, BaseModel]:
    best_model_name = max(
        self.evaluation_results.keys(),
        key=lambda x: self.evaluation_results[x].get(metric, 0)
    )
    best_model = self.trained_models[best_model_name]
    return best_model_name, best_model
```

> Relies only on the fact that models are stored and have evaluation metrics — no knowledge of internal structure.

#### 💡 Impact on Code Quality:
- **Flexibility**: Can compare any model type.
- **Scalability**: Add new models? No change needed.
- **Robustness**: Works as long as models follow the interface.

---

### ✅ **8. Logging (Cross-Cutting Concern)**

#### 📌 Code Snippet:

```python
logger.info(f"Training {model_type} model...")
logger.error(f"Failed to train {model_type}: {str(e)}")
logger.info(f"Best model: {best_model_name} with {metric}: {best_score:.4f}")
```

> Provides visibility into training progress, errors, and results.

#### 💡 Impact on Code Quality:
- **Observability**: You can trace which models were trained and how they performed.
- **Debugging Aid**: Errors are logged with context.
- **Audit Trail**: Useful for reproducibility and monitoring.

---

### ✅ **9. State Management & Encapsulation**

#### 📌 Code Snippet:

```python
self.trained_models = {}
self.evaluation_results = {}
```

> Stores trained models and their results for later use.

#### 💡 Impact on Code Quality:
- **Persistence Within Session**: Models can be compared, saved, or retrieved without retraining.
- **Convenience**: `get_best_model()`, `save_trained_models()` rely on this state.
- **Encapsulation**: Internal state is managed safely within the class.

---

### 🏁 Summary

| Design Pattern | Code Snippet | Code Quality Impact |
|---------------|-------------|---------------------|
| **Strategy Pattern** | `train_single_model` with `model_type` | Enables interchangeable model training |
| **Template Method** | Fixed training → evaluate → store flow | Ensures consistent workflow |
| **Composition** | `self.evaluator = ModelEvaluator(config)` | Promotes modularity and reuse |
| **Factory Pattern** | `ModelFactory.create_model(...)` | Centralizes and simplifies object creation |
| **Configuration Object** | `self.config.get('hyperparameter_tuning', {})` | Decouples behavior from code |
| **Single Responsibility** | One method per task | Improves clarity, testability, maintainability |
| **Polymorphism** | Works with any `BaseModel` subclass | Enables model-agnostic logic |
| **Logging** | `logger.info()`, `logger.error()` | Adds traceability and debugging support |
| **State Management** | `self.trained_models`, `self.evaluation_results` | Enables comparison, saving, and retrieval |

---

### ✅ Final Thoughts

`ModelTrainer` is a **powerful orchestrator** that brings together all parts of your ML pipeline:
- It uses **clean design patterns** to stay flexible and maintainable.
- It **composes** rather than inherits — a hallmark of professional design.
- It’s **configurable, observable, and extensible**.

This is not just a script — it’s a **production-ready training engine** that scales across models, experiments, and environments.

''')

#logger.py
st.markdown('''
`logger.py`
### ✅ **1. Factory Pattern**

#### 🔧 What it is:
The **Factory Pattern** centralizes object creation — here, the creation of properly configured `Logger` instances.

#### 📌 Code Snippet: `setup_logger` as a Logger Factory

```python
def setup_logger(name: str, level: str = "INFO", log_file: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers = []  # Clear existing handlers
    
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if provided)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
```

> This function **creates and configures** a logger with consistent formatting, handlers, and levels.

#### 💡 Impact on Code Quality:
- **Centralized Configuration**: All loggers follow the same format and behavior.
- **Reusability**: Any module can get a properly configured logger in one line.
- **Consistency**: Every log message looks the same across the project.
- **Avoids Duplication**: No need to repeat handler, formatter, or level setup in every file.

This is a **classic Factory Pattern** — you’re not just returning an object, you’re building it with a standard recipe.

---

### ✅ **2. Factory Pattern (Extended): `get_project_logger`)**

#### 📌 Code Snippet: Project-Specific Logger Factory

```python
def get_project_logger(module_name: str) -> logging.Logger:
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f"fraud_detection_{timestamp}.log")
    
    return setup_logger(module_name, level="INFO", log_file=log_file)
```

> Builds on `setup_logger` to provide **project-wide defaults**:
- Logs go to `logs/` directory
- Log file is timestamped (daily)
- Uses consistent naming convention

#### 💡 Impact on Code Quality:
- **Standardization**: All modules use the same logging structure.
- **Zero Setup for Users**: A module just calls `get_project_logger(__name__)` and gets a ready-to-use logger.
- **Automatic Directory Creation**: No need to manually create `logs/`.
- **Traceability**: Daily log files make it easy to track system behavior over time.

---

### ✅ **3. Singleton Pattern (Implicitly via `logging` Module)**

#### 🔧 What it is:
Python’s `logging.getLogger(name)` returns the **same logger instance** every time for a given name — this is **built-in Singleton behavior**.

#### 📌 Code Snippet:

```python
logger = logging.getLogger(name)
```

> If a logger with `name` already exists, it returns the same instance. Otherwise, it creates one.

#### 💡 Impact on Code Quality:
- **No Duplicate Handlers**: Without `logger.handlers = []`, you’d get repeated logs (a common issue). Your code **explicitly clears handlers**, preventing this.
- **Shared State**: All parts of the app using the same logger name share its configuration.
- **Efficiency**: No overhead of creating multiple instances for the same purpose.

Your `logger.handlers = []` line is **critical** — it ensures clean setup even if the logger was used before.

---

### ✅ **4. Template Method Pattern (Conceptual)**

#### 📌 Code Snippet: Standardized Logging Workflow

```python
# Inside setup_logger:
1. Get logger by name
2. Clear old handlers
3. Set level
4. Create formatter
5. Add console handler
6. Add file handler (if log_file)
7. Return configured logger
```

> This defines a **fixed sequence** for logger creation.

#### 💡 Impact on Code Quality:
- **Predictable Output**: Every logger behaves the same way.
- **Maintainability**: Change the format or add a handler? Do it in one place.
- **Reliability**: No risk of missing a handler or formatter.

---

### ✅ **5. Configuration via Parameters (vs Hardcoding)**

#### 📌 Code Snippet:

```python
def setup_logger(name: str, level: str = "INFO", log_file: str = None)
```

> Allows customization of:
- Logger name
- Log level
- Whether to log to file

#### 💡 Impact on Code Quality:
- **Flexibility**: Use different levels in dev vs production.
- **Testability**: Can disable file logging in tests.
- **User Control**: Callers decide their needs without changing internal logic.

---

### ✅ **6. Separation of Concerns**

#### 📌 Code Snippet: Two Distinct Functions

```python
setup_logger()        # Low-level: creates a configured logger
get_project_logger()  # High-level: project-specific defaults
```

> One handles **generic** setup, the other **project-specific** conventions.

#### 💡 Impact on Code Quality:
- **Modularity**: Core logic (`setup_logger`) can be reused even if project structure changes.
- **Abstraction**: Most modules only need `get_project_logger` — they don’t care about the details.
- **Extensibility**: You could add `get_api_logger()` or `get_training_logger()` later.

---

### 🏁 Summary

| Design Pattern | Code Snippet | Code Quality Impact |
|---------------|-------------|---------------------|
| **Factory Pattern** | `setup_logger`, `get_project_logger` | Centralizes and standardizes logger creation |
| **Singleton Pattern** | `logging.getLogger(name)` | Ensures one instance per name; avoids duplication |
| **Template Method** | Fixed sequence in `setup_logger` | Enforces consistent configuration workflow |
| **Separation of Concerns** | Two-layer design: generic + project-specific | Improves modularity and usability |
| **Configuration-Driven** | Parameters for level, log_file | Increases flexibility and reusability |

---

### ✅ Final Thoughts

`logger.py` is a **small file with big impact**:
- It turns Python’s flexible but complex `logging` module into a **simple, reliable utility**.
- It applies **professional design patterns** to solve common problems (duplicate logs, inconsistent formatting).
- It makes logging **effortless** for the rest of the project — just call `get_project_logger(__name__)`.

This is exactly how logging should be done in a production ML system: **consistent, maintainable, and invisible** to the rest of the code.

''')


#end of code