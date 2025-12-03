# 🎨 Style Rules Add-on

(Paste this to enforce code style and conventions)

---

# 🎨 **CODE STYLE & CONVENTIONS**

## **Python**

*   **Formatter**: Black (line length 88).
*   **Imports**: Sorted by `isort` (Standard Lib → Third Party → Local).
*   **Type Hints**: **MANDATORY** for all function arguments and return values.
*   **Docstrings**: Google Style. **MANDATORY** for all modules, classes, and public functions.
*   **Naming**:
    *   Classes: `PascalCase`
    *   Functions/Variables: `snake_case`
    *   Constants: `UPPER_CASE`
    *   Private members: `_leading_underscore`

## **Project Structure**

*   **Scripts**: Place executable scripts in `scripts/`. Use `if __name__ == "__main__":` blocks.
*   **Tests**: Place tests in `tests/`. Mirror the source directory structure.
*   **Configs**: Use YAML for experiment configs, Python dataclasses for internal configs.
*   **Paths**: Use `pathlib.Path` for all file system operations. **NEVER** use string concatenation for paths.

## **Data Handling**

*   **Ticks**: Store prices as **integer ticks** whenever possible to avoid floating-point errors.
*   **DataFrames**: Use Pandas. Ensure consistent column names (e.g., `open`, `high`, `low`, `close`, `volume`).
*   **Serialization**: Use Parquet for large datasets, JSON for metadata/configs.

## **Logging & Output**

*   **Logging**: Use the standard `logging` module. Do not use `print` for production code (scripts are okay).
*   **Progress**: Use `tqdm` for long-running loops.

## **Error Handling**

*   Use specific exception types.
*   Fail fast and provide informative error messages.

---
