"""
Filter validation utilities.

This module provides validation functions for filter keys, operators, and values
to ensure they conform to expected formats and constraints.
"""

from ...types import FilterValueType


def validate_joined_filter_format(filter_key: str) -> None:
    """
    Validate format of joined filter keys.

    Joined filter keys should follow the pattern: model.field or model.field__operator
    This function ensures the key is properly formatted and doesn't contain
    invalid patterns.

    Args:
        filter_key: Filter key to validate (e.g., 'user.name' or 'user.company.name__eq')

    Raises:
        ValueError: If filter key format is invalid

    Example:
        >>> validate_joined_filter_format('user.name')  # OK
        >>> validate_joined_filter_format('user.company.name__eq')  # OK
        >>> validate_joined_filter_format('.user.name')  # Raises ValueError
        >>> validate_joined_filter_format('user..name')  # Raises ValueError
    """
    if not filter_key or not isinstance(filter_key, str):
        raise ValueError("Filter key must be a non-empty string")

    if filter_key.startswith(".") or filter_key.endswith("."):
        raise ValueError(f"Invalid filter key format: {filter_key}")

    if ".." in filter_key:
        raise ValueError(f"Invalid filter key format (consecutive dots): {filter_key}")


def validate_filter_operator(operator: str, value: FilterValueType) -> None:
    """
    Validate filter operator and value combination.

    Some operators have specific requirements for their values (e.g., 'in' requires
    a list/tuple/set, 'between' requires exactly 2 values).

    Args:
        operator: Filter operator string
        value: Value associated with the operator

    Raises:
        ValueError: If operator and value combination is invalid

    Example:
        >>> validate_filter_operator('eq', 'test')  # OK
        >>> validate_filter_operator('in', [1, 2, 3])  # OK
        >>> validate_filter_operator('between', [1, 10])  # OK
        >>> validate_filter_operator('in', 'invalid')  # Raises ValueError
        >>> validate_filter_operator('between', [1])  # Raises ValueError
    """
    if operator in {"in", "not_in", "between"} and not isinstance(
        value, (tuple, list, set)
    ):
        raise ValueError(f"Operator '{operator}' requires a list, tuple, or set value")

    if (
        operator == "between"
        and isinstance(value, (tuple, list, set))
        and len(value) != 2
    ):
        raise ValueError("Between operator requires exactly 2 values")
