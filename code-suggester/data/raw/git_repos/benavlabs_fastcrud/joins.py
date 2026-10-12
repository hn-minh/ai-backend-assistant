"""
SQL Join Builder.

This module handles SQL JOIN clause generation for complex queries
involving relationships between models.
"""

from typing import Any
from sqlalchemy import Select

from ...types import ModelType
from ..field_management import extract_matching_columns_from_schema
from ..filtering import FilterProcessor


class JoinBuilder:
    """Handles SQL JOIN clause generation."""

    def __init__(self, model: type[ModelType]):
        """
        Initialize join builder for a specific model.

        Args:
            model: SQLAlchemy model class
        """
        self.model = model

    def prepare_joins(
        self,
        stmt: Select,
        joins_config: list[Any],
        use_temporary_prefix: bool = False,
        select_joined_columns: bool = True,
    ) -> Select:
        """
        Prepare and apply joins to SELECT statement.

        This method applies joins to the given SQL statement based on a list
        of JoinConfig objects. It supports both left and inner joins with
        column selection and filtering.

        Args:
            stmt: SQLAlchemy SELECT statement to modify
            joins_config: List of JoinConfig objects defining join operations
            use_temporary_prefix: Whether to use temporary prefixes for joins
            select_joined_columns: Whether to add joined columns to SELECT. Set to False for count operations.

        Returns:
            Modified SELECT statement with JOIN clauses applied

        Raises:
            ValueError: If unsupported join type is specified

        Example:
            >>> builder = JoinBuilder(User)
            >>> stmt = select(User)
            >>> joins = [JoinConfig(model=Profile, join_on=User.id==Profile.user_id)]
            >>> joined_stmt = builder.prepare_joins(stmt, joins)

        Note:
            This implementation is ported from FastCRUD._prepare_and_apply_joins
            and supports the core join functionality needed for most use cases.
        """
        for join in joins_config:
            model = getattr(join, "alias", None) or getattr(join, "model")

            join_select = extract_matching_columns_from_schema(
                model,
                getattr(join, "schema_to_select", None),
                getattr(join, "join_prefix", None),
                getattr(join, "alias", None),
                use_temporary_prefix,
            )

            joined_model_filters = []
            if hasattr(join, "filters") and join.filters:
                filter_processor = FilterProcessor(model)
                joined_model_filters = filter_processor.parse_filters(**join.filters)

            join_type = getattr(join, "join_type", "left").lower()
            join_on = getattr(join, "join_on")

            if join_type == "left":
                stmt = stmt.outerjoin(model, join_on)
                if select_joined_columns:
                    stmt = stmt.add_columns(*join_select)
            elif join_type == "inner":
                stmt = stmt.join(model, join_on)
                if select_joined_columns:
                    stmt = stmt.add_columns(*join_select)
            else:
                raise ValueError(
                    f"Unsupported join type: {join_type}. Supported types: 'left', 'inner'"
                )

            if joined_model_filters:
                stmt = stmt.filter(*joined_model_filters)

        return stmt
