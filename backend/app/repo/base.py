from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Generic, TypeVar

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select

from app.models.base import BaseModel, SoftDeleteModel

BaseModelType = TypeVar("BaseModelType", bound=BaseModel)
SoftDeleteModelType = TypeVar("SoftDeleteModelType", bound=SoftDeleteModel)


class BaseRepo(Generic[BaseModelType]):
    """Base class for repo operations on SQLAlchemy models.

    Provides common database operations that can be inherited by
    model-specific repo classes.

    Requirements
    ------------
    Models MUST inherit from `app.models.base.BaseModel`, which guarantees:

    - **ID field**: `id: int` (primary key, auto-increment)
    - **Timestamps**: `created_at`, `updated_at` (automatic tracking)

    For Custom Models
    -----------------
    If you have a model that should NOT inherit from BaseModel:

    1. **Option A**: Inherit from BaseModel anyway (unused fields are okay)
    2. **Option B**: Create a custom repo class without extending BaseRepo

    Type Parameters
    ---------------
    BaseModelType : BaseModel
        The SQLAlchemy model type this repo instance operates on.
        Must inherit from `app.models.base.BaseModel`.

    Examples
    --------
    Create a repo instance for your model:

    >>> from app.models.item import Item
    >>> from app.repo.base import BaseRepo
    >>>
    >>> class ItemRepo(BaseRepo[Item]):
    >>>     async def create(self, db, item_create):
    >>>         # Custom create logic here
    >>>         pass
    >>>
    >>> items_repo = ItemRepo(Item)
    """

    def __init__(self, model: type[BaseModelType]):
        """Initialize the repo instance with a model.

        Parameters
        ----------
        model : type[BaseModelType]
            The SQLAlchemy model class to perform operations on.
        """
        self.model = model

    def _build_query(
        self,
        options: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> Select[tuple[BaseModelType]]:
        """Build the base query with optional loading options and filters."""
        query = select(self.model)
        if options:
            query = query.options(*options)
        for field, value in kwargs.items():
            query = query.where(getattr(self.model, field) == value)
        return query

    async def get(
        self,
        db: AsyncSession,
        options: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> BaseModelType | None:
        """Fetch a single record by any field.

        Parameters
        ----------
        db : AsyncSession
            The database session.
        options : Sequence[Any] | None, default=None
            SQLAlchemy loading options (e.g., selectinload).
        **kwargs : Any
            Field-value pairs to filter by (e.g., email="user@example.com").

        Returns
        -------
        BaseModelType | None
            The matching record, or None if not found.

        Examples
        --------
        >>> from sqlalchemy.orm import selectinload
        >>> user = await crud.get(db, email="user@example.com", options=[selectinload(User.items)])
        >>> user = await crud.get(db, username="john", is_deleted=False)
        """
        query = self._build_query(options=options, **kwargs)
        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def get_multi(
        self,
        db: AsyncSession,
        offset: int = 0,
        limit: int = 100,
        options: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch multiple records with pagination.

        Parameters
        ----------
        db : AsyncSession
            The database session.
        offset : int, default=0
            Number of records to skip.
        limit : int, default=100
            Maximum number of records to return.
        options : Sequence[Any] | None, default=None
            SQLAlchemy loading options (e.g., selectinload).
        **kwargs : Any
            Field-value pairs to filter by.

        Returns
        -------
        dict[str, Any]
            Dictionary containing 'data' (list of records) and 'total_count'.

        Examples
        --------
        >>> from sqlalchemy.orm import selectinload
        >>> result = await crud.get_multi(db, offset=0, limit=10, options=[selectinload(User.items)])
        >>> users = result['data']
        >>> total = result['total_count']
        """
        # Build the filtered query (without options for count)
        query = self._build_query(**kwargs)

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total_count = total_result.scalar_one()

        # Get paginated data with options
        data_query = query.offset(offset).limit(limit)
        if options:
            data_query = data_query.options(*options)
        result = await db.execute(data_query)
        data = result.scalars().all()

        return {"data": data, "total_count": total_count}

    async def exists(
        self,
        db: AsyncSession,
        **kwargs: Any,
    ) -> bool:
        """Check if a record exists.

        Parameters
        ----------
        db : AsyncSession
            The database session.
        **kwargs : Any
            Field-value pairs to filter by.

        Returns
        -------
        bool
            True if a matching record exists, False otherwise.

        Examples
        --------
        >>> exists = await crud.exists(db, email="user@example.com")
        """
        query = self._build_query(**kwargs)

        count_query = select(func.count()).select_from(query.subquery())
        result = await db.execute(count_query)
        count = result.scalar_one()
        return count > 0

    async def count(
        self,
        db: AsyncSession,
        **kwargs: Any,
    ) -> int:
        """Count records matching the filter.

        Parameters
        ----------
        db : AsyncSession
            The database session.
        **kwargs : Any
            Field-value pairs to filter by.

        Returns
        -------
        int
            Number of matching records.
        """
        query = self._build_query(**kwargs)

        count_query = select(func.count()).select_from(query.subquery())
        result = await db.execute(count_query)
        return result.scalar_one()

    async def db_delete(
        self,
        db: AsyncSession,
        **kwargs: Any,
    ) -> bool:
        """Hard delete a record (removes from database).

        Parameters
        ----------
        db : AsyncSession
            The database session.
        **kwargs : Any
            Field-value pairs to identify the record to delete.

        Returns
        -------
        bool
            True if a record was deleted, False if not found.

        Examples
        --------
        >>> deleted = await crud.db_delete(db, id=123)
        """
        record = await self.get(db, **kwargs)
        if record is None:
            return False

        await db.delete(record)
        await db.commit()
        return True


class SoftDeleteRepo(BaseRepo[SoftDeleteModelType]):
    """Base class for repo operations on soft-deletable SQLAlchemy models.

    Extends :class:`BaseRepo` with soft-delete capabilities (exclude_deleted
    filtering, soft delete method).

    Requirements
    ------------
    Models MUST inherit from ``app.models.base.SoftDeleteModel``, which
    provides ``is_deleted`` and ``deleted_at`` fields.

    Type Parameters
    ---------------
    SoftDeleteModelType : SoftDeleteModel
        The SQLAlchemy model type this repo instance operates on.
        Must inherit from ``app.models.base.SoftDeleteModel``.

    Examples
    --------
    >>> from app.models.user import User
    >>> from app.repo.base import SoftDeleteRepo
    >>>
    >>> class UserRepo(SoftDeleteRepo[User]):
    >>>     async def create(self, db, user_create):
    >>>         pass
    >>>
    >>> users_repo = UserRepo(User)
    """

    exclude_deleted: bool = True
    """When True, automatically filters out soft-deleted records (``is_deleted=False``).
    Override at the class level or pass ``is_deleted`` explicitly to bypass."""

    def _build_query(
        self,
        options: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> Select[tuple[SoftDeleteModelType]]:
        """Build query with optional soft-delete filtering."""
        query = select(self.model)
        if options:
            query = query.options(*options)
        if self.exclude_deleted and "is_deleted" not in kwargs:
            query = query.where(getattr(self.model, "is_deleted") == False)  # noqa: E712
        for field, value in kwargs.items():
            query = query.where(getattr(self.model, field) == value)
        return query

    async def delete(
        self,
        db: AsyncSession,
        **kwargs: Any,
    ) -> bool:
        """Soft delete a record (sets ``is_deleted=True``, ``deleted_at=now``).

        Parameters
        ----------
        db : AsyncSession
            The database session.
        **kwargs : Any
            Field-value pairs to identify the record to delete.

        Returns
        -------
        bool
            True if a record was deleted, False if not found.

        Notes
        -----
        This is a "soft delete" — the record remains in the database
        but is marked as deleted. Use :meth:`BaseRepo.db_delete` for
        permanent removal.

        Examples
        --------
        >>> deleted = await crud.delete(db, username="john")
        """
        record = await self.get(db, **kwargs)
        if record is None:
            return False

        record.is_deleted = True
        record.deleted_at = datetime.now(UTC)
        db.add(record)
        await db.commit()
        await db.refresh(record)
        return True
