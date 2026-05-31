import pytest
from sqlalchemy.orm import selectinload

from app.models.user import User
from app.repo.users import users_repo
from tests.helpers.generators import create_user


@pytest.mark.asyncio
async def test_crud_get(db):
    user = await create_user(db)
    fetched_user = await users_repo.get(db, id=user.id)
    assert fetched_user is not None
    assert fetched_user.id == user.id
    assert fetched_user.email == user.email


@pytest.mark.asyncio
async def test_crud_get_with_options(db):
    user = await create_user(db)
    # Testing that options parameter is accepted and doesn't crash
    fetched_user = await users_repo.get(db, id=user.id, options=[selectinload(User.items)])
    assert fetched_user is not None
    assert fetched_user.id == user.id
    # Accessing items should not raise an error if selectinload worked
    assert isinstance(fetched_user.items, list)


@pytest.mark.asyncio
async def test_crud_get_not_found(db):
    fetched_user = await users_repo.get(db, id=999999)
    assert fetched_user is None


@pytest.mark.asyncio
async def test_crud_get_multi(db):
    await create_user(db)
    await create_user(db)
    result = await users_repo.get_multi(db, limit=10)
    assert len(result["data"]) >= 2
    assert result["total_count"] >= 2


@pytest.mark.asyncio
async def test_crud_get_multi_with_options(db):
    await create_user(db)
    # Testing that options parameter is accepted in get_multi
    result = await users_repo.get_multi(db, limit=10, options=[selectinload(User.items)])
    assert len(result["data"]) >= 1
    assert isinstance(result["data"][0].items, list)


@pytest.mark.asyncio
async def test_crud_exists(db):
    user = await create_user(db)
    exists = await users_repo.exists(db, id=user.id)
    assert exists is True

    not_exists = await users_repo.exists(db, id=999999)
    assert not_exists is False


@pytest.mark.asyncio
async def test_crud_count(db):
    initial_count = await users_repo.count(db)
    await create_user(db)
    new_count = await users_repo.count(db)
    assert new_count == initial_count + 1


@pytest.mark.asyncio
async def test_crud_delete(db):
    user = await create_user(db)
    deleted = await users_repo.delete(db, id=user.id)
    assert deleted is True

    # Check it is soft deleted (pass is_deleted=True to bypass the default exclude_deleted filter)
    fetched_user = await users_repo.get(db, id=user.id, is_deleted=True)
    assert fetched_user.is_deleted is True
    assert fetched_user.deleted_at is not None


@pytest.mark.asyncio
async def test_crud_db_delete(db):
    user = await create_user(db)
    deleted = await users_repo.db_delete(db, id=user.id)
    assert deleted is True

    # Check it is hard deleted
    fetched_user = await users_repo.get(db, id=user.id)
    assert fetched_user is None


@pytest.mark.asyncio
async def test_crud_db_delete_soft_deleted(db):
    user = await create_user(db)
    await users_repo.delete(db, id=user.id)

    deleted = await users_repo.db_delete(db, id=user.id)
    assert deleted is True

    fetched_user = await users_repo.get(db, id=user.id, is_deleted=True)
    assert fetched_user is None
