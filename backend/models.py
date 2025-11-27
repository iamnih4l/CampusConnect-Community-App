from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime # Use the Base from your database.py, do NOT redefine
from database import Base  # <-- import Base from database.py

# User table
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)
    interests = Column(Text)  # comma-separated interests
    profile_pic = Column(String, default="")

class Community(Base):
    __tablename__ = "communities"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(Text)
    creator = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# Post table
class Post(Base):
    __tablename__ = "posts"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String)
    content = Column(Text)
    image = Column(String, nullable=True)
    feed_type = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    likes = Column(Integer, default=0)

# Comment table
class Comment(Base):
    __tablename__ = "comments"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(Integer, ForeignKey("posts.id"))
    username = Column(String)
    comment = Column(Text)

# Like table
class Like(Base):
    __tablename__ = "likes"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(Integer, ForeignKey("posts.id"))
    username = Column(String)
