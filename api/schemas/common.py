"""
common.py — Shared schemas and utilities.
"""
from marshmallow import Schema, fields


class ErrorSchema(Schema):
    error = fields.String()
    message = fields.String()
    errors = fields.Dict()


class PaginationSchema(Schema):
    page = fields.Integer(load_default=1)
    per_page = fields.Integer(load_default=20)
    total = fields.Integer()
    pages = fields.Integer()


class SuccessSchema(Schema):
    status = fields.String()
    message = fields.String()
