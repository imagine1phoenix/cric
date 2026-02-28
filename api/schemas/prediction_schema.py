"""
prediction_schema.py — Marshmallow schemas for prediction validation.
"""
from marshmallow import Schema, fields, validate, validates, ValidationError


class PredictionRequestSchema(Schema):
    team1 = fields.String(required=True)
    team2 = fields.String(required=True)
    venue = fields.String(load_default="")
    city = fields.String(load_default="")
    league = fields.String(load_default="")
    match_type = fields.String(
        load_default="",
        data_key="format",
        validate=validate.OneOf(
            ["", "T20", "ODI", "Test", "T20I", "ODM", "IT20", "MDM"],
            error="Invalid format"
        ),
    )
    gender = fields.String(
        load_default="",
        validate=validate.OneOf(["", "male", "female"], error="Invalid gender"),
    )
    toss_winner = fields.String(load_default="")
    toss_decision = fields.String(
        load_default="",
        validate=validate.OneOf(["", "bat", "field"], error="Invalid toss decision"),
    )
    innings1_score = fields.Integer(load_default=None, allow_none=True)
    innings1_wickets = fields.Integer(load_default=None, allow_none=True)

    @validates("team1")
    def validate_team1(self, value):
        if not value or not value.strip():
            raise ValidationError("team1 is required")

    @validates("team2")
    def validate_team2(self, value):
        if not value or not value.strip():
            raise ValidationError("team2 is required")


class PredictionResponseSchema(Schema):
    predicted_winner = fields.String(attribute="winner")
    team1 = fields.String()
    team2 = fields.String()
    win_probability = fields.Float(attribute="confidence")
    team1_win_prob = fields.Float()
    team2_win_prob = fields.Float()
    confidence = fields.Float()
    confidence_interval_lower = fields.Float(attribute="ci_low")
    confidence_interval_upper = fields.Float(attribute="ci_high")
    model_accuracy = fields.Float()
    model_version = fields.String()
    prediction_id = fields.String()
    cached = fields.Boolean()
    explanation = fields.Dict(load_default=None)
    extended_predictions = fields.Dict(load_default=None)


class CompareRequestSchema(Schema):
    team1 = fields.String(required=True)
    team2 = fields.String(required=True)
    match_type = fields.String(load_default="", data_key="format")
