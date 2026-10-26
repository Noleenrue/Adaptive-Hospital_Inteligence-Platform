
from database.connection import get_connection


def save_prediction(
    patient_reference,
    raw_probability,
    calibrated_probability,
    risk_tier,
    model_name="readmission_xgb",
    model_version="1.0",
):
    """
    Save a prediction and return its database ID.
    """

    query = """
        INSERT INTO predictions (
            patient_reference,
            raw_probability,
            calibrated_probability,
            risk_tier,
            model_name,
            model_version
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id;
    """

    connection = get_connection()

    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (
                        patient_reference,
                        raw_probability,
                        calibrated_probability,
                        risk_tier,
                        model_name,
                        model_version,
                    ),
                )

                result = cursor.fetchone()
                return result["id"]

    finally:
        connection.close()


def get_prediction_history(limit=20):
    """
    Retrieve recent predictions from PostgreSQL.
    """

    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer between 1 and 100")

    query = """
        SELECT
            id,
            patient_reference,
            raw_probability,
            calibrated_probability,
            risk_tier,
            model_name,
            model_version,
            created_at
        FROM predictions
        ORDER BY created_at DESC, id DESC
        LIMIT %s;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(query, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    finally:
        connection.close()
