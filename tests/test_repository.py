
from uuid import uuid4

from database.connection import get_connection
from database.repository import (
    save_prediction,
    get_prediction_history,
)


def test_save_and_retrieve_prediction():
    reference = f"TEST-{uuid4().hex[:12]}"
    prediction_id = None

    try:
        prediction_id = save_prediction(
            patient_reference=reference,
            raw_probability=0.54,
            calibrated_probability=0.14,
            risk_tier="Moderate",
        )

        assert isinstance(prediction_id, int)

        connection = get_connection()

        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM predictions WHERE id = %s",
                    (prediction_id,),
                )
                record = cursor.fetchone()

                assert record is not None
                assert record["patient_reference"] == reference
                assert abs(record["calibrated_probability"] - 0.14) < 1e-9
                assert record["risk_tier"] == "Moderate"

        finally:
            connection.close()

        history = get_prediction_history(limit=100)

        assert any(
            row["id"] == prediction_id
            for row in history
        )

    finally:
        if prediction_id is not None:
            connection = get_connection()
            try:
                with connection:
                    with connection.cursor() as cursor:
                        cursor.execute(
                            "DELETE FROM predictions WHERE id = %s",
                            (prediction_id,),
                        )
            finally:
                connection.close()
