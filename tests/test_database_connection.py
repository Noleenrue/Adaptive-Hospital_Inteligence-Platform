from database.connection import get_connection


def test_database_connection():
    """
    Verify that Python can connect to PostgreSQL
    and access the correct project database.
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_database();")
            result = cursor.fetchone()

            assert result["current_database"] == "adaptive_hospital_db"

            cursor.execute("SELECT to_regclass('public.predictions');")
            table_result = cursor.fetchone()

            assert table_result["to_regclass"] == "predictions"

    finally:
        connection.close()
