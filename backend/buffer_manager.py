from database import get_db_connection


BUFFER_TABLES = [f"buffer_{i}" for i in range(1, 7)]

# Copy asset data to a buffer table
def copy_to_buffer(buffer_table, asset_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(f"DELETE FROM {buffer_table}")  # Clear old data
        cursor.execute(f"INSERT INTO {buffer_table} SELECT * FROM option_chain WHERE asset_id = %s", (asset_id,))
        conn.commit()
        print(f"Copied data for asset {asset_id} to {buffer_table}")

    except Exception as e:
        conn.rollback()
        print(f"Error copying to {buffer_table}: {e}")

    finally:
        cursor.close()
        conn.close()

# Update comparison buffers dynamically
def update_comparison_buffers(selected_assets):
    if len(selected_assets) > 6:
        print("Error: You can only compare up to 6 assets.")
        return

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        for i, asset_name in enumerate(selected_assets):
            cursor.execute("SELECT id FROM assets WHERE asset_name = %s", (asset_name,))
            asset = cursor.fetchone()

            if asset:
                copy_to_buffer(BUFFER_TABLES[i], asset[0])
            else:
                print(f"Asset {asset_name} not found in database.")

    except Exception as e:
        print(f"Error updating comparison buffers: {e}")

    finally:
        cursor.close()
        conn.close()
