"""Shared grouping keys for response activation analyses."""


def response_unit_key(metadata):
    """Use the task item, or a paired repeat key when a task has no item (e.g. DAT)."""
    item = str(metadata.get("Item", "")).strip()
    if item:
        return item

    repeat = str(metadata.get("Repeat", "")).strip()
    seed = str(metadata.get("Seed", "")).strip()
    if repeat:
        run_id = (str(metadata.get("RUN_ID", "")).strip()
                  or str(metadata.get("Run_ID", "")).strip()
                  or str(metadata.get("Model_Seed", "")).strip()
                  or "single-run")
        return f"__repeat__:{run_id}:{repeat}"
    if seed:
        run_id = (str(metadata.get("RUN_ID", "")).strip()
                  or str(metadata.get("Run_ID", "")).strip()
                  or str(metadata.get("Model_Seed", "")).strip()
                  or "single-run")
        return f"__seed__:{run_id}:{seed}"
    return None
