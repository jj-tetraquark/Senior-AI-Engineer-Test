class EventLog:
    def __init__(self, output_file=None):
        self._new_instance_counts = {}
        self._current_interactions = []
        self._output_file = output_file
        self._output_file_handle = None
        self._last_time = 0

    def __enter__(self):
        if self._output_file:
            self._output_file_handle = open(self._output_file, "w")
        return self

    def __exit__(self, type, value, traceback):
        for obj, total in self._new_instance_counts.items():
            self._log_event(self._last_time, f"{obj} total: {total}")

        if self._output_file:
            self._output_file_handle.close()

    def update_state(self, time, state):
        self._update_new_instance_counts(time, state["new_instances"])
        self._update_interactions(time, state["interactions"])
        self._last_time = time

    def _log_event(self, time, event):
        formatted_event = f"{time:05} - {event}"
        print(formatted_event)
        if self._output_file:
            self._output_file_handle.write(f"{formatted_event}\n")

    def _update_new_instance_counts(self, time, new_instances):
        for obj, count in new_instances.items():
            if count == 0:
                continue

            if obj in self._new_instance_counts:
                self._new_instance_counts[obj] += count
            else:
                self._new_instance_counts[obj] = count

            self._log_event(
                time,
                f"{count} new instance of {obj}. "
                f"Total is now: {self._new_instance_counts[obj]}",
            )

    def _update_interactions(self, time, interactions):
        interaction_changes = set(interactions).symmetric_difference(
            self._current_interactions
        )

        for change in interaction_changes:
            object_a, object_b, verb = change
            if change in self._current_interactions:
                self._log_event(time, f"{object_a} no longer {verb} {object_b}")
                self._current_interactions.remove(change)
            else:
                self._log_event(time, f"{object_a} {verb} {object_b}")
                self._current_interactions.append(change)
