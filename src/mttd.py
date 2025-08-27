
def mean_time_to_detection(real_changes, detected_changes, delta):
    real_changes = sorted(real_changes)
    detected_changes = sorted(detected_changes)
    matched_detected = set()
    time_diffs = []
    for rc in real_changes:
        match = [dc for dc in detected_changes if abs(dc - rc) <= delta and dc not in matched_detected]
        if match:
            time_diffs.append(abs(match[0] - rc))
            matched_detected.add(match[0])
    if time_diffs:
        return sum(time_diffs) / len(time_diffs)
    else:
        return None
