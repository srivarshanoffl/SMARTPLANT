"""
SmartPlant AI - Alert Engine & Cooldown Manager
-----------------------------------------------
Dedicated industrial alert engine managing lifecycle, severity categorization,
intelligent deduplication, and cooldown mechanisms for energy anomalies.
"""

import os
import json
import uuid
from datetime import datetime, timezone


class AlertRecord:
    """Structured representation of an operational anomaly alert."""
    def __init__(self,
                 alert_id=None,
                 timestamp=None,
                 interval_index=0,
                 alert_type='forecast_deviation',
                 actual_usage_kWh=0.0,
                 predicted_usage_kWh=0.0,
                 error_kWh=0.0,
                 absolute_error_kWh=0.0,
                 percentage_error=None,
                 severity='Medium',
                 explanation='',
                 suggested_action='',
                 status='New',
                 created_at=None,
                 acknowledged_at=None,
                 acknowledged_by=None,
                 resolved_at=None,
                 resolved_by=None,
                 resolution_note=None,
                 occurrence_count=1,
                 last_seen_timestamp=None):
        self.alert_id = alert_id or f"ALT-{uuid.uuid4().hex[:8].upper()}"
        self.timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        self.interval_index = interval_index
        self.alert_type = alert_type
        self.actual_usage_kWh = actual_usage_kWh
        self.predicted_usage_kWh = predicted_usage_kWh
        self.error_kWh = error_kWh
        self.absolute_error_kWh = absolute_error_kWh
        self.percentage_error = percentage_error
        self.severity = severity
        self.explanation = explanation
        self.suggested_action = suggested_action
        self.status = status
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.acknowledged_at = acknowledged_at
        self.acknowledged_by = acknowledged_by
        self.resolved_at = resolved_at
        self.resolved_by = resolved_by
        self.resolution_note = resolution_note
        self.occurrence_count = occurrence_count
        self.last_seen_timestamp = last_seen_timestamp or self.timestamp

    def to_dict(self):
        return {
            'alert_id': self.alert_id,
            'timestamp': str(self.timestamp),
            'interval_index': self.interval_index,
            'alert_type': self.alert_type,
            'actual_usage_kWh': round(self.actual_usage_kWh, 2) if self.actual_usage_kWh is not None else None,
            'predicted_usage_kWh': round(self.predicted_usage_kWh, 2) if self.predicted_usage_kWh is not None else None,
            'error_kWh': round(self.error_kWh, 2) if self.error_kWh is not None else None,
            'absolute_error_kWh': round(self.absolute_error_kWh, 2) if self.absolute_error_kWh is not None else None,
            'percentage_error': round(self.percentage_error, 1) if self.percentage_error is not None else None,
            'severity': self.severity,
            'explanation': self.explanation,
            'suggested_action': self.suggested_action,
            'status': self.status,
            'created_at': self.created_at,
            'acknowledged_at': self.acknowledged_at,
            'acknowledged_by': self.acknowledged_by,
            'resolved_at': self.resolved_at,
            'resolved_by': self.resolved_by,
            'resolution_note': self.resolution_note,
            'occurrence_count': self.occurrence_count,
            'last_seen_timestamp': str(self.last_seen_timestamp)
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class AlertEngine:
    """
    Manages generation, deduplication, cooldown, and lifecycle
    (New -> Acknowledged -> Resolved) of industrial energy alerts.
    """
    def __init__(self, storage_path=None, cooldown_intervals=2):
        self.storage_path = storage_path
        self.cooldown_intervals = cooldown_intervals  # Number of intervals to suppress identical alerts
        self.alerts = []
        self.active_alert_by_key = {}  # (alert_type, severity) -> (alert_id, last_interval_index)
        self.total_intervals_processed = 0
        self.suppressed_duplicate_count = 0

        if self.storage_path and os.path.exists(self.storage_path):
            self.load()

    def process_anomaly(self, anomaly_result, timestamp=None, interval_index=0, actual_usage=0.0, predicted_usage=0.0):
        """
        Processes an AnomalyResult. If anomalous, generates or deduplicates an AlertRecord.
        Returns: (alert_record, is_new_alert)
        """
        self.total_intervals_processed += 1

        if not anomaly_result.is_anomaly:
            return None, False

        # Key for deduplication
        dedup_key = (anomaly_result.anomaly_type, anomaly_result.severity)
        is_duplicate = False

        if dedup_key in self.active_alert_by_key:
            prev_alert_id, prev_interval = self.active_alert_by_key[dedup_key]
            # If within cooldown window and previous alert is still New or Acknowledged
            if (interval_index - prev_interval) <= self.cooldown_intervals:
                # Find previous alert and update occurrence count
                for a in self.alerts:
                    if a.alert_id == prev_alert_id and a.status in ['New', 'Acknowledged']:
                        a.occurrence_count += 1
                        a.last_seen_timestamp = str(timestamp)
                        self.active_alert_by_key[dedup_key] = (prev_alert_id, interval_index)
                        self.suppressed_duplicate_count += 1
                        is_duplicate = True
                        if self.storage_path:
                            self.save()
                        return a, False

        # Create new alert
        new_alert = AlertRecord(
            timestamp=str(timestamp),
            interval_index=interval_index,
            alert_type=anomaly_result.anomaly_type,
            actual_usage_kWh=actual_usage,
            predicted_usage_kWh=predicted_usage,
            error_kWh=anomaly_result.error_kWh,
            absolute_error_kWh=anomaly_result.absolute_error_kWh,
            percentage_error=anomaly_result.percentage_error,
            severity=anomaly_result.severity,
            explanation=anomaly_result.explanation,
            suggested_action=anomaly_result.suggested_action,
            status='New'
        )

        self.alerts.append(new_alert)
        self.active_alert_by_key[dedup_key] = (new_alert.alert_id, interval_index)

        if self.storage_path:
            self.save()

        return new_alert, True

    def acknowledge_alert(self, alert_id, user="Operator"):
        """Acknowledge an active alert."""
        for a in self.alerts:
            if a.alert_id == alert_id:
                a.status = 'Acknowledged'
                a.acknowledged_at = datetime.now(timezone.utc).isoformat()
                a.acknowledged_by = user
                if self.storage_path:
                    self.save()
                return True
        return False

    def resolve_alert(self, alert_id, user="Operator", resolution_note="Investigated"):
        """Mark an alert as resolved."""
        for a in self.alerts:
            if a.alert_id == alert_id:
                a.status = 'Resolved'
                a.resolved_at = datetime.now(timezone.utc).isoformat()
                a.resolved_by = user
                a.resolution_note = resolution_note
                # Clear from active cooldown key
                dedup_key = (a.alert_type, a.severity)
                if self.active_alert_by_key.get(dedup_key, (None, None))[0] == alert_id:
                    del self.active_alert_by_key[dedup_key]
                if self.storage_path:
                    self.save()
                return True
        return False

    def get_alerts(self, status=None, severity=None, alert_type=None, limit=None, offset=0):
        """Query alerts with filtering and pagination."""
        res = list(reversed(self.alerts))  # Newest first

        if status:
            if isinstance(status, str):
                status = [status.lower()]
            else:
                status = [s.lower() for s in status]
            res = [a for a in res if a.status.lower() in status]

        if severity:
            if isinstance(severity, str):
                severity = [severity.lower()]
            else:
                severity = [s.lower() for s in severity]
            res = [a for a in res if a.severity.lower() in severity]

        if alert_type:
            if isinstance(alert_type, str):
                alert_type = [alert_type.lower()]
            else:
                alert_type = [t.lower() for t in alert_type]
            res = [a for a in res if a.alert_type.lower() in alert_type]

        if offset:
            res = res[offset:]
        if limit:
            res = res[:limit]

        return res

    def get_summary(self):
        """Return high-level summary KPIs for the Alert Center."""
        total_alerts = len(self.alerts)
        active_new = sum(1 for a in self.alerts if a.status == 'New')
        acknowledged = sum(1 for a in self.alerts if a.status == 'Acknowledged')
        resolved = sum(1 for a in self.alerts if a.status == 'Resolved')
        crit_count = sum(1 for a in self.alerts if a.severity == 'Critical' and a.status != 'Resolved')
        high_count = sum(1 for a in self.alerts if a.severity == 'High' and a.status != 'Resolved')

        # Type breakdown
        by_type = {}
        for a in self.alerts:
            by_type[a.alert_type] = by_type.get(a.alert_type, 0) + 1

        # Severity breakdown
        by_severity = {'Low': 0, 'Medium': 0, 'High': 0, 'Critical': 0}
        for a in self.alerts:
            by_severity[a.severity] = by_severity.get(a.severity, 0) + 1

        return {
            'total_intervals_processed': self.total_intervals_processed,
            'total_alerts_generated': total_alerts,
            'active_new_alerts': active_new,
            'acknowledged_alerts': acknowledged,
            'resolved_alerts': resolved,
            'critical_unresolved_alerts': crit_count,
            'high_unresolved_alerts': high_count,
            'suppressed_duplicates_count': self.suppressed_duplicate_count,
            'alerts_by_type': by_type,
            'alerts_by_severity': by_severity
        }

    def clear(self):
        """Reset engine state."""
        self.alerts = []
        self.active_alert_by_key = {}
        self.total_intervals_processed = 0
        self.suppressed_duplicate_count = 0
        if self.storage_path and os.path.exists(self.storage_path):
            os.remove(self.storage_path)

    def save(self):
        """Serialize alert history to JSON."""
        if not self.storage_path:
            return
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        data = {
            'total_intervals_processed': self.total_intervals_processed,
            'suppressed_duplicate_count': self.suppressed_duplicate_count,
            'alerts': [a.to_dict() for a in self.alerts]
        }
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load alert history from JSON."""
        if not self.storage_path or not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            self.total_intervals_processed = data.get('total_intervals_processed', 0)
            self.suppressed_duplicate_count = data.get('suppressed_duplicate_count', 0)
            self.alerts = [AlertRecord.from_dict(d) for d in data.get('alerts', [])]
        except Exception:
            self.alerts = []
