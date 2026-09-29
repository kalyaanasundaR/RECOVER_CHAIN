"""Contact permission: consent, opt-out and suppression.

The minimum the Policy Engine needs to make a truthful contact decision - not a CRM.

- A record says a customer ALLOWED, OPTED_OUT of, or is SUPPRESSED for a channel,
  from a stated source, effective from a stated time. Records are append-only
  (infrastructure.orm.ContactPermissionModel).
- ``communication`` covers customer-facing messages, ``payment`` covers
  merchant-initiated charges (retries), ``all`` covers both.
- OPTED_OUT is channel-scoped: a customer who opts out of messages can still have a
  retry of a payment they authorised.
- SUPPRESSED applies to the customer as a whole (legal hold, dispute, deceased,
  cease-and-desist ...) and must be recorded with channel ``all``. It stays in force
  until a LATER ``all`` record lifts it; a channel-specific ALLOWED does not lift it.
- Otherwise, for each channel the newest effective record among that channel's and the
  ``all`` records applies.
- No record at all is UNKNOWN. UNKNOWN is not consent: what the Policy Engine does
  with it is the explicit MerchantPolicy.unknown_consent_* setting.

This module records no legal or compliance claim; it only makes recorded permissions
enforceable and auditable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime


class PermissionStatus:
    ALLOWED = "ALLOWED"
    OPTED_OUT = "OPTED_OUT"
    SUPPRESSED = "SUPPRESSED"
    UNKNOWN = "UNKNOWN"  # never recorded; means "no record exists"

    RECORDABLE = (ALLOWED, OPTED_OUT, SUPPRESSED)
    BLOCKING = (OPTED_OUT, SUPPRESSED)


class PermissionChannel:
    COMMUNICATION = "communication"
    PAYMENT = "payment"
    ALL = "all"

    RECORDABLE = (COMMUNICATION, PAYMENT, ALL)
    ACTION_CHANNELS = (COMMUNICATION, PAYMENT)  # the channels an action can use


def validate_permission(channel: str, status: str) -> None:
    """Raise ValueError for a record that must not be stored."""
    if channel not in PermissionChannel.RECORDABLE:
        raise ValueError(f"channel must be one of {PermissionChannel.RECORDABLE}")
    if status not in PermissionStatus.RECORDABLE:
        raise ValueError(f"status must be one of {PermissionStatus.RECORDABLE}")
    if status == PermissionStatus.SUPPRESSED and channel != PermissionChannel.ALL:
        raise ValueError(
            "a suppression applies to the customer, not to one channel: record it with "
            "channel 'all'"
        )


def _aware(dt: datetime | None) -> datetime:
    if dt is None:
        return datetime.min.replace(tzinfo=UTC)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _order(record) -> tuple:
    return (_aware(record.effective_at), _aware(record.recorded_at), record.id)


@dataclass
class ResolvedPermission:
    channel: str
    status: str
    record_id: str | None = None
    record_channel: str | None = None
    source: str | None = None
    reason: str | None = None
    effective_at: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_permission(
    records, channel: str, at: datetime | None = None, known_at: datetime | None = None
) -> ResolvedPermission:
    """The permission in force for ``channel`` at ``at`` (default: now).

    ``known_at`` makes this bitemporal: when given, only records that had actually been
    RECORDED by ``known_at`` are considered, so a record entered later (a backdated
    import, a correction) cannot change what an earlier moment is reported to have
    known. Phase 5 requirement: "a later opt-out must not rewrite history to make an
    earlier permitted contact appear forbidden." Omitted (the default), every stored
    record is eligible regardless of when it was recorded - correct for "what applies
    right now" (the Policy Engine calls it this way, with at=now, where every existing
    record was necessarily recorded no later than now anyway).

    ``records`` are objects with id, channel, status, source, reason, effective_at and
    recorded_at attributes (ContactPermissionModel rows)."""
    at = _aware(at or datetime.now(UTC))
    effective = [r for r in records if _aware(r.effective_at) <= at]
    if known_at is not None:
        known_at = _aware(known_at)
        effective = [r for r in effective if _aware(r.recorded_at) <= known_at]
    latest_all = max(
        (r for r in effective if r.channel == PermissionChannel.ALL), key=_order, default=None
    )
    latest_channel = max((r for r in effective if r.channel == channel), key=_order, default=None)

    if latest_all is not None and latest_all.status == PermissionStatus.SUPPRESSED:
        chosen = latest_all
    else:
        candidates = [r for r in (latest_all, latest_channel) if r is not None]
        if not candidates:
            return ResolvedPermission(channel=channel, status=PermissionStatus.UNKNOWN)
        chosen = max(candidates, key=_order)

    return ResolvedPermission(
        channel=channel,
        status=chosen.status,
        record_id=chosen.id,
        record_channel=chosen.channel,
        source=chosen.source,
        reason=chosen.reason,
        effective_at=_aware(chosen.effective_at).isoformat(),
    )


def resolve_all(
    records, at: datetime | None = None, known_at: datetime | None = None
) -> dict[str, ResolvedPermission]:
    records = list(records)
    return {
        ch: resolve_permission(records, ch, at, known_at)
        for ch in PermissionChannel.ACTION_CHANNELS
    }


def content_hash(
    *,
    customer_id: str,
    channel: str,
    status: str,
    source: str,
    reason: str | None,
    effective_at: datetime,
) -> str:
    """A deterministic fingerprint of one permission record's meaning, used only to make
    re-running a consent import idempotent (application.contact_permissions_import). Two
    records with the same fingerprint for the same customer are the same fact reported
    twice, not two events - unlike a repeated real change of mind, which always has a
    different effective_at."""
    import hashlib

    canonical = "|".join(
        [
            customer_id,
            channel,
            status,
            source,
            (reason or "").strip(),
            _aware(effective_at).isoformat(),
        ]
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
