# Archived — pre-migration database backups

Manual `.bak_*` snapshots taken before earlier schema migrations (Phase 11,
Phase 16). Not gitignored (`*.db` doesn't match a `.db.bak_*` suffix), so
they were sitting as untracked disk clutter at the backend root. Moved here
rather than deleted — a database backup is not something this project
deletes speculatively (see repo's own standing rule: never delete the DB as
a cache workaround). Safe to delete for real once you've confirmed you will
never need to roll back past those migrations.
