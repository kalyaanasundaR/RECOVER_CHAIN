import { useEffect, useRef, useState } from 'react';
import { StepProps } from '../types';
import { buildCases, listCases, getCase } from '../../lib/api';
import { Spinner, ErrorNote, Note, Row, BigStat } from '../../ui';
import { CATEGORY, DOMAIN_BASIS } from '../../lib/format';

export default function S06Detection({ ctx, patch, next, setAction }: StepProps) {
    const [done, setDone] = useState<any>(null);
    const [err, setErr] = useState('');
    const started = useRef(false);

    useEffect(() => {
        if (started.current) return;
        started.current = true;
        // Already generated on an earlier visit (or a restored run) — don't
        // re-run generate-cases (every row would be a duplicate). Just show it.
        if (ctx.build?.case_ids?.length) {
            setDone(ctx.build);
            return;
        }
        (async () => {
            try {
                const build = await buildCases(ctx.importId!, 75);
                const all = await listCases();
                const idset = new Set<string>(build.case_ids || []);
                // Order by value, highest first — so the picker's "1 of N" is the
                // highest-value case and step 07 opens on it.
                const mine = (all || [])
                    .filter((c: any) => idset.has(c.case_id))
                    .sort((a: any, b: any) => Number(b.amount_at_risk) - Number(a.amount_at_risk));
                const ids: string[] = mine.map((c: any) => c.case_id);
                const active = mine[0];
                // Prefetch the case AI Analysis will show so step 07 opens instantly
                // (no spinner, no round-trip) instead of loading on arrival.
                let snap: any = undefined;
                if (active?.case_id) {
                    try {
                        snap = await getCase(active.case_id);
                    } catch {
                        /* step 07 will retry */
                    }
                }
                patch({
                    build,
                    caseIds: ids,
                    caseCount: mine.length,
                    activeCaseId: active?.case_id,
                    snap,
                });
                setDone(build);
            } catch (e: any) {
                setErr(e.message);
            }
        })();
    }, []);

    useEffect(() => {
        setAction(done ? { label: 'Proceed →', onClick: next } : null);
    }, [done]);

    const c = done?.counters || {};
    const skips = [
        c.not_at_risk && `${c.not_at_risk} already paid`,
        c.invalid_amount && `${c.invalid_amount} invalid amount`,
        c.invalid_currency && `${c.invalid_currency} unknown currency`,
        c.invalid_entity && `${c.invalid_entity} missing customer`,
        c.invalid_timestamp && `${c.invalid_timestamp} invalid date`,
        c.invalid_target && `${c.invalid_target} missing result`,
        c.ambiguous_target && `${c.ambiguous_target} unclear result`,
        c.case_closed && `${c.case_closed} on an already-closed case`,
        c.unresolved_domain && `${c.unresolved_domain} with an unrecognised revenue type`,
    ].filter(Boolean);
    const dc = done?.domain_classification;
    const domainLine = !dc
        ? null
        : dc.mode === 'PER_ROW'
          ? `Revenue type set row by row — ${DOMAIN_BASIS[dc.basis] || dc.basis} “${dc.category_column}”`
          : `Revenue type: ${CATEGORY[dc.domain] || dc.domain} — ${DOMAIN_BASIS[dc.basis] || dc.basis}`;
    // Duplicates are reported on their own line, so they are not "skipped" here.
    const skipped = (c.rows_skipped ?? 0) - (c.duplicate_event ?? 0);

    return (
        <div className="max-w-2xl">
            <h1 className="text-3xl font-bold tracking-tight">Finding revenue at risk</h1>
            <p className="mt-3 text-[--muted]">
                Every unpaid row becomes a <b className="text-[--ink]">case</b>. Each case runs the
                same checks: how risky it is, why the payment failed, how much is likely to come
                back, what to do about it, and whether we’re allowed to do that automatically.
            </p>

            {!done && !err && (
                <div className="mt-8 space-y-3">
                    <Spinner label="Generating recovery cases…" />
                    <p className="text-sm text-[--muted]">
                        Each row runs the full pipeline — risk, diagnosis, recovery estimate,
                        recommended action, policy check — on the server. A large file can take up
                        to a minute.
                    </p>
                </div>
            )}

            {err && (
                <div className="mt-6">
                    <ErrorNote>{err}</ErrorNote>
                </div>
            )}

            {done && (
                <div className="mt-8 step-in">
                    <BigStat
                        label="Recovery cases created"
                        value={c.cases_created ?? done.cases_generated}
                        tone="blue"
                        sub={`from ${c.rows_seen ?? '—'} rows`}
                    />
                    <ul className="mt-6">
                        {domainLine && <Row ok>{domainLine}</Row>}
                        <Row ok>
                            {c.rows_accepted ?? done.cases_generated} rows became recovery cases
                        </Row>
                        {(c.duplicate_event ?? 0) > 0 && (
                            <Row warn>
                                {c.duplicate_event} rows were already in RecoverChain — their
                                existing cases are shown instead of creating copies
                            </Row>
                        )}
                        {done.truncated && (
                            <Row warn>
                                {c.rows_not_processed} rows after the first {c.rows_seen} were not
                                processed
                            </Row>
                        )}
                        {skipped > 0 && (
                            <Row warn>
                                {skipped} rows skipped
                                {skips.length
                                    ? ` — ${skips.join(', ')}`
                                    : ' — already paid or nothing to recover'}
                            </Row>
                        )}
                    </ul>
                    <div className="mt-8">
                        <Note>
                            Each case ran the real pipeline: risk assessment → cause diagnosis →
                            recovery estimate → recommended action → policy check → (act, if
                            approved) → verify. The next screens walk through the highest-value
                            case.
                        </Note>
                    </div>
                </div>
            )}
        </div>
    );
}
