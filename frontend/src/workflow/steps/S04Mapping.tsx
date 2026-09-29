import { useEffect, useMemo, useState } from 'react';
import { StepProps } from '../types';
import { Pill, ErrorNote, Note } from '../../ui';
import {
    confirmColumns,
    getDatasetDomain,
    prepareForCases,
    DatasetDomainContract,
    RevenueDomain,
} from '../../lib/api';
import { confidenceTone } from '../../lib/format';

const ROLES = [
    {
        key: 'CUSTOMER_ID',
        label: 'Customer ID',
        hint: 'who owes the money',
        aliases: ['ACCOUNT_ID', 'ENTITY_ID'],
    },
    { key: 'AMOUNT', label: 'Amount', hint: 'how much', aliases: ['BALANCE'] },
    {
        key: 'TIMESTAMP',
        label: 'Date',
        hint: 'when it failed or was due',
        aliases: ['SETTLEMENT_DATE'],
    },
    { key: 'OUTCOME', label: 'Result', hint: 'paid or failed', aliases: ['TARGET'] },
];

// The revenue-type choice is either one domain for the whole file, or "per row" from a
// category column (sent as an EVENT_CATEGORY mapping).
const PER_ROW = 'PER_ROW:';

export default function S04Mapping({ ctx, patch, next, setAction }: StepProps) {
    const columns: string[] = useMemo(
        () => (ctx.preview?.columns || []).map((c: any) => c.original_name),
        [ctx.preview],
    );
    const detected: any[] = ctx.status?.detected_canonical_fields || [];

    const [pick, setPick] = useState<Record<string, string>>(() => {
        const g: Record<string, string> = {};
        for (const r of ROLES) {
            const hit = detected.find(
                (f: any) => f.canonical_field === r.key || r.aliases.includes(f.canonical_field),
            );
            if (hit) g[r.key] = hit.original_column;
        }
        return g;
    });
    const [err, setErr] = useState('');
    const [contract, setContract] = useState<DatasetDomainContract | null>(null);
    const [domainChoice, setDomainChoice] = useState('');

    // What the server can tell about the file's revenue type. It is only a suggestion:
    // the choice below is sent explicitly, and the server re-classifies every run.
    useEffect(() => {
        getDatasetDomain(ctx.importId!)
            .then((d) => {
                setContract(d);
                const cl = d.domain_classification;
                if (cl && cl.status !== 'NEEDS_MAPPING') {
                    setDomainChoice(
                        cl.mode === 'PER_ROW' ? PER_ROW + cl.category_column : cl.domain || '',
                    );
                }
            })
            .catch(() => setContract(null));
    }, [ctx.importId]);

    const classification = contract?.domain_classification;
    const perRowColumn =
        classification?.category_column || classification?.inferred?.category_column || null;
    const suggested = !!classification && classification.status !== 'NEEDS_MAPPING';

    function confidenceFor(col: string) {
        return detected.find((f: any) => f.original_column === col)?.confidence as
            string | undefined;
    }
    const lowOrMissing = ROLES.filter(
        (r) => !pick[r.key] || ['LOW', 'UNKNOWN'].includes(confidenceFor(pick[r.key]) || 'UNKNOWN'),
    );

    async function confirm() {
        setErr('');
        const missing = ROLES.filter((r) => !pick[r.key]);
        if (missing.length) {
            setErr(`Choose a column for: ${missing.map((m) => m.label).join(', ')}`);
            throw new Error('incomplete');
        }
        if (!domainChoice) {
            setErr('Choose what kind of revenue this file describes.');
            throw new Error('incomplete');
        }
        const categoryColumn = domainChoice.startsWith(PER_ROW)
            ? domainChoice.slice(PER_ROW.length)
            : null;
        const used = new Set([...Object.values(pick), ...(categoryColumn ? [categoryColumn] : [])]);
        const mappings = [
            ...ROLES.map((r) => ({
                original_column: pick[r.key],
                canonical_field: r.key,
                action: 'confirm',
            })),
            ...(categoryColumn
                ? [
                      {
                          original_column: categoryColumn,
                          canonical_field: 'EVENT_CATEGORY',
                          action: 'confirm',
                      },
                  ]
                : []),
            ...columns
                .filter((c) => !used.has(c))
                .map((c) => ({ original_column: c, canonical_field: 'UNKNOWN', action: 'unused' })),
        ];
        try {
            const res = await confirmColumns(
                ctx.importId!,
                mappings,
                categoryColumn ? null : (domainChoice as RevenueDomain),
            );
            const readiness = await prepareForCases(ctx.importId!);
            if (!String(readiness.readiness_status || '').includes('READY')) {
                setErr(
                    `Not usable for a run — ${(readiness.warnings || []).join(' ') || readiness.readiness_status}`,
                );
                throw new Error('not ready');
            }
            patch({ readiness, detail: { ...ctx.detail, __classification: res.classification } });
            next();
        } catch (e: any) {
            if (!['incomplete', 'not ready'].includes(e.message)) setErr(e.message);
            throw e;
        }
    }

    useEffect(() => {
        setAction({ label: 'Confirm mapping →', busy: 'Saving and preparing…', onClick: confirm });
    }, [pick, domainChoice]);

    return (
        <div className="max-w-2xl">
            <h1 className="text-3xl font-bold tracking-tight">Confirm the mapping</h1>
            <p className="mt-3 text-[--muted]">
                Match your columns to the four fields RecoverChain needs, and say what kind of
                revenue the file holds. We’ve guessed from your data — confirm each one.
            </p>

            <div className="mt-8 space-y-4">
                {ROLES.map((r) => {
                    const conf = pick[r.key] ? confidenceFor(pick[r.key]) : undefined;
                    const needsConfirm =
                        !pick[r.key] || ['LOW', 'UNKNOWN'].includes(conf || 'UNKNOWN');
                    return (
                        <div
                            key={r.key}
                            className={`rounded-xl border px-4 py-3.5 ${needsConfirm ? 'border-amber-500/30 bg-amber-500/[0.04]' : 'border-[--line] bg-[--panel]'}`}
                        >
                            <div className="flex items-center justify-between">
                                <div>
                                    <div className="text-sm font-medium">{r.label}</div>
                                    <div className="text-xs text-[--faint]">{r.hint}</div>
                                </div>
                                {conf && (
                                    <Pill tone={confidenceTone(conf)}>
                                        {conf.toLowerCase()} confidence
                                    </Pill>
                                )}
                                {needsConfirm && !conf && <Pill tone="amber">please choose</Pill>}
                            </div>
                            <select
                                value={pick[r.key] || ''}
                                onChange={(e) =>
                                    setPick((p) => ({ ...p, [r.key]: e.target.value }))
                                }
                                className="mt-3 w-full rounded-lg border border-[--line] bg-[--bg] px-3 py-2 text-sm"
                            >
                                <option value="">— choose a column —</option>
                                {columns.map((c) => (
                                    <option key={c} value={c}>
                                        {c}
                                    </option>
                                ))}
                            </select>
                        </div>
                    );
                })}

                <div
                    className={`rounded-xl border px-4 py-3.5 ${!domainChoice || !suggested ? 'border-amber-500/30 bg-amber-500/[0.04]' : 'border-[--line] bg-[--panel]'}`}
                >
                    <div className="flex items-center justify-between">
                        <div>
                            <div className="text-sm font-medium">Revenue type</div>
                            <div className="text-xs text-[--faint]">
                                what kind of money is at risk
                            </div>
                        </div>
                        {suggested && domainChoice ? (
                            <Pill tone="blue">suggested from the data</Pill>
                        ) : (
                            <Pill tone="amber">please choose</Pill>
                        )}
                    </div>
                    <select
                        value={domainChoice}
                        onChange={(e) => setDomainChoice(e.target.value)}
                        className="mt-3 w-full rounded-lg border border-[--line] bg-[--bg] px-3 py-2 text-sm"
                    >
                        <option value="">— choose a type —</option>
                        {(contract?.domains || []).map((d) => (
                            <option key={d.domain} value={d.domain}>
                                {d.label}
                            </option>
                        ))}
                        {perRowColumn && !Object.values(pick).includes(perRowColumn) && (
                            <option value={PER_ROW + perRowColumn}>
                                Varies by row — use column “{perRowColumn}”
                            </option>
                        )}
                    </select>
                    {classification && (
                        <div className="mt-2 text-xs text-[--faint]">
                            {suggested
                                ? `Why: ${classification.reason}`
                                : 'The data doesn’t say what kind of revenue this is, and it is not assumed to be a failed payment.'}
                        </div>
                    )}
                </div>
            </div>

            {err && (
                <div className="mt-5">
                    <ErrorNote>{err}</ErrorNote>
                </div>
            )}

            <div className="mt-8">
                {lowOrMissing.length > 0 ? (
                    <Note tone="amber">
                        {lowOrMissing.length} field(s) were an uncertain guess — please check them
                        before continuing. Nothing is assumed silently; the server re-validates your
                        choices.
                    </Note>
                ) : (
                    <Note>
                        The server re-checks every mapping (columns exist, no duplicate roles, the
                        result column isn’t a post-outcome field).
                    </Note>
                )}
            </div>
        </div>
    );
}
