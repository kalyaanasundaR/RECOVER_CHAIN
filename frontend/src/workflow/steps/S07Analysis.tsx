import { useEffect } from 'react';
import { StepProps } from '../types';
import { useSnap } from './useSnap';
import { Spinner, ErrorNote, Note, BigStat } from '../../ui';
import { money, moneyMaybe, pct, WHY_FAILED, CATEGORY, CONFIDENCE_LABEL } from '../../lib/format';

export default function S07Analysis({ ctx, patch, next, setAction }: StepProps) {
    const { snap, loading, err, refresh } = useSnap(ctx, patch);
    useEffect(() => {
        setAction(snap ? { label: 'Proceed →', onClick: next } : null);
    }, [snap]);

    if (loading || !snap)
        return err ? (
            <ErrorNote onRetry={refresh}>{err}</ErrorNote>
        ) : (
            <Spinner label="Loading case…" />
        );

    const dx = snap.diagnosis;
    const pred = snap.ml_shadow_prediction;
    const ccy = snap.currency || 'INR';
    const ids: string[] = ctx.caseIds || [];
    const many = ids.length > 1;
    const pos = ids.indexOf(ctx.activeCaseId || '') + 1;

    return (
        <div className="max-w-2xl">
            <div className="flex items-center justify-between gap-3">
                <span className="font-mono text-sm text-[--faint]">{snap.case_id}</span>
                {many ? (
                    <label className="flex items-center gap-2 text-xs text-[--faint]">
                        case
                        <select
                            value={ctx.activeCaseId || ''}
                            onChange={(e) => patch({ activeCaseId: e.target.value })}
                            className="max-w-[11rem] rounded-md border border-[--line] bg-[--bg] px-2 py-1 font-mono text-xs text-[--ink]"
                        >
                            {ids.map((id, i) => (
                                <option key={id} value={id}>
                                    {i + 1}/{ids.length} · {id}
                                </option>
                            ))}
                        </select>
                    </label>
                ) : (
                    (ctx.caseCount || 1) > 1 && (
                        <span className="text-xs text-[--faint]">
                            highest-value of {ctx.caseCount} cases
                        </span>
                    )
                )}
            </div>
            {many && (
                <p className="mt-1 text-xs text-[--faint]">
                    {pos <= 1
                        ? 'Showing the highest-value case.'
                        : `Showing case ${pos} of ${ids.length}.`}{' '}
                    Steps 8–10 continue with whichever you pick.
                </p>
            )}

            <div className="mt-3">
                <BigStat
                    label={`${snap.customer_id} · ${CATEGORY[snap.risk_category] || snap.risk_category}`}
                    value={`${money(snap.amount_at_risk, ccy)}`}
                    tone="red"
                    sub="at risk"
                />
            </div>

            <p className="mt-6 rounded-xl border border-[--line] bg-[--panel] px-4 py-3 text-sm leading-relaxed text-[--muted]">
                In plain words: <b className="text-[--ink]">{snap.customer_id}</b> has{' '}
                <b className="text-[--ink]">{money(snap.amount_at_risk, ccy)}</b> unpaid
                {dx && (
                    <>
                        {' '}
                        because {(WHY_FAILED[dx.cause_category] || dx.cause_category).toLowerCase()}
                    </>
                )}
                .
                {pred != null && (
                    <>
                        {' '}
                        From past cases like this, about{' '}
                        <b className="text-[--ink]">{pct(pred.recovery_probability)}</b> is likely
                        to come back
                        {snap.expected_recoverable_value && (
                            <>
                                {' '}
                                — roughly{' '}
                                <b className="text-[--ink]">
                                    {moneyMaybe(snap.expected_recoverable_value, ccy)}
                                </b>
                            </>
                        )}
                        .
                    </>
                )}
            </p>

            <div className="mt-8 grid gap-8 sm:grid-cols-3">
                <div>
                    <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[--faint]">
                        Why it’s at risk
                    </div>
                    <div className="mt-2 text-lg font-medium">
                        {dx ? WHY_FAILED[dx.cause_category] || dx.cause_category : 'Not determined'}
                    </div>
                    {dx && (
                        <div className="mt-1 text-sm text-[--muted]">
                            Confidence {pct(dx.confidence)}
                        </div>
                    )}
                </div>
                <div>
                    <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[--faint]">
                        Recovery probability
                    </div>
                    <div className="mt-2 text-lg font-medium text-sky-300">
                        {pred ? pct(pred.recovery_probability) : '—'}
                    </div>
                    <div className="mt-1 text-sm text-[--muted]">
                        advisory estimate
                        {pred?.confidence_tier && (
                            <> · {CONFIDENCE_LABEL[pred.confidence_tier] || pred.confidence_tier}</>
                        )}
                    </div>
                </div>
                <div>
                    <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[--faint]">
                        Expected recovery
                    </div>
                    <div className="mt-2 text-lg font-medium text-emerald-300">
                        {snap.expected_recoverable_value
                            ? moneyMaybe(snap.expected_recoverable_value, ccy)
                            : '—'}
                    </div>
                    <div className="mt-1 text-sm text-[--muted]">amount × probability</div>
                </div>
            </div>

            {pred?.explanation && pred.explanation.length > 0 && (
                <div className="mt-8">
                    <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[--faint]">
                        Why the model estimated this
                    </div>
                    <ul className="mt-2 space-y-1 text-sm text-[--muted]">
                        {pred.explanation
                            .filter((c: any) => c.contribution !== null && Math.abs(c.contribution) >= 0.005)
                            .slice(0, 3)
                            .map((c: any) => (
                                <li key={c.feature}>
                                    <span className="text-[--ink]">{c.feature}</span>{' '}
                                    {c.direction === 'increased' ? 'increased' : 'decreased'} the
                                    estimate
                                </li>
                            ))}
                    </ul>
                    <p className="mt-1 text-xs text-[--faint]">
                        Model association only — not a claim about what actually caused this
                        customer’s behaviour.
                    </p>
                </div>
            )}

            <div className="mt-10">
                <Note>
                    Risk level and cause come from RecoverChain’s <b>rule-based</b> engines
                    (deterministic-v1.0). The recovery probability is an <b>advisory model</b> (
                    {pred?.model_version || 'baseline'}, running shadow-only) — it never decides an
                    action; a person or the policy engine does.
                </Note>
            </div>
        </div>
    );
}
