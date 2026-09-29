import { useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { StepProps } from '../types';
import { useSnap } from './useSnap';
import { Spinner, ErrorNote, Note, KV, BigStat, Button, CountUp } from '../../ui';
import { money, moneyMaybe, ACTION, outcomeVerdict, PROVENANCE_LABEL } from '../../lib/format';
import { useMotionPref } from '../../lib/motion';
import { markWorkflowDone } from '../../lib/progress';

export default function S11Result({ ctx, patch, setAction }: StepProps) {
    const nav = useNavigate();
    const [motion] = useMotionPref();
    const { snap, loading, err, refresh } = useSnap(ctx, patch);
    // The recovered-amount count-up is a reveal animation for the moment
    // verification just happened. It must not replay every time this step is
    // revisited or the page is reloaded: the amount is a decided financial
    // fact by then, and animating from ₹0 again means a quick glance (or a
    // screenshot) briefly sees the wrong number even though the persisted,
    // authoritative result never changed. `useRef` captures this once, at
    // first render, from whatever `snap` already resolved to synchronously.
    const outcomeAlreadyKnown = useRef(!!snap?.outcome);

    // reaching the Verified Result step completes the process — unlock the report
    useEffect(() => {
        setAction(null);
        markWorkflowDone();
    }, []);

    if (loading || !snap) {
        return err ? (
            <ErrorNote onRetry={refresh}>{err}</ErrorNote>
        ) : (
            <Spinner label="Loading result…" />
        );
    }

    const o = snap.outcome;
    const ccy = snap.currency || 'INR';
    const top = snap.recommendation?.top_candidate;
    const v = outcomeVerdict(o?.status);
    // The action can complete (simulated) with no outcome event on record yet — that is
    // a true, distinct state, not "nothing happened" (Phase 5: ACTION ATTEMPT != REVENUE
    // RECOVERED). Only once a real outcome event has been ingested for this obligation
    // does `o` exist.
    const actionCompletedNoOutcomeYet =
        !o && snap.execution_record?.status === 'COMPLETED_SIMULATED';

    return (
        <div className="max-w-2xl">
            <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[--faint]">
                Recovery result
            </div>

            {o ? (
                <>
                    <div className="mt-3">
                        <BigStat label={snap.customer_id} value={v.label} tone={v.tone as any} />
                    </div>
                    <div className="mt-8 rounded-2xl border border-[--line] bg-[--panel] px-5 py-2">
                        <KV k="Original amount at risk" v={moneyMaybe(o.expected_amount, ccy)} />
                        <KV
                            k="Actual amount recovered"
                            v={
                                <span className="text-emerald-300 tabular">
                                    <CountUp
                                        motion={motion && !outcomeAlreadyKnown.current}
                                        to={Number(
                                            o.actual_amount_recovered?.amount ??
                                                o.actual_amount_recovered ??
                                                0,
                                        )}
                                        format={(n) => money(n, ccy)}
                                    />
                                </span>
                            }
                        />
                        {Number(o.outstanding_amount?.amount ?? 0) > 0 && (
                            <KV k="Still outstanding" v={moneyMaybe(o.outstanding_amount, ccy)} />
                        )}
                        <KV
                            k="Recovery action"
                            v={top ? ACTION[top.action_type] || top.action_type : '—'}
                        />
                        <KV k="Final status" v={v.label} />
                        <KV
                            k="Reconciliation"
                            v={
                                <span className="text-sm text-[--muted]">
                                    {o.reconciliation_status}
                                </span>
                            }
                        />
                    </div>
                    <div className="mt-8">
                        <Note>
                            Verified against a recorded outcome event (
                            {o.verification_source === 'RECORDED_OUTCOME_EVENTS'
                                ? 'an authoritative settlement'
                                : o.verification_source}
                            ). The outcome is checked independently — running an action does not by
                            itself mean money was recovered.
                            {o.provenance && o.provenance !== 'production' && (
                                <>
                                    {' '}
                                    This figure is from{' '}
                                    {PROVENANCE_LABEL[o.provenance] || o.provenance} — not
                                    production evidence.
                                </>
                            )}
                        </Note>
                    </div>
                </>
            ) : actionCompletedNoOutcomeYet ? (
                <>
                    <div className="mt-3">
                        <BigStat
                            label={snap.customer_id}
                            value="Action completed — outcome pending"
                            tone="blue"
                        />
                    </div>
                    <div className="mt-6 rounded-2xl border border-[--line] bg-[--panel] px-5 py-4 text-sm text-[--muted]">
                        {top ? ACTION[top.action_type] || top.action_type : 'The action'} ran in the
                        sandbox (simulated) — no real charge or message was sent, and no money has
                        been recorded as recovered yet. That only happens once an authoritative
                        outcome event (a settlement) is received for this obligation.
                    </div>
                </>
            ) : (
                <>
                    <div className="mt-3">
                        <BigStat
                            label={snap.customer_id}
                            value="No recovery attempted"
                            tone="gray"
                        />
                    </div>
                    <div className="mt-6 rounded-2xl border border-[--line] bg-[--panel] px-5 py-4 text-sm text-[--muted]">
                        {snap.policy_decision?.reason ||
                            'The policy check did not approve an action, so nothing was executed.'}
                    </div>
                </>
            )}

            <div className="mt-10 flex flex-wrap gap-3">
                <Button onClick={() => nav('/cases')}>
                    Review all {ctx.caseCount ?? ''} cases →
                </Button>
                <Button variant="ghost" onClick={() => nav('/insights')}>
                    Open the report
                </Button>
            </div>
        </div>
    );
}
