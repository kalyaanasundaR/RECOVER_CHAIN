import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import {
    getCase,
    getCaseHistory,
    getAgentTrace,
    decideCase,
    executeCase,
    REVIEWER_KEY_STORAGE,
} from '../lib/api';
import {
    money,
    moneyMaybe,
    pct,
    policyVerdict as safetyCheck,
    outcomeVerdict as outcome,
    riskWord,
    WHY_FAILED,
    ACTION,
    CATEGORY,
    DOMAIN_BASIS,
    PERMISSION,
    PROVENANCE_LABEL,
    CONFIDENCE_LABEL,
} from '../lib/format';
import { Card, Pill, Button, Spinner, ErrorNote } from '../ui';

function Line({ label, children }: { label: string; children: React.ReactNode }) {
    return (
        <div className="grid grid-cols-1 gap-1 py-3 sm:grid-cols-3 sm:gap-4">
            <div className="text-sm font-medium text-slate-400">{label}</div>
            <div className="text-sm text-slate-200 sm:col-span-2">{children}</div>
        </div>
    );
}

export default function CaseView() {
    const { caseId = '' } = useParams();
    const [c, setC] = useState<any>(null);
    const [history, setHistory] = useState<any[]>([]);
    const [trace, setTrace] = useState<any>(null);
    const [err, setErr] = useState('');
    const [loading, setLoading] = useState(true);
    const [note, setNote] = useState('');
    const [working, setWorking] = useState('');

    const load = () => {
        setLoading(true);
        setErr('');
        Promise.all([
            getCase(caseId),
            getCaseHistory(caseId),
            getAgentTrace(caseId).catch(() => null), // advisory panel: never blocks the page
        ])
            .then(([snap, h, tr]) => {
                setC(snap);
                setHistory(h || snap.audit_history || []);
                setTrace(tr);
            })
            .catch((e) => setErr(e.message))
            .finally(() => setLoading(false));
    };
    useEffect(load, [caseId]);

    async function decide(decision: 'APPROVE' | 'REJECT') {
        setWorking(decision === 'APPROVE' ? 'Approving…' : 'Rejecting…');
        try {
            await decideCase(caseId, decision, note || '(no note)');
            if (decision === 'APPROVE') {
                // Do not swallow this. The backend can legitimately refuse the execution
                // (policy revalidation, a case that has already acted, a duplicate in
                // flight); silently discarding that left the operator believing the
                // action had run. Running the action is not the end of the story: it
                // only simulates sending a payment retry or a message. Money moving is a
                // separate, later fact - an authoritative outcome event - which is why
                // there is no verify call here any more (Phase 5): "Result" stays "Not
                // run yet" until one actually arrives.
                await executeCase(caseId);
            }
            await load();
            setNote('');
        } catch (e: any) {
            setErr(e.message);
        }
        setWorking('');
    }

    if (loading) return <Spinner label="Loading case…" />;
    if (err || !c) return <ErrorNote onRetry={load}>{err || 'Not found'}</ErrorNote>;

    const s = safetyCheck(c.policy_decision?.status);
    const o = outcome(c.outcome?.status);
    const r = riskWord(c.risk_assessment?.risk_level);
    const rec = c.recommendation?.top_candidate;
    const stopped = c.current_state === 'STOPPED';
    // Recovery is halted: never offer an approval that the backend would refuse.
    const needsOk = !stopped && c.policy_decision?.status === 'ESCALATE';
    // Why the case has its revenue type, and what the customer permits (durable records).
    const basis = (c.domain?.bases || []).map((b: string) => DOMAIN_BASIS[b] || b).join('; ');
    const perm = (channel: string) => {
        const p = c.contact_permission?.[channel];
        return p ? PERMISSION[p.status] || p.status : '—';
    };
    const permUnknown = ['communication', 'payment'].some(
        (ch) => c.contact_permission?.[ch]?.status === 'UNKNOWN',
    );
    const owed = Number(c.outcome?.outstanding_amount?.amount ?? 0);

    return (
        <div className="fade-in mx-auto max-w-3xl space-y-6">
            <Link
                to="/cases"
                className="inline-flex items-center gap-1.5 text-sm text-slate-400 hover:text-slate-200"
            >
                <ArrowLeft size={15} /> All cases
            </Link>

            <div>
                <h1 className="text-2xl font-bold text-slate-100">{c.customer_id}</h1>
                <p className="mt-1 text-slate-400">
                    {CATEGORY[c.risk_category] || c.risk_category} ·{' '}
                    {money(c.amount_at_risk, c.currency)} at risk
                </p>
                <div className="mt-3 flex flex-wrap gap-2">
                    <Pill tone={r.tone}>Risk: {r.label}</Pill>
                    {stopped ? (
                        <Pill tone="gray">Stopped</Pill>
                    ) : (
                        <Pill tone={s.tone}>{s.label}</Pill>
                    )}
                    {c.outcome && <Pill tone={o.tone}>{o.label}</Pill>}
                </div>
            </div>

            {needsOk && (
                <Card title="This one needs your OK" subtitle={s.note}>
                    <p className="text-sm text-slate-300">
                        Suggested action:{' '}
                        <b>{rec ? ACTION[rec.action_type] || rec.action_type : '—'}</b>. Reason it
                        was held: {c.policy_decision?.reason}
                    </p>
                    <input
                        type="password"
                        autoComplete="off"
                        placeholder="Reviewer key (this browser tab only — needed to approve)"
                        onChange={(e) => {
                            try {
                                sessionStorage.setItem(REVIEWER_KEY_STORAGE, e.target.value);
                            } catch {
                                /* storage unavailable: falls back to the default key */
                            }
                        }}
                        className="mt-3 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm"
                    />
                    <textarea
                        value={note}
                        onChange={(e) => setNote(e.target.value)}
                        rows={2}
                        placeholder="Add a note (optional)"
                        className="mt-3 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm"
                    />
                    <div className="mt-3 flex gap-3">
                        <Button onClick={() => decide('APPROVE')} disabled={!!working}>
                            {working || 'Approve & run'}
                        </Button>
                        <Button
                            variant="ghost"
                            onClick={() => decide('REJECT')}
                            disabled={!!working}
                        >
                            Reject
                        </Button>
                    </div>
                </Card>
            )}

            <Card title="What happened">
                <div className="divide-y divide-slate-800">
                    <Line label="What failed">
                        {CATEGORY[c.risk_category] || c.risk_category} of{' '}
                        {money(c.amount_at_risk, c.currency)}
                        {basis && <span className="text-slate-500"> · type {basis}</span>}
                    </Line>
                    <Line label="Obligation">
                        {c.reference_id ? (
                            <>
                                Reference <span className="font-mono">{c.reference_id}</span> —
                                later events with this reference join this case
                            </>
                        ) : (
                            'No reference in the source — not linked to any other event'
                        )}
                    </Line>
                    <Line label="Why it failed">
                        {c.diagnosis
                            ? WHY_FAILED[c.diagnosis.cause_category] || c.diagnosis.cause_category
                            : 'Not worked out yet'}
                    </Line>
                    <Line label="Chance of getting it back">
                        {c.ml_shadow_prediction ? (
                            <>
                                ~{pct(c.ml_shadow_prediction.recovery_probability)}{' '}
                                <span className="text-slate-500">
                                    (a helper estimate — a person always decides)
                                    {c.ml_shadow_prediction.confidence_tier && (
                                        <>
                                            {' '}
                                            ·{' '}
                                            {CONFIDENCE_LABEL[c.ml_shadow_prediction.confidence_tier] ||
                                                c.ml_shadow_prediction.confidence_tier}
                                        </>
                                    )}
                                </span>
                            </>
                        ) : (
                            '—'
                        )}
                    </Line>
                    {c.ml_shadow_prediction?.explanation?.length > 0 && (
                        <Line label="Why the model estimated this">
                            <span className="text-slate-500">
                                {c.ml_shadow_prediction.explanation
                                    .filter(
                                        (e: any) =>
                                            e.contribution !== null && Math.abs(e.contribution) >= 0.005
                                    )
                                    .slice(0, 3)
                                    .map((e: any) => `${e.feature} ${e.direction} it`)
                                    .join(' · ') || 'No single factor dominated'}
                                {' — model association, not a causation claim'}
                            </span>
                        </Line>
                    )}
                    <Line label="Likely recovery amount">
                        {c.expected_recoverable_value
                            ? money(c.expected_recoverable_value, c.currency)
                            : '—'}
                    </Line>
                    {trace?.agent && (
                        <Line label="Handled by">
                            {trace.agent.agent_id}{' '}
                            <span className="text-slate-500">
                                (v{trace.agent.agent_version} · proposes only — the Policy Engine
                                decides)
                            </span>
                        </Line>
                    )}
                    {trace && (
                        <Line label="Trace">
                            <span className="font-mono text-xs">{trace.trace_id}</span>{' '}
                            <span className="text-slate-500">· {trace.journey?.label}</span>
                        </Line>
                    )}
                    {trace?.stages?.approval?.status === 'OCCURRED' && (
                        <Line label="Approved by">
                            {trace.stages.approval.reviews?.slice(-1)[0]?.actor?.principal_id}{' '}
                            <span className="text-slate-500">
                                ({trace.stages.approval.reviews?.slice(-1)[0]?.actor?.role} ·
                                approval is not recovery)
                            </span>
                        </Line>
                    )}
                    {trace?.stages?.revalidation &&
                        trace.stages.revalidation.status !== 'NOT_OCCURRED' && (
                            <Line label="Revalidation">
                                {trace.stages.revalidation.latest_result ||
                                    trace.stages.revalidation.status}{' '}
                                <span className="text-slate-500">
                                    · policy{' '}
                                    {trace.stages.revalidation.revalidations?.slice(-1)[0]
                                        ?.policy_version ||
                                        trace.stages.policy?.policy_version}
                                </span>
                            </Line>
                        )}
                    <Line label="What we’ll try">
                        {rec
                            ? ACTION[rec.action_type] || rec.action_type
                            : 'Nothing suitable found'}
                    </Line>
                    <Line label="Contact permission">
                        Messages: {perm('communication')} · Payment retries: {perm('payment')}
                        {permUnknown && (
                            <span className="text-slate-500">
                                {' '}
                                — “no record” is not consent; the merchant policy decides
                            </span>
                        )}
                    </Line>
                    <Line label="Safety check">
                        <span className="font-medium">{s.label}.</span> {s.note}
                    </Line>
                    <Line label="Action taken">
                        {c.execution_record
                            ? `${c.execution_record.status === 'COMPLETED_SIMULATED' ? 'Done (simulated)' : c.execution_record.status}`
                            : 'Not yet'}
                    </Line>
                    <Line label="Result">
                        {c.outcome ? (
                            <>
                                {o.label} —{' '}
                                {moneyMaybe(c.outcome.actual_amount_recovered, c.currency)} of{' '}
                                {moneyMaybe(c.outcome.expected_amount, c.currency)} recovered
                                {c.outcome.provenance && c.outcome.provenance !== 'production' && (
                                    <span className="text-amber-400">
                                        {' '}
                                        ·{' '}
                                        {PROVENANCE_LABEL[c.outcome.provenance] ||
                                            c.outcome.provenance}
                                    </span>
                                )}
                                {owed > 0 && c.outcome.status !== 'PENDING_VERIFICATION' && (
                                    <span className="text-slate-500">
                                        {' '}
                                        · {moneyMaybe(
                                            c.outcome.outstanding_amount,
                                            c.currency,
                                        )}{' '}
                                        still owed
                                    </span>
                                )}
                            </>
                        ) : c.execution_record?.status === 'COMPLETED_SIMULATED' ? (
                            <>
                                Awaiting an outcome —{' '}
                                <span className="text-slate-500">
                                    the action ran, but no settlement has been reported yet. Running
                                    the action is not by itself money recovered.
                                </span>
                            </>
                        ) : (
                            'Not run yet'
                        )}
                    </Line>
                </div>
            </Card>

            {trace?.stages && (
                <Card title="Trace" subtitle="Read-only reconstruction — stages that did not happen say so">
                    <ol className="space-y-2 text-sm">
                        {[
                            ['agent', 'Agent'],
                            ['ml_prediction', 'AI prediction'],
                            ['policy', 'Policy'],
                            ['approval', 'Approval'],
                            ['revalidation', 'Revalidation'],
                            ['execution', 'Execution'],
                            ['outcome', 'Outcome'],
                        ].map(([k, label]) => (
                            <li key={k} className="flex justify-between gap-4">
                                <span className="text-slate-300">{label}</span>
                                <span className="text-slate-400">
                                    {trace.stages[k]?.status}
                                    {k === 'policy' && trace.stages.policy?.decision
                                        ? ` · ${trace.stages.policy.decision}`
                                        : ''}
                                </span>
                            </li>
                        ))}
                    </ol>
                </Card>
            )}

            <Card title="History">
                <ol className="space-y-2 text-sm">
                    {history.map((h, i) => (
                        <li key={i} className="flex justify-between gap-4 text-slate-400">
                            <span className="text-slate-300">
                                {h.evidence?.action || h.to_state}
                            </span>
                            <span className="tabular-nums text-slate-500">
                                {h.timestamp ? new Date(h.timestamp).toLocaleTimeString() : ''}
                            </span>
                        </li>
                    ))}
                </ol>
            </Card>
        </div>
    );
}
