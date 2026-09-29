import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
    getOverview,
    listCases,
    getSystemSummary,
    getSystemHealth,
    getModels,
    getModelMonitoring,
    downloadCasesCsv,
} from '../lib/api';
import { computeInsights } from '../lib/insights';
import { money, moneyByCurrency, pct, WHY_FAILED, ACTION, CATEGORY, riskWord } from '../lib/format';
import { Card, BigStat, Pill, Spinner, ErrorNote, Empty, Note } from '../ui';

function Bar({
    label,
    value,
    max,
    right,
    tone = 'blue',
}: {
    label: React.ReactNode;
    value: number;
    max: number;
    right?: React.ReactNode;
    tone?: string;
}) {
    const w = max > 0 ? Math.max(2, Math.round((value / max) * 100)) : 0;
    const color =
        tone === 'green'
            ? 'bg-emerald-500/60'
            : tone === 'red'
              ? 'bg-rose-500/60'
              : tone === 'amber'
                ? 'bg-amber-500/60'
                : 'bg-sky-500/60';
    return (
        <div className="py-2">
            <div className="flex items-baseline justify-between gap-4 text-sm">
                <span className="text-[--ink]">{label}</span>
                <span className="tabular text-[--muted]">{right}</span>
            </div>
            <div className="mt-1.5 h-1.5 rounded-full bg-white/5">
                <div className={`h-full rounded-full ${color}`} style={{ width: `${w}%` }} />
            </div>
        </div>
    );
}

export default function Insights() {
    const [d, setD] = useState<any>(null);
    const [err, setErr] = useState('');
    const [loading, setLoading] = useState(true);

    const load = () => {
        setLoading(true);
        setErr('');
        Promise.all([
            getOverview(),
            listCases(500),
            getSystemSummary(),
            getSystemHealth(),
            getModels(),
            getModelMonitoring().catch(() => null), // Phase 14 — advisory panel only, never blocks the page
        ])
            .then(([metrics, cases, summary, health, models, monitoring]) =>
                setD({
                    metrics,
                    cases: cases || [],
                    summary,
                    health,
                    models,
                    monitoring,
                    ins: computeInsights(cases || []),
                }),
            )
            .catch((e) => setErr(e.message))
            .finally(() => setLoading(false));
    };
    useEffect(load, []);

    if (loading) return <Spinner label="Building the report…" />;
    if (err) return <ErrorNote onRetry={load}>{err}</ErrorNote>;

    const { ins, summary: s, health: h, models, monitoring, metrics: m } = d;
    // Headline money comes from the backend aggregate over EVERY case. It used to be
    // summed here from the /cases page (at most 200 cases) and silently undercounted.
    // One headline row per currency: RecoverChain converts nothing, so amounts in
    // different currencies are never added into one figure.
    const breakdown = m.currency_breakdown || {};
    const ccys: string[] = Object.keys(breakdown).sort();
    const headline = ccys.map((ccy) => ({
        ccy,
        atRisk: Number(breakdown[ccy].total_revenue_at_risk ?? 0),
        recovered: Number(breakdown[ccy].verified_recovery ?? 0),
        // Phase 5: "still outstanding" is what real outcome events say is still owed —
        // not at-risk minus recovered, which would count cancelled/withdrawn balances
        // as if they were still collectible.
        outstanding: Number(breakdown[ccy].outstanding_amount ?? 0),
        cancelled: Number(breakdown[ccy].cancelled_amount ?? 0),
    }));
    const unmatchedEvents = Number(m.unmatched_outcome_events ?? 0);
    const unmatchedAmount = Number(m.unmatched_outcome_amount ?? 0);
    const partialBreakdown = s.cases_count > ins.total;

    if (ins.total === 0) {
        return (
            <div className="space-y-6">
                <h1 className="text-2xl font-bold">Insights</h1>
                <Empty title="No recovery runs yet">
                    <Link to="/" className="font-medium text-sky-400 hover:underline">
                        Start a run
                    </Link>{' '}
                    and the report fills in.
                </Empty>
            </div>
        );
    }

    const promoted = (models.items || []).filter((x: any) => x.status === 'SELECTED');
    // Bars are sized by money only when every listed case shares one currency.
    const barSize = (g: any) =>
        ins.singleCurrency ? g.atRiskByCcy[ins.singleCurrency] || 0 : g.count;
    const maxCategorySize = Math.max(...ins.byCategory.map(barSize), 1);
    const maxActionCount = Math.max(...ins.byAction.map((g: any) => g.count), 1);

    return (
        <div className="space-y-10">
            <div>
                <h1 className="text-2xl font-bold">Insights</h1>
                <p className="mt-1 text-[--muted]">
                    Everything {s.cases_count} recovery cases across {s.datasets_count} imported
                    dataset
                    {s.datasets_count === 1 ? '' : 's'} have told us so far.
                </p>
            </div>

            {/* headline — one row per currency */}
            <div className="space-y-6">
                {headline.map(({ ccy, atRisk, recovered, outstanding, cancelled }) => {
                    const suffix = ccys.length > 1 ? ` (${ccy})` : '';
                    return (
                        <div key={ccy} className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
                            <BigStat label={`Money at risk${suffix}`} value={money(atRisk, ccy)} />
                            <BigStat
                                label={`Recovered${suffix}`}
                                value={money(recovered, ccy)}
                                tone="green"
                            />
                            <BigStat
                                label={`Still outstanding${suffix}`}
                                value={money(outstanding, ccy)}
                                tone="red"
                                sub={
                                    cancelled > 0
                                        ? `+ ${money(cancelled, ccy)} cancelled — not expected`
                                        : undefined
                                }
                            />
                            <BigStat
                                label={`Recovery rate${suffix}`}
                                value={pct(atRisk ? recovered / atRisk : 0)}
                                tone="blue"
                                sub="recovered ÷ at risk"
                            />
                        </div>
                    );
                })}
            </div>

            {unmatchedEvents > 0 && (
                <Note tone="amber">
                    {unmatchedEvents} outcome event{unmatchedEvents === 1 ? '' : 's'} (
                    {money(unmatchedAmount)}) could not be matched to any obligation and are not
                    counted as recovered.
                </Note>
            )}

            {(m.mixed_currency || partialBreakdown) && (
                <div className="space-y-2">
                    {m.mixed_currency && (
                        <Note>
                            Figures are shown separately for {ccys.join(', ')}. RecoverChain does
                            not convert currencies, so amounts in different currencies are never
                            added together.
                        </Note>
                    )}
                    {partialBreakdown && (
                        <Note>
                            The breakdowns below cover the latest {ins.total} of {s.cases_count}{' '}
                            cases.
                        </Note>
                    )}
                </div>
            )}

            {/* outcomes */}
            <Card
                title="Recovery outcomes"
                subtitle="Verified against the sandbox settlement source."
            >
                {[
                    ['Recovered in full', ins.outcomeCounts.FULLY_RECOVERED, 'green'],
                    ['Partially recovered', ins.outcomeCounts.PARTIALLY_RECOVERED, 'amber'],
                    ['Not recovered', ins.outcomeCounts.NOT_RECOVERED, 'red'],
                    ['Pending check', ins.outcomeCounts.PENDING_VERIFICATION, 'blue'],
                    // Includes cases with no outcome event yet, whether or not an action
                    // has run — Phase 5: an executed action is not itself a recovered
                    // amount, so it stays here until a real outcome arrives.
                    ['No outcome yet', ins.outcomeCounts.none, 'blue'],
                    ['Cancelled', ins.outcomeCounts.CANCELLED, 'gray'],
                ].map(([label, n, tone]: any) => (
                    <Bar
                        key={label}
                        label={label}
                        value={n}
                        max={ins.total}
                        tone={tone}
                        right={`${n} case${n === 1 ? '' : 's'}`}
                    />
                ))}
            </Card>

            {/* by cause */}
            <Card
                title="Why revenue is failing"
                subtitle="Root cause is diagnosed by the rule-based engine."
            >
                <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead className="text-left text-xs uppercase tracking-wide text-[--faint]">
                            <tr>
                                <th className="pb-2 pr-4">Cause</th>
                                <th className="pb-2 pr-4">Cases</th>
                                <th className="pb-2 pr-4">At risk</th>
                                <th className="pb-2 pr-4">Recovered</th>
                                <th className="pb-2">Avg recovery odds</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-[--line]">
                            {ins.byCause.map((g: any) => (
                                <tr key={g.key}>
                                    <td className="py-2.5 pr-4">{WHY_FAILED[g.key] || g.key}</td>
                                    <td className="py-2.5 pr-4 tabular text-[--muted]">
                                        {g.count}
                                    </td>
                                    <td className="py-2.5 pr-4 tabular text-[--muted]">
                                        {moneyByCurrency(g.atRiskByCcy)}
                                    </td>
                                    <td className="py-2.5 pr-4 tabular text-emerald-300">
                                        {moneyByCurrency(g.recoveredByCcy)}
                                    </td>
                                    <td className="py-2.5 tabular text-sky-300">
                                        {g.avgProb == null ? '—' : pct(g.avgProb)}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </Card>

            {/* handling */}
            <div className="grid gap-6 lg:grid-cols-2">
                <Card
                    title="How cases are handled"
                    subtitle="The PolicyEngine is the sole authority."
                >
                    {[
                        ['Approved — acted automatically', ins.policyCounts.PERMITTED, 'green'],
                        ['Sent for human approval', ins.policyCounts.ESCALATE, 'amber'],
                        ['Waiting (cooling period)', ins.policyCounts.WAIT, 'amber'],
                        ['Blocked by a rule', ins.policyCounts.DENIED, 'red'],
                    ].map(([label, n, tone]: any) => (
                        <Bar
                            key={label}
                            label={label}
                            value={n}
                            max={ins.total}
                            tone={tone}
                            right={`${n}`}
                        />
                    ))}
                    <div className="mt-4 flex items-center gap-3">
                        <BigStat
                            label="Automation rate"
                            value={pct(ins.automationRate)}
                            tone="blue"
                        />
                        <p className="text-sm text-[--muted]">
                            of cases were safe to act on without a person.
                            {ins.escalatedCount > 0 && ` ${ins.escalatedCount} needed sign-off.`}
                        </p>
                    </div>
                </Card>

                <Card title="Actions the system chose">
                    {ins.byAction.map((g: any) => (
                        <Bar
                            key={g.key}
                            label={ACTION[g.key] || g.key}
                            value={g.count}
                            max={maxActionCount}
                            right={`${g.count}`}
                        />
                    ))}
                </Card>
            </div>

            {/* risk + category */}
            <div className="grid gap-6 lg:grid-cols-2">
                <Card title="Risk mix">
                    {ins.byRisk.map((g: any) => {
                        const r = riskWord(g.key);
                        return (
                            <Bar
                                key={g.key}
                                label={<Pill tone={r.tone}>{r.label}</Pill>}
                                value={g.count}
                                max={ins.total}
                                tone={r.tone}
                                right={`${g.count} · ${moneyByCurrency(g.atRiskByCcy)}`}
                            />
                        );
                    })}
                </Card>
                <Card title="Where the risk sits">
                    {ins.byCategory.map((g: any) => (
                        <Bar
                            key={g.key}
                            label={CATEGORY[g.key] || g.key}
                            value={barSize(g)}
                            max={maxCategorySize}
                            right={moneyByCurrency(g.atRiskByCcy)}
                        />
                    ))}
                </Card>
            </div>

            {/* the system */}
            <Card title="How RecoverChain works" subtitle="Straight from the running system.">
                <div className="grid gap-6 sm:grid-cols-3">
                    <div>
                        <div className="text-xs font-semibold uppercase tracking-wide text-[--faint]">
                            Machine learning
                        </div>
                        <div className="mt-1 font-medium">{h.ml_subsystem?.mode}</div>
                        <div className="text-sm text-[--muted]">
                            {h.ml_subsystem?.authority?.toLowerCase().replace(/_/g, ' ')}
                        </div>
                    </div>
                    <div>
                        <div className="text-xs font-semibold uppercase tracking-wide text-[--faint]">
                            Decision authority
                        </div>
                        <div className="mt-1 font-medium">Policy engine</div>
                        <div className="text-sm text-[--muted]">
                            {h.policy_engine?.authority?.toLowerCase().replace(/_/g, ' ')}, gate{' '}
                            {h.policy_engine?.execution_gate?.toLowerCase()}
                        </div>
                    </div>
                    <div>
                        <div className="text-xs font-semibold uppercase tracking-wide text-[--faint]">
                            Execution
                        </div>
                        <div className="mt-1 font-medium">
                            {h.execution_engine?.mode?.toLowerCase().replace(/_/g, ' ')}
                        </div>
                        <div className="text-sm text-[--muted]">
                            {h.execution_engine?.adapter} · live gateways:{' '}
                            {String(h.execution_engine?.live_gateways_connected)}
                        </div>
                    </div>
                </div>

                <div className="mt-6 grid gap-4 text-sm sm:grid-cols-4">
                    {[
                        ['Datasets imported', s.datasets_count],
                        ['Recovery cases', s.cases_count],
                        ['Actions executed', s.executions_count],
                        ['Audit records', s.audit_records_count],
                    ].map(([k, v]: any) => (
                        <div key={k} className="rounded-xl border border-[--line] bg-[--panel] p-3">
                            <div className="text-xs text-[--faint]">{k}</div>
                            <div className="mt-0.5 text-lg font-bold tabular">
                                {Number(v).toLocaleString()}
                            </div>
                        </div>
                    ))}
                </div>

                <div className="mt-6">
                    <Note>
                        {promoted.length} model run{promoted.length === 1 ? '' : 's'} recorded in
                        the registry ({models.total_count} total). No model is promoted to live
                        scoring — recovery probability is produced by the deterministic baseline and
                        shown as advisory only. Execution and settlement are simulated in a sandbox;
                        no live payment gateway is connected.
                    </Note>
                </div>

                {monitoring && (
                    <div className="mt-6 rounded-xl border border-[--line] bg-[--panel] p-4">
                        <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[--faint]">
                            Model monitoring (advisory — not a business outcome)
                        </div>
                        <div className="mt-3 grid gap-4 text-sm sm:grid-cols-2 lg:grid-cols-4">
                            <div>
                                <div className="text-xs text-[--faint]">Current model</div>
                                <div className="mt-0.5 font-medium">
                                    {monitoring.champion
                                        ? `${monitoring.champion.model_version} (${monitoring.champion.model_id})`
                                        : 'None trained'}
                                </div>
                            </div>
                            <div>
                                <div className="text-xs text-[--faint]">Evaluation status</div>
                                <div className="mt-0.5 font-medium">
                                    {monitoring.champion?.promotion_gate?.stage ?? '—'}
                                </div>
                            </div>
                            <div>
                                <div className="text-xs text-[--faint]">Calibration</div>
                                <div className="mt-0.5 font-medium">
                                    {monitoring.champion?.calibrated ? 'Calibrated' : 'Not calibrated'}
                                </div>
                            </div>
                            <div>
                                <div className="text-xs text-[--faint]">Drift vs. training data</div>
                                <div className="mt-0.5 font-medium">
                                    {monitoring.champion?.last_known_train_test_drift === true
                                        ? 'Drift detected at training time'
                                        : monitoring.champion?.last_known_train_test_drift === false
                                          ? 'No drift at training time'
                                          : 'Not computed'}
                                </div>
                            </div>
                            <div>
                                <div className="text-xs text-[--faint]">Shadow / challenger</div>
                                <div className="mt-0.5 font-medium">
                                    Shadow only — no challenger approved for promotion
                                </div>
                            </div>
                            <div>
                                <div className="text-xs text-[--faint]">Production-validation status</div>
                                <div className="mt-0.5 font-medium">
                                    {monitoring.production_validation_status}
                                </div>
                            </div>
                            <div>
                                <div className="text-xs text-[--faint]">Prediction-vs-reality evidence</div>
                                <div className="mt-0.5 font-medium">
                                    {monitoring.prediction_vs_reality.status === 'COMPUTED'
                                        ? `${monitoring.prediction_vs_reality.n} matched cases`
                                        : `Insufficient evidence (${monitoring.prediction_vs_reality.n} matched case${monitoring.prediction_vs_reality.n === 1 ? '' : 's'})`}
                                </div>
                            </div>
                            <div>
                                <div className="text-xs text-[--faint]">Next-best-action model</div>
                                <div className="mt-0.5 font-medium">{monitoring.nba_model_status}</div>
                            </div>
                        </div>
                        <p className="mt-3 text-xs text-[--faint]">
                            Model status describes the prediction pipeline only — it is never the
                            business outcome. Actual recovered amounts come only from authoritative
                            outcome events, shown elsewhere on this page.
                        </p>
                    </div>
                )}
            </Card>

            <div>
                <Link to="/cases" className="text-sm font-medium text-sky-400 hover:underline">
                    See every case →
                </Link>
                <button
                    type="button"
                    onClick={() => downloadCasesCsv().catch((e) => setErr(e.message))}
                    className="ml-5 text-sm font-medium text-sky-400 hover:underline"
                >
                    Download CSV
                </button>
            </div>
        </div>
    );
}
