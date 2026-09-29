import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { listCases } from '../lib/api';
import {
    money,
    pct,
    policyVerdict as safetyCheck,
    outcomeVerdict as outcome,
    riskWord,
    ACTION,
} from '../lib/format';
import { Card, Pill, Spinner, ErrorNote, Empty } from '../ui';

const FILTERS = [
    { key: 'all', label: 'All' },
    { key: 'need', label: 'Need your OK' },
    { key: 'running', label: 'In progress' },
    { key: 'done', label: 'Finished' },
];

export default function CasesList() {
    const [cases, setCases] = useState<any[]>([]);
    const [err, setErr] = useState('');
    const [loading, setLoading] = useState(true);
    const [filter, setFilter] = useState('all');
    const [q, setQ] = useState('');

    const load = () => {
        setLoading(true);
        setErr('');
        listCases()
            .then((d) => setCases(d || []))
            .catch((e) => setErr(e.message))
            .finally(() => setLoading(false));
    };
    useEffect(load, []);

    const rows = useMemo(
        () =>
            cases.filter((c) => {
                const text = `${c.customer_id} ${c.case_id}`.toLowerCase();
                if (q && !text.includes(q.toLowerCase())) return false;
                // A stopped case is halted, not in progress, and it is not waiting on
                // anyone's approval — it belongs with the finished work.
                const stopped = c.current_state === 'STOPPED';
                if (filter === 'need') return !stopped && c.policy_status === 'ESCALATE';
                if (filter === 'done') return stopped || !!c.outcome_status;
                if (filter === 'running')
                    return !stopped && !c.outcome_status && c.policy_status !== 'ESCALATE';
                return true;
            }),
        [cases, filter, q],
    );

    if (loading) return <Spinner label="Loading cases…" />;
    if (err) return <ErrorNote onRetry={load}>{err}</ErrorNote>;

    return (
        <div className="fade-in space-y-6">
            <div>
                <h1 className="text-2xl font-bold text-slate-100">Cases</h1>
                <p className="mt-1 text-slate-400">
                    One row per recovery case. Chance-back is the estimate; Recovered is what the
                    sandbox settled.
                </p>
            </div>

            <div className="flex flex-wrap items-center gap-2">
                {FILTERS.map((f) => (
                    <button
                        key={f.key}
                        onClick={() => setFilter(f.key)}
                        className={`rounded-lg px-3 py-1.5 text-sm font-medium ${filter === f.key ? 'bg-blue-600 text-white' : 'bg-slate-900 text-slate-400 hover:text-slate-200'}`}
                    >
                        {f.label}
                    </button>
                ))}
                <input
                    value={q}
                    onChange={(e) => setQ(e.target.value)}
                    placeholder="Search customer…"
                    className="ml-auto w-56 rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-sm"
                />
            </div>

            <Card>
                {rows.length === 0 ? (
                    <Empty title="No cases here">Start a recovery run to create some.</Empty>
                ) : (
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                            <thead className="text-left text-xs uppercase tracking-wide text-slate-500">
                                <tr>
                                    <th className="pb-2 pr-4">Customer</th>
                                    <th className="pb-2 pr-4">Amount</th>
                                    <th className="pb-2 pr-4">Risk</th>
                                    <th className="pb-2 pr-4">Chance back</th>
                                    <th className="pb-2 pr-4">What we’ll try</th>
                                    <th className="pb-2 pr-4">Status</th>
                                    <th className="pb-2 pr-4">Result</th>
                                    <th className="pb-2">Recovered</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-800">
                                {rows.map((c) => {
                                    const s = safetyCheck(c.policy_status);
                                    const o = outcome(c.outcome_status);
                                    const r = riskWord(c.risk_level);
                                    return (
                                        <tr key={c.case_id} className="hover:bg-slate-800/40">
                                            <td className="py-3 pr-4">
                                                <Link
                                                    to={`/cases/${c.case_id}`}
                                                    className="font-medium text-slate-100 hover:text-blue-400"
                                                >
                                                    {c.customer_id}
                                                </Link>
                                            </td>
                                            <td className="py-3 pr-4 tabular-nums text-slate-300">
                                                {money(c.amount_at_risk, c.currency)}
                                            </td>
                                            <td className="py-3 pr-4">
                                                <Pill tone={r.tone}>{r.label}</Pill>
                                            </td>
                                            <td className="py-3 pr-4 tabular-nums text-slate-300">
                                                {pct(c.recovery_probability)}
                                            </td>
                                            <td className="py-3 pr-4 text-slate-300">
                                                {c.recommended_action
                                                    ? ACTION[c.recommended_action] ||
                                                      c.recommended_action
                                                    : '—'}
                                            </td>
                                            <td className="py-3 pr-4">
                                                {/* A halted case must not advertise the
                                                    policy's last verdict as if recovery
                                                    were still running. */}
                                                {c.current_state === 'STOPPED' ? (
                                                    <Pill tone="gray">Stopped</Pill>
                                                ) : (
                                                    <Pill tone={s.tone}>{s.label}</Pill>
                                                )}
                                            </td>
                                            <td className="py-3 pr-4">
                                                {c.outcome_status ? (
                                                    <Pill tone={o.tone}>{o.label}</Pill>
                                                ) : (
                                                    <span className="text-slate-600">—</span>
                                                )}
                                            </td>
                                            <td className="py-3 tabular-nums text-emerald-300">
                                                {c.actual_amount_recovered ? (
                                                    money(c.actual_amount_recovered, c.currency)
                                                ) : (
                                                    <span className="text-slate-600">—</span>
                                                )}
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                )}
            </Card>
        </div>
    );
}
