// Pure aggregation over the real /cases list. No fabricated values — every
// number here is a count or sum of fields the backend returned. Money is summed PER
// CURRENCY: RecoverChain performs no conversion, so amounts in different currencies are
// never added into one figure.

const num = (v: any) => Number(v ?? 0);
const amt = (v: any) => (v && typeof v === 'object' ? Number(v.amount) : Number(v ?? 0));
const ccyOf = (c: any) => String(c.currency || 'INR').toUpperCase();

export type ByCurrency = Record<string, number>;

export interface Group {
    key: string;
    count: number;
    atRiskByCcy: ByCurrency;
    recoveredByCcy: ByCurrency;
    avgProb: number | null;
}

const addTo = (m: ByCurrency, ccy: string, v: number) => {
    m[ccy] = (m[ccy] || 0) + v;
};

function groupBy(cases: any[], field: string, singleCurrency: string | null): Group[] {
    const map = new Map<
        string,
        {
            count: number;
            atRiskByCcy: ByCurrency;
            recoveredByCcy: ByCurrency;
            probSum: number;
            probN: number;
        }
    >();
    for (const c of cases) {
        const k = c[field] || '—';
        const g = map.get(k) || {
            count: 0,
            atRiskByCcy: {},
            recoveredByCcy: {},
            probSum: 0,
            probN: 0,
        };
        g.count++;
        addTo(g.atRiskByCcy, ccyOf(c), num(c.amount_at_risk));
        addTo(g.recoveredByCcy, ccyOf(c), amt(c.actual_amount_recovered));
        if (c.recovery_probability != null) {
            g.probSum += num(c.recovery_probability);
            g.probN++;
        }
        map.set(k, g);
    }
    // Ordered by money only when every case shares one currency; otherwise by count.
    const size = (g: Group) => (singleCurrency ? g.atRiskByCcy[singleCurrency] || 0 : g.count);
    return [...map.entries()]
        .map(([key, g]) => ({
            key,
            count: g.count,
            atRiskByCcy: g.atRiskByCcy,
            recoveredByCcy: g.recoveredByCcy,
            avgProb: g.probN ? g.probSum / g.probN : null,
        }))
        .sort((a, b) => size(b) - size(a));
}

export function computeInsights(cases: any[]) {
    const total = cases.length;
    const currencies = [...new Set(cases.map(ccyOf))].sort();
    const singleCurrency = currencies.length === 1 ? currencies[0] : null;
    const atRiskByCcy: ByCurrency = {};
    const recoveredByCcy: ByCurrency = {};
    for (const c of cases) {
        addTo(atRiskByCcy, ccyOf(c), num(c.amount_at_risk));
        addTo(recoveredByCcy, ccyOf(c), amt(c.actual_amount_recovered));
    }
    const withOutcome = cases.filter((c) => c.outcome_status);
    const executed = cases.filter((c) => c.execution_status === 'COMPLETED_SIMULATED');
    const escalated = cases.filter((c) => c.policy_status === 'ESCALATE');
    const permitted = cases.filter((c) => c.policy_status === 'PERMITTED');
    const auto = total ? permitted.length / total : 0;

    return {
        total,
        currencies,
        singleCurrency,
        atRiskByCcy,
        recoveredByCcy,
        outcomeCounts: {
            FULLY_RECOVERED: cases.filter((c) => c.outcome_status === 'FULLY_RECOVERED').length,
            PARTIALLY_RECOVERED: cases.filter((c) => c.outcome_status === 'PARTIALLY_RECOVERED')
                .length,
            NOT_RECOVERED: cases.filter((c) => c.outcome_status === 'NOT_RECOVERED').length,
            PENDING_VERIFICATION: cases.filter((c) => c.outcome_status === 'PENDING_VERIFICATION')
                .length,
            CANCELLED: cases.filter((c) => c.outcome_status === 'CANCELLED').length,
            none: total - withOutcome.length,
        },
        policyCounts: {
            PERMITTED: permitted.length,
            ESCALATE: escalated.length,
            WAIT: cases.filter((c) => c.policy_status === 'WAIT').length,
            DENIED: cases.filter((c) => c.policy_status === 'DENIED').length,
            none: cases.filter((c) => !c.policy_status).length,
        },
        automationRate: auto,
        executedCount: executed.length,
        escalatedCount: escalated.length,
        byCause: groupBy(cases, 'cause_category', singleCurrency),
        byRisk: groupBy(cases, 'risk_level', singleCurrency),
        byAction: groupBy(cases, 'recommended_action', singleCurrency),
        byCategory: groupBy(cases, 'risk_category', singleCurrency),
    };
}
