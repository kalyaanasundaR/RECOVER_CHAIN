// One place for every backend call. Endpoints unchanged; names are plain.
const BASE = '/api';

// The key is build-time config, not a source constant. Set VITE_API_KEY in the
// deploy environment; the fallback only matches the backend's local dev default.
const API_KEY = import.meta.env.VITE_API_KEY ?? 'test-api-key';
// CONFIDENTIALITY (Phase 17): API_KEY is bundled into the browser. It is a DEVELOPMENT-ONLY
// credential and is NOT a secret - anyone who can load this app can read it. Never put a
// production credential here. Privileged actions do not rely on it: approving an escalation
// needs a REVIEWER credential that the reviewer types in at runtime (kept in sessionStorage
// for this browser tab only, never bundled, never persisted). Server-side role checks decide;
// hiding a button is not authorization.
const KEY = { 'X-API-Key': API_KEY };
export const REVIEWER_KEY_STORAGE = 'rc-reviewer-key';
const reviewerHeaders = () => {
    let k: string | null = null;
    try {
        k = sessionStorage.getItem(REVIEWER_KEY_STORAGE);
    } catch {
        k = null;
    }
    return { 'X-API-Key': k || API_KEY, 'Content-Type': 'application/json' };
};
const JSON_KEY = { ...KEY, 'Content-Type': 'application/json' };

async function j(res: Response) {
    if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        const d = body.detail;
        // A structured error (e.g. DOMAIN_NEEDS_MAPPING) carries its own message.
        const msg = typeof d === 'string' ? d : d?.message || (d ? JSON.stringify(d) : '');
        throw new Error(msg || `${res.status} ${res.statusText}`);
    }
    return res.json();
}

// -- domain semantics -------------------------------------------------------------
export type RevenueDomain =
    | 'FAILED_PAYMENT'
    | 'CHECKOUT_ABANDONMENT'
    | 'FAILED_SUBSCRIPTION'
    | 'OVERDUE_INVOICE'
    | 'BROKEN_PROMISE';

// How a dataset's rows get their revenue domain (backend domain_classification.py).
export interface DomainClassification {
    status: 'DECLARED' | 'INFERRED' | 'NEEDS_MAPPING';
    mode: 'DATASET' | 'PER_ROW' | null;
    domain: RevenueDomain | null;
    canonical_domain: string | null;
    basis: string | null;
    category_column: string | null;
    value_domains: Record<string, RevenueDomain>;
    unresolved_values: string[];
    reason: string;
    evidence: Record<string, unknown>[];
    warnings: string[];
    inferred: Partial<DomainClassification> | null;
    rows_examined: number;
    classifier_version: string;
}

// GET /datasets/{id}/domain — the dataset category contract.
export interface DatasetDomainContract {
    dataset_id: string;
    declared_domain: RevenueDomain | null;
    source_system: string | null;
    schema: { original_column: string; canonical_field: string }[];
    content_sha256: string | null;
    domain_classification: DomainClassification | null;
    domains: { domain: RevenueDomain; canonical_name: string; label: string }[];
}

// A customer's contact permission for one channel, resolved from durable records.
export interface ResolvedPermission {
    channel: 'communication' | 'payment';
    status: 'ALLOWED' | 'OPTED_OUT' | 'SUPPRESSED' | 'UNKNOWN';
    record_id: string | null;
    record_channel: string | null;
    source: string | null;
    reason: string | null;
    effective_at: string | null;
}

// -- overview / report -------------------------------------------------------
export const getOverview = () => fetch(`${BASE}/dashboard/metrics`).then(j);
export const getSystemSummary = () => fetch(`${BASE}/system/summary`).then(j);
export const getSystemHealth = () => fetch(`${BASE}/system/health`).then(j);
export const getModels = () => fetch(`${BASE}/system/models`, { headers: KEY }).then(j);
export const getModelMonitoring = () =>
    fetch(`${BASE}/system/model-monitoring`, { headers: KEY }).then(j);

// -- cases ------------------------------------------------------------------
export const listCases = (limit = 200) => fetch(`${BASE}/cases?limit=${limit}`).then(j);
export const getCase = (id: string) =>
    fetch(`${BASE}/system/cases/${id}`, { headers: KEY }).then(j);
// Operator-only bulk export: a plain <a href> cannot send the key header, so fetch + blob.
export const downloadCasesCsv = async () => {
    const res = await fetch(`${BASE}/system/cases.csv`, { headers: KEY });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    const url = URL.createObjectURL(await res.blob());
    const a = document.createElement('a');
    a.href = url;
    a.download = 'cases.csv';
    a.click();
    URL.revokeObjectURL(url);
};
export const getAgentTrace = (id: string) =>
    fetch(`${BASE}/system/cases/${id}/agent-trace`, { headers: KEY }).then(j);
export const getCaseHistory = (id: string) =>
    fetch(`${BASE}/cases/${id}/audit`, { headers: KEY }).then(j);
export const runCase = (id: string) =>
    fetch(`${BASE}/cases/${id}/advance`, { method: 'POST', headers: KEY }).then(j);
export const executeCase = (id: string) =>
    fetch(`${BASE}/cases/${id}/execute`, { method: 'POST', headers: KEY }).then(j);
export const verifyCase = (id: string, reference: string) =>
    fetch(`${BASE}/cases/${id}/verify`, {
        method: 'POST',
        headers: JSON_KEY,
        body: JSON.stringify({ external_reference: reference }),
    }).then(j);
export const decideCase = (id: string, decision: 'APPROVE' | 'REJECT', note: string) =>
    fetch(`${BASE}/cases/${id}/human-review`, {
        method: 'POST',
        headers: reviewerHeaders(),
        body: JSON.stringify({ decision, note }),
    }).then(j);

// -- data import ----------------------------------------------------------
// Mutating dataset routes are auth-guarded server-side; send the key.
export const uploadData = (file: File) => {
    const fd = new FormData();
    fd.append('file', file);
    return fetch(`${BASE}/datasets/upload`, { method: 'POST', headers: KEY, body: fd }).then(j);
};
export const analyzeData = (id: string) =>
    fetch(`${BASE}/datasets/${id}/analyze`, { method: 'POST', headers: KEY }).then(j);
export const getImport = (id: string) => fetch(`${BASE}/datasets/${id}`).then(j);
export const getImportStatus = (id: string) =>
    fetch(`${BASE}/datasets/${id}/workflow-status`).then(j);
export const previewData = (id: string, limit = 8) =>
    fetch(`${BASE}/datasets/${id}/preview?limit=${limit}`).then(j);
// declaredDomain: a domain for every row, null to clear one, undefined to leave it as is.
export const confirmColumns = (
    id: string,
    mappings: any[],
    declaredDomain?: RevenueDomain | null,
) =>
    fetch(`${BASE}/datasets/${id}/mapping`, {
        method: 'POST',
        headers: JSON_KEY,
        body: JSON.stringify(
            declaredDomain === undefined
                ? { mappings }
                : { mappings, declared_domain: declaredDomain },
        ),
    }).then(j);
export const getDatasetDomain = (id: string): Promise<DatasetDomainContract> =>
    fetch(`${BASE}/datasets/${id}/domain`).then(j);
export const prepareForCases = (id: string) =>
    fetch(`${BASE}/datasets/${id}/ml-readiness`, { method: 'POST', headers: KEY }).then(j);
export const buildCases = (id: string, maxCases = 100) =>
    fetch(`${BASE}/datasets/${id}/generate-cases`, {
        method: 'POST',
        headers: JSON_KEY,
        body: JSON.stringify({ max_cases: maxCases }),
    }).then(j);
