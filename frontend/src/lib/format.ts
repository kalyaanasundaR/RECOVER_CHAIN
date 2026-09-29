export const money = (n: any, ccy = 'INR') => {
    const v = Number(n ?? 0);
    try {
        return v.toLocaleString('en-IN', {
            style: 'currency',
            currency: ccy,
            maximumFractionDigits: 2,
        });
    } catch {
        return (
            '₹' + v.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
        );
    }
};

export const pct = (n: any) =>
    n === null || n === undefined ? '—' : `${Math.round(Number(n) * 100)}%`;

const amt = (v: any) => (v && typeof v === 'object' ? v.amount : v);
// A {amount, currency} object is shown in its own currency.
export const moneyMaybe = (v: any, ccy = 'INR') =>
    money(amt(v), v && typeof v === 'object' && v.currency ? v.currency : ccy);

// Amounts in several currencies, one per currency. Never added together: RecoverChain
// performs no conversion.
export const moneyByCurrency = (byCcy: Record<string, number>) => {
    const entries = Object.entries(byCcy).sort(([a], [b]) => a.localeCompare(b));
    if (!entries.length) return money(0);
    return entries.map(([ccy, v]) => money(v, ccy)).join(' · ');
};

// why a case's events carry their revenue type (backend DomainBasis)
export const DOMAIN_BASIS: Record<string, string> = {
    DECLARED_BY_SOURCE: 'named by the source system',
    DECLARED_FOR_DATASET: 'declared for the file',
    CATEGORY_COLUMN_MAPPED: 'from the category column you confirmed',
    CATEGORY_COLUMN_RECOGNISED: 'from the file’s category column',
    SCHEMA_EVIDENCE: 'from columns only this kind of data has',
    PAYMENT_FAILURE_EVIDENCE: 'from its payment-failure columns',
    LEGACY_UNRECORDED: 'not recorded (imported before revenue types were tracked)',
};

// a customer's contact permission for one channel
export const PERMISSION: Record<string, string> = {
    ALLOWED: 'Allowed',
    OPTED_OUT: 'Opted out',
    SUPPRESSED: 'Suppressed',
    UNKNOWN: 'No record',
};

// Phase 8B: labels an outcome's evidence source when it is not (yet) real-world —
// production is the only value that renders no badge at all (CaseView.tsx).
export const PROVENANCE_LABEL: Record<string, string> = {
    test: 'test data',
    external_source_simulated: 'simulated external-source data',
    human_confirmed: 'human-confirmed',
    mixed: 'mixed evidence sources',
};

// Phase 12: a documented-methodology confidence bucket, not a fabricated
// statistical confidence interval — see backend application/explainability.py.
export const CONFIDENCE_LABEL: Record<string, string> = {
    HIGH: 'High confidence',
    MEDIUM: 'Medium confidence',
    LOW: 'Low confidence',
};

// plain words for backend enums ------------------------------------------------
export const WHY_FAILED: Record<string, string> = {
    INSUFFICIENT_FUNDS: 'Not enough money in the account',
    NETWORK_FAILURE: 'A temporary network / bank glitch',
    PAYMENT_METHOD_INVALID: 'The card or payment method is no longer valid',
    PAYMENT_FRICTION: 'The customer got stuck at payment',
    MANDATE_FAILURE: 'The recurring-payment authorisation broke',
    UNRESOLVED_DISPUTE: 'There is an open dispute on the invoice',
    MISSED_COMMITMENT: 'The customer missed a promised payment date',
    CONFLICTING_EVIDENCE: 'The signals disagree — needs a look',
    UNKNOWN: 'Cause not clear from the data',
};

export const ACTION: Record<string, string> = {
    RETRY_PAYMENT: 'Retry the payment',
    RETRY_BILLING: 'Retry the charge',
    REQUEST_PAYMENT_METHOD_UPDATE: 'Ask the customer to update their card',
    SEND_PAYMENT_REMINDER: 'Send a payment reminder',
    SEND_CHECKOUT_REMINDER: 'Send a checkout reminder',
    OFFER_CHECKOUT_ASSISTANCE: 'Offer checkout help',
    SEND_SUBSCRIPTION_REMINDER: 'Send a subscription reminder',
    SEND_INVOICE_REMINDER: 'Send an invoice reminder',
    SEND_PAYMENT_LINK: 'Send a payment link',
    ESCALATE_COLLECTION: 'Hand to collections',
    SEND_PROMISE_REMINDER: 'Remind about the promised payment',
    REQUEST_NEW_COMMITMENT: 'Ask for a new payment date',
    ESCALATE_TO_HUMAN: 'Needs human review',
    NO_ACTION_POSSIBLE: 'Nothing to do',
};

// plain names for the real PolicyEngine rule_names
export const RULE_LABEL: Record<string, string> = {
    FinancialAutomatedLimit: 'Automated-amount limit',
    PaymentMaxRetries: 'Retry limit',
    PaymentRetryCooldown: 'Cooling period',
    CommunicationMaxMessages: 'Contact limit',
    HumanEscalationAction: 'Human-approval threshold',
    StopRule: 'Stop rule',
    ConsentCheck: 'Consent',
    ContactPermission: 'Contact permission',
    ActionDomainApplicability: 'Action fits the revenue type',
    EvidenceCheck: 'Evidence present',
};

type Tone = 'green' | 'amber' | 'red' | 'blue' | 'gray' | 'violet';

export function policyVerdict(status?: string): { label: string; tone: Tone; note: string } {
    switch (status) {
        case 'PERMITTED':
            return { label: 'Approved', tone: 'green', note: 'Safe to act on automatically.' };
        case 'WAIT':
            return { label: 'Wait', tone: 'amber', note: 'In a cooling period — try again later.' };
        case 'ESCALATE':
            return {
                label: 'Needs approval',
                tone: 'amber',
                note: 'Above the automated limit — a person must approve.',
            };
        case 'DENIED':
            return { label: 'Blocked', tone: 'red', note: 'A safety rule stops this action.' };
        default:
            return { label: 'Not checked', tone: 'gray', note: '' };
    }
}

export function outcomeVerdict(status?: string): { label: string; tone: Tone } {
    switch (status) {
        case 'FULLY_RECOVERED':
            return { label: 'Recovered', tone: 'green' };
        case 'PARTIALLY_RECOVERED':
            return { label: 'Partially recovered', tone: 'amber' };
        case 'NOT_RECOVERED':
            return { label: 'Not recovered', tone: 'red' };
        case 'PENDING_VERIFICATION':
            return { label: 'Pending', tone: 'blue' };
        case 'CANCELLED':
            return { label: 'Cancelled', tone: 'gray' };
        default:
            return { label: 'Not run', tone: 'gray' };
    }
}

export function riskWord(level?: string): { label: string; tone: Tone } {
    switch (level) {
        case 'CRITICAL':
            return { label: 'Critical', tone: 'red' };
        case 'HIGH':
            return { label: 'High', tone: 'red' };
        case 'MEDIUM':
            return { label: 'Medium', tone: 'amber' };
        case 'LOW':
            return { label: 'Low', tone: 'gray' };
        default:
            return { label: '—', tone: 'gray' };
    }
}

export function confidenceTone(c?: string): Tone {
    return c === 'HIGH' ? 'green' : c === 'MEDIUM' ? 'blue' : c === 'LOW' ? 'amber' : 'gray';
}

export const CATEGORY: Record<string, string> = {
    FAILED_PAYMENT: 'Failed payment',
    CHECKOUT_ABANDONMENT: 'Abandoned checkout',
    FAILED_SUBSCRIPTION: 'Failed subscription',
    OVERDUE_INVOICE: 'Overdue invoice',
    BROKEN_PROMISE: 'Broken promise to pay',
};

// which of the four roles a detected canonical field fills (for step 4)
export const ROLE_OF: Record<string, string> = {
    CUSTOMER_ID: 'Customer',
    ACCOUNT_ID: 'Customer',
    ENTITY_ID: 'Customer',
    AMOUNT: 'Amount',
    BALANCE: 'Amount',
    TIMESTAMP: 'Date',
    SETTLEMENT_DATE: 'Date',
    OUTCOME: 'Result',
    TARGET: 'Result',
    TRANSACTION_ID: 'Transaction ID',
    STATUS: 'Status',
    CURRENCY: 'Currency',
    PAYMENT_METHOD: 'Payment method',
};

// one plain line for each role — what it is, in words anyone gets
export const ROLE_HINT: Record<string, string> = {
    Customer: 'who owes the money',
    Amount: 'how much is at stake',
    Date: 'when the payment failed or was due',
    Result: 'did it get paid, or fail',
};

// the four things every case needs, in order
export const NEEDED_ROLES = ['Customer', 'Amount', 'Date', 'Result'] as const;

// plain, short answers to "what is this?" for the jargon on the connection step
export const PLAIN: Record<string, string> = {
    features:
        'Extra columns the model studies to spot patterns — like the failure reason or how overdue a bill is.',
    heldout:
        'Columns we deliberately hide from the model. Some are just an ID; some would let it "cheat" by seeing the answer.',
    shadowModel:
        'A trained second opinion. It guesses how likely money is to come back — but it never makes the decision. A fixed rulebook does.',
    leak: 'A column that already contains the answer (or something only known afterwards). If the model saw it, its score would be fake.',
    split: 'We train on older rows and test on newer ones, so the score reflects real unseen data — not memorised rows.',
    baseline:
        'A simple built-in estimate used until a trained model is ready, or when the data is too thin to trust one.',
};
