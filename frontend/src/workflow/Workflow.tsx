import { Suspense, lazy, useMemo, useState, useCallback, useEffect } from 'react';
import { Sparkles, RotateCcw } from 'lucide-react';
import { STEPS, Ctx, Action } from './types';
import Rail from './Rail';
import Frame from './Frame';
import ModeSwitch from '../ModeSwitch';
import { useMotionPref } from '../lib/motion';

import S01 from './steps/S01Upload';
import S02 from './steps/S02Review';
import S03 from './steps/S03Quality';
import S04 from './steps/S04Mapping';
import S05 from './steps/S05Connection';
import S06 from './steps/S06Detection';
import S07 from './steps/S07Analysis';
import S08 from './steps/S08Decision';
import S09 from './steps/S10Recovery';
import S10 from './steps/S11Result';

const Backdrop = lazy(() => import('./Backdrop'));
const PANELS = [S01, S02, S03, S04, S05, S06, S07, S08, S09, S10];

// M3a — keep an in-progress run alive across a refresh / accidental navigation.
const SAVE_KEY = 'rc-run';
type Saved = { idx: number; ctx: Ctx };
function loadSaved(): Saved | null {
    try {
        const raw = sessionStorage.getItem(SAVE_KEY);
        if (!raw) return null;
        const s = JSON.parse(raw);
        if (typeof s?.idx === 'number' && s.idx > 0 && s.ctx?.importId) return s;
    } catch {
        /* ignore */
    }
    return null;
}

export default function Workflow() {
    const restored = useMemo(loadSaved, []);
    const [idx, setIdx] = useState(restored?.idx ?? 0);
    const [ctx, setCtx] = useState<Ctx>(restored?.ctx ?? {});
    const [action, setAction] = useState<Action | null>(null);
    const [motion, setMotion] = useMotionPref();

    useEffect(() => {
        try {
            if (idx > 0 && ctx.importId) {
                // Persist the run's position and identifiers, but NOT `snap` — the
                // cached GET /system/cases/{id} response. useSnap only re-fetches when
                // the case id changes, so a restored snapshot was treated as current
                // however old it was, and the Recovery step decides whether to offer
                // "Approve & execute" by reading snap.execution_record. Dropping it
                // makes the restored run re-read the case from the backend, which is
                // the authority. `build` is deliberately kept: S06Detection uses it to
                // avoid re-running generate-cases.
                const { snap: _snap, ...durable } = ctx;
                sessionStorage.setItem(SAVE_KEY, JSON.stringify({ idx, ctx: durable }));
            } else sessionStorage.removeItem(SAVE_KEY);
        } catch {
            /* ignore */
        }
    }, [idx, ctx]);

    const restart = useCallback(() => {
        try {
            sessionStorage.removeItem(SAVE_KEY);
        } catch {
            /* ignore */
        }
        setAction(null);
        setCtx({});
        setIdx(0);
    }, []);

    const patch = useCallback((p: Partial<Ctx>) => setCtx((c) => ({ ...c, ...p })), []);
    const next = useCallback(() => {
        setAction(null);
        setIdx((i) => Math.min(i + 1, PANELS.length - 1));
    }, []);
    const back = useCallback(() => {
        setAction(null);
        setIdx((i) => Math.max(i - 1, 0));
    }, []);

    const Panel = PANELS[idx];
    const progress = idx / (PANELS.length - 1);
    const pct = useMemo(() => Math.round(progress * 100), [progress]);

    return (
        <div
            className={`relative min-h-screen overflow-hidden text-[--ink] ${motion ? '' : 'bg-stage'}`}
        >
            {motion && (
                <Suspense fallback={null}>
                    <div className="pointer-events-none absolute inset-0 -z-10 opacity-45">
                        <Backdrop progress={progress} />
                    </div>
                    {/* scrim keeps text crisp over the motion layer */}
                    <div className="pointer-events-none absolute inset-0 -z-10 bg-gradient-to-b from-[--bg]/70 via-[--bg]/40 to-[--bg]/80" />
                </Suspense>
            )}

            <header className="sticky top-0 z-20 border-b border-[--line] bg-[--bg]/80 backdrop-blur">
                <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-3">
                    <div className="flex items-center gap-3">
                        <span className="grid h-7 w-7 place-items-center rounded-md bg-[--accent] text-xs font-bold text-white">
                            RC
                        </span>
                        <ModeSwitch />
                    </div>
                    <div className="flex items-center gap-4 text-xs text-[--faint]">
                        <button
                            onClick={() => setMotion(!motion)}
                            title={motion ? 'Turn motion off' : 'Turn motion on'}
                            className={`inline-flex items-center gap-1.5 rounded-md px-2 py-1 ${motion ? 'text-sky-300' : 'text-[--faint] hover:text-[--muted]'}`}
                        >
                            <Sparkles size={13} /> motion
                        </button>
                        {idx > 0 && (
                            <button
                                onClick={restart}
                                title="Start over"
                                className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-[--faint] hover:text-[--muted]"
                            >
                                <RotateCcw size={13} /> restart
                            </button>
                        )}
                        <span>Step {STEPS[idx].n} / 10</span>
                    </div>
                </div>
                <div className="h-0.5 bg-[--line]">
                    <div
                        className="h-full bg-[--accent] transition-all duration-500"
                        style={{ width: `${pct}%` }}
                    />
                </div>
            </header>

            <div className="mx-auto grid max-w-6xl grid-cols-1 gap-10 px-6 py-10 md:grid-cols-[220px_1fr]">
                <div className="md:sticky md:top-24 md:self-start">
                    <Rail idx={idx} />
                    {idx > 0 && (
                        <button
                            onClick={back}
                            className="mt-6 text-xs text-[--faint] hover:text-[--muted]"
                        >
                            ← back
                        </button>
                    )}
                </div>

                <Frame idx={idx} action={action} motion={motion}>
                    <Panel ctx={ctx} patch={patch} next={next} back={back} setAction={setAction} />
                </Frame>
            </div>
        </div>
    );
}
